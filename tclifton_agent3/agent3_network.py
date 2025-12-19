"""
Agent 3: PPO Actor-Critic Network with Policy-Pruned 2-ply Search

Architecture (from project spec):
- Same ResNet backbone as Agent 2 (shared between actor and critic)
- Policy head (actor): outputs action probabilities
- Value head (critic): outputs state value
- Training: PPO (Proximal Policy Optimization)
- Evaluation: Policy prunes to top-K moves, then 2-ply search on those

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
import sys
sys.path.append("..")

from backgammon_engine import (
    _new_game, _roll_dice, _actions, _apply_move, _reward,
    _to_canonical, W_BAR, B_BAR, W_OFF, B_OFF, NUM_POINTS, NUM_CHECKERS,
    int8
)

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

BOARD_LENGTH = 24
CONV_INPUT_CHANNELS = 15  # Feature planes per point
AUX_INPUT_SIZE = 6        # Auxiliary features
FILTERS = 128             # Conv channels
NUM_RESIDUAL_BLOCKS = 9   # As per spec (same as Agent 2)
MAX_SCORE = 3.0           # Backgammon score range [-3, +3]

# Pre-compute all 21 dice rolls and probabilities
DICE_ROLLS = []
DICE_PROBS = []
for r1 in range(1, 7):
    for r2 in range(1, r1 + 1):
        DICE_ROLLS.append(np.array([r1, r2], dtype=np.int8))
        DICE_PROBS.append(1/36 if r1 == r2 else 2/36)
DICE_PROBS = np.array(DICE_PROBS, dtype=np.float32)


# =============================================================================
# FEATURE ENCODING (same as Agent 2)
# =============================================================================

def encode_board_features_batch(states: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Batch encode board states into feature planes."""
    batch_size = len(states)
    board_features = np.zeros((batch_size, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
    aux_features = np.zeros((batch_size, AUX_INPUT_SIZE), dtype=np.float32)

    for b, state in enumerate(states):
        for i in range(BOARD_LENGTH):
            point_idx = i + 1
            count = int(state[point_idx])

            if count == 0:
                board_features[b, i, 0] = 1.0
            elif count > 0:  # White
                idx = min(count, 6) * 2 - 1
                board_features[b, i, idx] = 1.0
                if count > 6:
                    board_features[b, i, 13] = (count - 6) / 9.0
            else:  # Black (negative)
                black_count = -count
                idx = min(black_count, 6) * 2
                board_features[b, i, idx] = 1.0
                if black_count > 6:
                    board_features[b, i, 14] = (black_count - 6) / 9.0

        # Auxiliary features
        white_bar = int(state[W_BAR])
        black_bar = -int(state[B_BAR])
        aux_features[b, 0] = 1.0 if white_bar > 0 else 0.0
        aux_features[b, 1] = 1.0 if black_bar > 0 else 0.0
        aux_features[b, 2] = white_bar / 15.0
        aux_features[b, 3] = black_bar / 15.0
        aux_features[b, 4] = int(state[W_OFF]) / 15.0
        aux_features[b, 5] = abs(int(state[B_OFF])) / 15.0

    return board_features, aux_features


def encode_single_state(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Encode a single board state."""
    board_features = np.zeros((BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
    aux_features = np.zeros(AUX_INPUT_SIZE, dtype=np.float32)

    for i in range(BOARD_LENGTH):
        point_idx = i + 1
        count = int(state[point_idx])

        if count == 0:
            board_features[i, 0] = 1.0
        elif count > 0:
            idx = min(count, 6) * 2 - 1
            board_features[i, idx] = 1.0
            if count > 6:
                board_features[i, 13] = (count - 6) / 9.0
        else:
            black_count = -count
            idx = min(black_count, 6) * 2
            board_features[i, idx] = 1.0
            if black_count > 6:
                board_features[i, 14] = (black_count - 6) / 9.0

    white_bar = int(state[W_BAR])
    black_bar = -int(state[B_BAR])
    aux_features[0] = 1.0 if white_bar > 0 else 0.0
    aux_features[1] = 1.0 if black_bar > 0 else 0.0
    aux_features[2] = white_bar / 15.0
    aux_features[3] = black_bar / 15.0
    aux_features[4] = int(state[W_OFF]) / 15.0
    aux_features[5] = abs(int(state[B_OFF])) / 15.0

    return board_features, aux_features


# =============================================================================
# NEURAL NETWORK
# =============================================================================

class ResidualBlockV2(nn.Module):
    """ResNet v2 block with pre-activation (LayerNorm + ReLU before conv)."""
    
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.ln1 = nn.LayerNorm([channels, BOARD_LENGTH])
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.ln2 = nn.LayerNorm([channels, BOARD_LENGTH])
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding='same')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.ln1(x))
        out = self.conv1(out)
        out = F.relu(self.ln2(out))
        out = self.conv2(out)
        return residual + out


class PPOActorCriticNet(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Architecture:
    - Shared backbone: Conv1d(15->128, kernel=7) + 9 ResNet v2 blocks
    - Global average pooling + concatenate aux features
    - Policy head (actor): Dense -> softmax over moves
    - Value head (critic): Dense -> scalar value
    
    The policy head outputs logits for a FIXED number of action slots.
    During inference, we mask out illegal actions.
    """
    
    def __init__(self, 
                 num_res_blocks: int = NUM_RESIDUAL_BLOCKS, 
                 filters: int = FILTERS,
                 max_actions: int = 512):
        """
        Args:
            num_res_blocks: Number of residual blocks (9 per spec)
            filters: Number of conv filters (128 per spec)
            max_actions: Maximum number of legal moves to support
        """
        super().__init__()
        
        self.max_actions = max_actions
        
        # Shared backbone (same as Agent 2)
        self.initial_conv = nn.Conv1d(CONV_INPUT_CHANNELS, filters, kernel_size=7, padding='same')
        self.res_blocks = nn.ModuleList([
            ResidualBlockV2(filters, kernel_size=3) for _ in range(num_res_blocks)
        ])
        
        # Feature dimension after global pooling
        feature_dim = filters + AUX_INPUT_SIZE  # 128 + 6 = 134
        
        # Value head (critic) - same as Agent 2's value network
        self.value_fc1 = nn.Linear(feature_dim, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
        # Policy head (actor) - outputs logits for each possible action slot
        # We use a fixed output size and mask illegal actions
        self.policy_fc1 = nn.Linear(feature_dim, 128)
        self.policy_fc2 = nn.Linear(128, max_actions)
    
    def forward_features(self, board_features: torch.Tensor, 
                         aux_features: torch.Tensor) -> torch.Tensor:
        """Extract shared features from observation."""
        # Conv expects (batch, channels, length)
        x = board_features.transpose(1, 2)  # (batch, 15, 24)
        
        x = F.relu(self.initial_conv(x))
        
        for block in self.res_blocks:
            x = block(x)
        
        # Global average pooling
        x = x.mean(dim=2)  # (batch, filters)
        
        # Concatenate auxiliary features
        x = torch.cat([x, aux_features], dim=1)  # (batch, filters + aux)
        
        return x
    
    def forward(self, board_features: torch.Tensor, aux_features: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both policy and value.
        
        Args:
            board_features: (batch, 24, 15) board feature planes
            aux_features: (batch, 6) auxiliary features
            action_mask: (batch, max_actions) boolean mask, True for legal actions
        
        Returns:
            policy_logits: (batch, max_actions) raw logits (masked if mask provided)
            value: (batch, 1) state value
        """
        features = self.forward_features(board_features, aux_features)
        
        # Value head
        value = F.relu(self.value_fc1(features))
        value = torch.tanh(self.value_fc2(value)) * MAX_SCORE
        
        # Policy head
        policy_logits = F.relu(self.policy_fc1(features))
        policy_logits = self.policy_fc2(policy_logits)
        
        # Mask illegal actions with large negative value
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(~action_mask, float('-inf'))
        
        return policy_logits, value
    
    def get_value(self, board_features: torch.Tensor, 
                  aux_features: torch.Tensor) -> torch.Tensor:
        """Get only the value (for critic updates)."""
        features = self.forward_features(board_features, aux_features)
        value = F.relu(self.value_fc1(features))
        value = torch.tanh(self.value_fc2(value)) * MAX_SCORE
        return value
    
    def get_policy(self, board_features: torch.Tensor, aux_features: torch.Tensor,
                   action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get only the policy logits (for actor updates)."""
        features = self.forward_features(board_features, aux_features)
        policy_logits = F.relu(self.policy_fc1(features))
        policy_logits = self.policy_fc2(policy_logits)
        
        if action_mask is not None:
            policy_logits = policy_logits.masked_fill(~action_mask, float('-inf'))
        
        return policy_logits


# =============================================================================
# PPO AGENT
# =============================================================================

class PPOAgent:
    """
    PPO Agent for Backgammon.
    
    Key features:
    - Uses policy network to get action probabilities
    - Can use policy to prune to top-K moves, then 2-ply on those (policy-pruned search)
    - Trained with PPO objective
    """
    
    def __init__(self, 
                 lr: float = 3e-4,
                 gamma: float = 1.0,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_actions: int = 512,
                 device: str = None):
        """
        Args:
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_actions: Maximum number of action slots
            device: 'cuda' or 'cpu'
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_actions = max_actions
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Create network
        self.network = PPOActorCriticNet(max_actions=max_actions).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        # For tracking
        self.training = True
    
    def _get_action_mask(self, num_moves: int) -> torch.Tensor:
        """Create action mask for given number of legal moves."""
        mask = torch.zeros(self.max_actions, dtype=torch.bool, device=self.device)
        mask[:num_moves] = True
        return mask
    
    @torch.no_grad()
    def get_value(self, state: np.ndarray) -> float:
        """Get value estimate for a state (canonical form)."""
        self.network.eval()
        board_feat, aux_feat = encode_single_state(state)
        board_tensor = torch.from_numpy(board_feat).unsqueeze(0).to(self.device)
        aux_tensor = torch.from_numpy(aux_feat).unsqueeze(0).to(self.device)
        value = self.network.get_value(board_tensor, aux_tensor)
        return float(value.item())
    
    @torch.no_grad()
    def batch_value(self, canonical_states: List[np.ndarray]) -> np.ndarray:
        """Batch value computation."""
        if len(canonical_states) == 0:
            return np.array([], dtype=np.float32)
        
        self.network.eval()
        board_np, aux_np = encode_board_features_batch(canonical_states)
        board_tensor = torch.from_numpy(board_np).to(self.device)
        aux_tensor = torch.from_numpy(aux_np).to(self.device)
        values = self.network.get_value(board_tensor, aux_tensor)
        return values.cpu().numpy().flatten()
    
    @torch.no_grad()
    def get_action_probs(self, state: np.ndarray, num_moves: int) -> np.ndarray:
        """
        Get action probabilities for a state.
        
        Args:
            state: Canonical state
            num_moves: Number of legal moves
        
        Returns:
            probs: (num_moves,) action probabilities
        """
        self.network.eval()
        board_feat, aux_feat = encode_single_state(state)
        board_tensor = torch.from_numpy(board_feat).unsqueeze(0).to(self.device)
        aux_tensor = torch.from_numpy(aux_feat).unsqueeze(0).to(self.device)
        
        action_mask = self._get_action_mask(num_moves).unsqueeze(0)
        logits, _ = self.network(board_tensor, aux_tensor, action_mask)
        
        probs = F.softmax(logits[0, :num_moves], dim=0)
        return probs.cpu().numpy()
    
    def select_action(self, state: np.ndarray, player: int, dice: np.ndarray,
                      deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Select action using the policy.
        
        Args:
            state: Current game state
            player: Current player (+1 or -1)
            dice: Current dice roll
            deterministic: If True, select argmax; else sample
        
        Returns:
            action_idx: Index of selected action in legal moves list
            log_prob: Log probability of selected action
            value: Value estimate of current state
        """
        moves, _ = _actions(state, player, dice)
        
        if len(moves) == 0:
            return 0, 0.0, 0.0
        if len(moves) == 1:
            canonical = _to_canonical(state, player)
            value = self.get_value(canonical)
            return 0, 0.0, value
        
        # Get canonical state
        canonical = _to_canonical(state, player)
        
        self.network.eval()
        board_feat, aux_feat = encode_single_state(canonical)
        board_tensor = torch.from_numpy(board_feat).unsqueeze(0).to(self.device)
        aux_tensor = torch.from_numpy(aux_feat).unsqueeze(0).to(self.device)
        
        action_mask = self._get_action_mask(len(moves)).unsqueeze(0)
        
        with torch.no_grad():
            logits, value = self.network(board_tensor, aux_tensor, action_mask)
        
        # Get action probabilities
        probs = F.softmax(logits[0, :len(moves)], dim=0)
        
        if deterministic:
            action_idx = torch.argmax(probs).item()
        else:
            # Sample from distribution
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()
        
        log_prob = torch.log(probs[action_idx] + 1e-8).item()
        
        return action_idx, log_prob, value.item()
    
    def select_move_policy_pruned_2ply(self, state: np.ndarray, player: int, 
                                        dice: np.ndarray, top_k: int = 5) -> int:
        """
        Policy-pruned 2-ply search.
        
        1. Use policy to get top-K most likely moves
        2. Run 2-ply expectimax only on those K moves
        3. Return the best one
        
        This is much faster than full 2-ply when there are many legal moves.
        
        Args:
            state: Current game state
            player: Current player
            dice: Current dice roll
            top_k: Number of moves to consider (default 5)
        
        Returns:
            action_idx: Index of selected action
        """
        moves, afterstates = _actions(state, player, dice)
        
        if len(moves) == 0:
            return 0
        if len(moves) == 1:
            return 0
        
        # If fewer moves than top_k, just evaluate all
        if len(moves) <= top_k:
            pruned_indices = list(range(len(moves)))
        else:
            # Get policy probabilities and select top-K
            canonical = _to_canonical(state, player)
            probs = self.get_action_probs(canonical, len(moves))
            pruned_indices = np.argsort(probs)[-top_k:][::-1]  # Top K by probability
        
        # Now run 2-ply search only on pruned moves
        best_value = float('-inf')
        best_idx = pruned_indices[0]
        
        for m_idx in pruned_indices:
            afterstate = afterstates[m_idx]
            expected_value = 0.0
            
            # For each possible opponent dice roll
            for d_idx, opp_dice in enumerate(DICE_ROLLS):
                opp_moves, opp_afterstates = _actions(afterstate, -player, opp_dice)
                
                if len(opp_moves) == 0:
                    # Opponent can't move, evaluate afterstate
                    canonical = _to_canonical(afterstate, player)
                    val = self.get_value(canonical)
                else:
                    # Find opponent's best response (minimizes our value)
                    opp_canonicals = [_to_canonical(opp_afterstates[i], player) 
                                      for i in range(len(opp_moves))]
                    opp_values = self.batch_value(opp_canonicals)
                    val = np.min(opp_values)
                
                expected_value += DICE_PROBS[d_idx] * val
            
            if expected_value > best_value:
                best_value = expected_value
                best_idx = m_idx
        
        return best_idx
    
    def select_move_1ply(self, state: np.ndarray, player: int, 
                         dice: np.ndarray) -> int:
        """Simple 1-ply greedy search using value network."""
        moves, afterstates = _actions(state, player, dice)
        
        if len(moves) <= 1:
            return 0
        
        canonical_states = [_to_canonical(afterstates[i], player) for i in range(len(moves))]
        values = self.batch_value(canonical_states)
        return int(np.argmax(values))
    
    def compute_gae(self, rewards: List[float], values: List[float], 
                    dones: List[bool], next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value of final next state
        
        Returns:
            advantages: GAE advantages
            returns: Value targets (advantages + values)
        """
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + self.gamma * next_val - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            
            advantages[t] = last_gae
        
        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns
    
    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Agent 3 PPO Network...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create network
    net = PPOActorCriticNet(max_actions=512).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    batch_size = 4
    board = torch.randn(batch_size, BOARD_LENGTH, CONV_INPUT_CHANNELS).to(device)
    aux = torch.randn(batch_size, AUX_INPUT_SIZE).to(device)
    action_mask = torch.ones(batch_size, 512, dtype=torch.bool).to(device)
    action_mask[:, 10:] = False  # Only 10 legal actions
    
    policy_logits, value = net(board, aux, action_mask)
    print(f"\nForward pass:")
    print(f"  Policy logits shape: {policy_logits.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Value range: [{value.min().item():.2f}, {value.max().item():.2f}]")
    
    # Test agent
    print("\nTesting PPO Agent...")
    agent = PPOAgent(device=device)
    
    # Test with real game state
    player, dice, state = _new_game()
    moves, _ = _actions(state, player, dice)
    print(f"Initial state: player={player}, dice={dice}, num_moves={len(moves)}")
    
    # Test action selection
    action_idx, log_prob, value = agent.select_action(state, player, dice)
    print(f"Selected action: idx={action_idx}, log_prob={log_prob:.4f}, value={value:.4f}")
    
    # Test policy-pruned 2-ply
    best_idx = agent.select_move_policy_pruned_2ply(state, player, dice, top_k=5)
    print(f"Policy-pruned 2-ply selected: idx={best_idx}")
    
    print("\n? All tests passed!")
