"""
Agent 4: Stochastic MuZero Network Architecture

This implements the neural network components for Stochastic MuZero:
- Representation Network: h(o) -> s (observation to latent state)
- Dynamics Network: g(s, a, c) -> s', r (latent transition with chance outcome)
- Prediction Network: f(s) -> p, v (policy and value from latent state)
- Afterstate Dynamics: ga(s, a) -> sa (deterministic afterstate)
- Chance Encoder: gc(sa, c) -> s' (apply chance outcome)

Reference: Antonoglou-Schrittwieser-Ozair-Hubert-Silver 2022
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import sys
sys.path.append("..")

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

BOARD_LENGTH = 24
CONV_INPUT_CHANNELS = 15  # Feature planes per point
AUX_INPUT_SIZE = 6        # Auxiliary features
FILTERS = 128             # Conv channels
NUM_RESIDUAL_BLOCKS = 9   # ResNet blocks (same as Agent 2)
LATENT_SIZE = 256         # Latent state dimension
MAX_SCORE = 3.0           # Backgammon score range [-3, +3]
NUM_DICE_OUTCOMES = 21    # 21 possible dice rolls
POLICY_SIZE = 25 * 25     # 25x25 grid for source-destination submoves


# =============================================================================
# FEATURE ENCODING (same as Agent 2)
# =============================================================================

def encode_board_features(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode the board state into feature planes.
    
    Feature planes (15 total for the 24 points):
    0: Empty point
    1-2: Blot (1 checker) - white/black
    3-4: Made point (2 checkers)
    5-6: Builder (3 checkers)
    7-8: Basic anchor (4 checkers)
    9-10: Deep anchor (5 checkers)
    11-12: Permanent anchor (6 checkers)
    13-14: Overflow (>6 checkers, normalized)
    
    Auxiliary features (6 total):
    0-1: Bar indicator (binary)
    2-3: Bar count (normalized)
    4-5: Borne-off count (normalized)
    """
    board_features = np.zeros((BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
    aux_features = np.zeros(AUX_INPUT_SIZE, dtype=np.float32)
    
    for i in range(BOARD_LENGTH):
        point_idx = i + 1
        count = state[point_idx]
        
        if count == 0:
            board_features[i, 0] = 1.0
        elif count > 0:  # White
            idx = min(count, 6) * 2 - 1
            board_features[i, idx] = 1.0
            if count > 6:
                board_features[i, 13] = (count - 6) / 9.0
        else:  # Black (negative)
            black_count = abs(count)
            idx = min(black_count, 6) * 2
            board_features[i, idx] = 1.0
            if black_count > 6:
                board_features[i, 14] = (black_count - 6) / 9.0
    
    # Auxiliary features
    W_BAR, B_BAR, W_OFF, B_OFF = 0, 25, 26, 27
    
    # Bar indicators
    aux_features[0] = 1.0 if state[W_BAR] > 0 else 0.0
    aux_features[1] = 1.0 if state[B_BAR] < 0 else 0.0
    
    # Bar counts (normalized)
    aux_features[2] = state[W_BAR] / 15.0
    aux_features[3] = abs(state[B_BAR]) / 15.0
    
    # Borne-off counts (normalized)
    aux_features[4] = state[W_OFF] / 15.0
    aux_features[5] = abs(state[B_OFF]) / 15.0
    
    return board_features, aux_features


def encode_board_features_batch(states: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Batch version of encode_board_features."""
    batch_size = len(states)
    board_features = np.zeros((batch_size, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
    aux_features = np.zeros((batch_size, AUX_INPUT_SIZE), dtype=np.float32)
    
    for b, state in enumerate(states):
        board_features[b], aux_features[b] = encode_board_features(state)
    
    return board_features, aux_features


# =============================================================================
# RESIDUAL BLOCKS
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


class ResidualBlockFC(nn.Module):
    """Fully connected residual block for latent space."""
    
    def __init__(self, size: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(size)
        self.fc1 = nn.Linear(size, size)
        self.ln2 = nn.LayerNorm(size)
        self.fc2 = nn.Linear(size, size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.ln1(x))
        out = self.fc1(out)
        out = F.relu(self.ln2(out))
        out = self.fc2(out)
        return residual + out


# =============================================================================
# REPRESENTATION NETWORK: h(observation) -> latent_state
# =============================================================================

class RepresentationNetwork(nn.Module):
    """
    Encodes the raw observation (board state) into a latent representation.
    Uses the same ConvNet/ResNet backbone as Agent 2.
    """
    
    def __init__(self, num_res_blocks: int = NUM_RESIDUAL_BLOCKS, filters: int = FILTERS):
        super().__init__()
        
        # Initial convolution
        self.initial_conv = nn.Conv1d(CONV_INPUT_CHANNELS, filters, kernel_size=7, padding='same')
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlockV2(filters, kernel_size=3) for _ in range(num_res_blocks)
        ])
        
        # Final projection to latent space
        self.fc1 = nn.Linear(filters + AUX_INPUT_SIZE, LATENT_SIZE)
        self.ln_out = nn.LayerNorm(LATENT_SIZE)
    
    def forward(self, board_features: torch.Tensor, aux_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            board_features: (batch, 24, 15) board feature planes
            aux_features: (batch, 6) auxiliary features
        Returns:
            latent_state: (batch, LATENT_SIZE)
        """
        # Conv expects (batch, channels, length)
        x = board_features.transpose(1, 2)  # (batch, 15, 24)
        
        x = F.relu(self.initial_conv(x))
        
        for block in self.res_blocks:
            x = block(x)
        
        # Global average pooling
        x = x.mean(dim=2)  # (batch, filters)
        
        # Concatenate auxiliary features
        x = torch.cat([x, aux_features], dim=1)  # (batch, filters + aux)
        
        # Project to latent space
        x = F.relu(self.fc1(x))
        x = self.ln_out(x)
        
        return x


# =============================================================================
# DYNAMICS NETWORK: g(latent_state, action, chance) -> next_latent_state, reward
# =============================================================================

class DynamicsNetwork(nn.Module):
    """
    Predicts the next latent state and reward given:
    - Current latent state
    - Action (encoded as one-hot or embedding)
    - Chance outcome (dice roll, encoded as one-hot)
    
    For Stochastic MuZero, we split this into:
    1. Afterstate dynamics: ga(s, a) -> sa
    2. Chance outcome: gc(sa, c) -> s'
    """
    
    def __init__(self, num_actions: int = POLICY_SIZE, num_chance_outcomes: int = NUM_DICE_OUTCOMES):
        super().__init__()
        
        self.num_actions = num_actions
        self.num_chance_outcomes = num_chance_outcomes
        
        # Action embedding
        self.action_embedding = nn.Embedding(num_actions, 64)
        
        # Afterstate network: s + a -> sa
        self.afterstate_fc1 = nn.Linear(LATENT_SIZE + 64, LATENT_SIZE)
        self.afterstate_res = nn.ModuleList([ResidualBlockFC(LATENT_SIZE) for _ in range(2)])
        self.afterstate_ln = nn.LayerNorm(LATENT_SIZE)
        
        # Chance embedding
        self.chance_embedding = nn.Embedding(num_chance_outcomes, 32)
        
        # Chance transition network: sa + c -> s'
        self.chance_fc1 = nn.Linear(LATENT_SIZE + 32, LATENT_SIZE)
        self.chance_res = nn.ModuleList([ResidualBlockFC(LATENT_SIZE) for _ in range(2)])
        self.chance_ln = nn.LayerNorm(LATENT_SIZE)
        
        # Reward prediction (from afterstate)
        self.reward_fc1 = nn.Linear(LATENT_SIZE, 64)
        self.reward_fc2 = nn.Linear(64, 1)
    
    def afterstate(self, latent_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute afterstate given state and action.
        
        Args:
            latent_state: (batch, LATENT_SIZE)
            action: (batch,) action indices
        Returns:
            afterstate: (batch, LATENT_SIZE)
        """
        action_emb = self.action_embedding(action)  # (batch, 64)
        x = torch.cat([latent_state, action_emb], dim=1)
        x = F.relu(self.afterstate_fc1(x))
        for block in self.afterstate_res:
            x = block(x)
        return self.afterstate_ln(x)
    
    def chance_transition(self, afterstate: torch.Tensor, chance: torch.Tensor) -> torch.Tensor:
        """
        Apply chance outcome to afterstate.
        
        Args:
            afterstate: (batch, LATENT_SIZE)
            chance: (batch,) chance outcome indices (0-20 for dice rolls)
        Returns:
            next_state: (batch, LATENT_SIZE)
        """
        chance_emb = self.chance_embedding(chance)  # (batch, 32)
        x = torch.cat([afterstate, chance_emb], dim=1)
        x = F.relu(self.chance_fc1(x))
        for block in self.chance_res:
            x = block(x)
        return self.chance_ln(x)
    
    def predict_reward(self, afterstate: torch.Tensor) -> torch.Tensor:
        """Predict reward from afterstate."""
        x = F.relu(self.reward_fc1(afterstate))
        return torch.tanh(self.reward_fc2(x)) * MAX_SCORE
    
    def forward(self, latent_state: torch.Tensor, action: torch.Tensor, 
                chance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full dynamics: state + action + chance -> next_state, reward
        """
        afterstate = self.afterstate(latent_state, action)
        reward = self.predict_reward(afterstate)
        next_state = self.chance_transition(afterstate, chance)
        return next_state, reward


# =============================================================================
# PREDICTION NETWORK: f(latent_state) -> policy, value
# =============================================================================

class PredictionNetwork(nn.Module):
    """
    Predicts policy and value from a latent state.
    
    Policy: 25x25 grid of source-destination pairs for submoves
    Value: Scalar in [-3, +3]
    """
    
    def __init__(self):
        super().__init__()
        
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(LATENT_SIZE, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            ResidualBlockFC(256),
            ResidualBlockFC(256),
        )
        
        # Policy head
        self.policy_fc1 = nn.Linear(256, 256)
        self.policy_fc2 = nn.Linear(256, POLICY_SIZE)  # 25x25 = 625
        
        # Value head
        self.value_fc1 = nn.Linear(256, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def forward(self, latent_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent_state: (batch, LATENT_SIZE)
        Returns:
            policy_logits: (batch, 625) raw logits for 25x25 submove grid
            value: (batch, 1) value in [-3, +3]
        """
        x = self.trunk(latent_state)
        
        # Policy
        policy = F.relu(self.policy_fc1(x))
        policy_logits = self.policy_fc2(policy)  # (batch, 625)
        
        # Value
        value = F.relu(self.value_fc1(x))
        value = torch.tanh(self.value_fc2(value)) * MAX_SCORE
        
        return policy_logits, value


# =============================================================================
# CHANCE OUTCOME PREDICTION (for planning)
# =============================================================================

class ChancePredictor(nn.Module):
    """
    Predicts the distribution over chance outcomes (dice rolls) from afterstate.
    In backgammon, dice are random, so this just outputs the known prior.
    But the network can learn to weight them based on game state.
    """
    
    def __init__(self):
        super().__init__()
        # The true dice distribution is fixed, but we include this for completeness
        # In practice for backgammon we'll use the known probabilities
        self.fc = nn.Linear(LATENT_SIZE, NUM_DICE_OUTCOMES)
        
        # Pre-compute the true dice probabilities
        self.register_buffer('dice_probs', self._compute_dice_probs())
    
    def _compute_dice_probs(self) -> torch.Tensor:
        """Compute true dice roll probabilities."""
        probs = []
        for r1 in range(1, 7):
            for r2 in range(1, r1 + 1):
                if r1 == r2:
                    probs.append(1/36)  # Doubles
                else:
                    probs.append(2/36)  # Non-doubles
        return torch.tensor(probs, dtype=torch.float32)
    
    def forward(self, afterstate: torch.Tensor, use_true_probs: bool = True) -> torch.Tensor:
        """
        Args:
            afterstate: (batch, LATENT_SIZE)
            use_true_probs: If True, return known dice probabilities
        Returns:
            probs: (batch, 21) probability distribution over dice outcomes
        """
        if use_true_probs:
            batch_size = afterstate.size(0)
            return self.dice_probs.unsqueeze(0).expand(batch_size, -1)
        else:
            logits = self.fc(afterstate)
            return F.softmax(logits, dim=-1)


# =============================================================================
# COMPLETE MUZERO NETWORK
# =============================================================================

class StochasticMuZeroNetwork(nn.Module):
    """
    Complete Stochastic MuZero network combining all components.
    """
    
    def __init__(self):
        super().__init__()
        
        self.representation = RepresentationNetwork()
        self.dynamics = DynamicsNetwork()
        self.prediction = PredictionNetwork()
        self.chance_predictor = ChancePredictor()
    
    def initial_inference(self, board_features: torch.Tensor, 
                          aux_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initial inference from raw observation.
        
        Args:
            board_features: (batch, 24, 15)
            aux_features: (batch, 6)
        Returns:
            latent_state: (batch, LATENT_SIZE)
            policy_logits: (batch, 625)
            value: (batch, 1)
        """
        latent_state = self.representation(board_features, aux_features)
        policy_logits, value = self.prediction(latent_state)
        return latent_state, policy_logits, value
    
    def recurrent_inference(self, latent_state: torch.Tensor, action: torch.Tensor,
                            chance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Recurrent inference in latent space.
        
        Args:
            latent_state: (batch, LATENT_SIZE)
            action: (batch,) action indices
            chance: (batch,) chance outcome indices
        Returns:
            next_latent_state: (batch, LATENT_SIZE)
            reward: (batch, 1)
            policy_logits: (batch, 625)
            value: (batch, 1)
        """
        next_state, reward = self.dynamics(latent_state, action, chance)
        policy_logits, value = self.prediction(next_state)
        return next_state, reward, policy_logits, value
    
    def get_afterstate(self, latent_state: torch.Tensor, 
                       action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get afterstate and chance distribution.
        
        Args:
            latent_state: (batch, LATENT_SIZE)
            action: (batch,) action indices
        Returns:
            afterstate: (batch, LATENT_SIZE)
            chance_probs: (batch, 21)
        """
        afterstate = self.dynamics.afterstate(latent_state, action)
        chance_probs = self.chance_predictor(afterstate, use_true_probs=True)
        return afterstate, chance_probs
    
    def apply_chance(self, afterstate: torch.Tensor, 
                     chance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply chance outcome to afterstate.
        
        Args:
            afterstate: (batch, LATENT_SIZE)
            chance: (batch,) chance outcome indices
        Returns:
            next_state: (batch, LATENT_SIZE)
            policy_logits: (batch, 625)
            value: (batch, 1)
        """
        next_state = self.dynamics.chance_transition(afterstate, chance)
        policy_logits, value = self.prediction(next_state)
        return next_state, policy_logits, value


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def action_to_index(source: int, dest: int) -> int:
    """Convert (source, dest) pair to action index."""
    return source * 25 + dest


def index_to_action(index: int) -> Tuple[int, int]:
    """Convert action index to (source, dest) pair."""
    source = index // 25
    dest = index % 25
    return source, dest


def dice_roll_to_index(r1: int, r2: int) -> int:
    """
    Convert dice roll to index (0-20).
    Dice are sorted: r1 >= r2.
    """
    idx = 0
    for d1 in range(1, 7):
        for d2 in range(1, d1 + 1):
            if d1 == r1 and d2 == r2:
                return idx
            idx += 1
    raise ValueError(f"Invalid dice roll: ({r1}, {r2})")


def index_to_dice_roll(idx: int) -> Tuple[int, int]:
    """Convert index to dice roll."""
    i = 0
    for r1 in range(1, 7):
        for r2 in range(1, r1 + 1):
            if i == idx:
                return r1, r2
            i += 1
    raise ValueError(f"Invalid dice index: {idx}")


# All 21 dice outcomes with probabilities
DICE_OUTCOMES = []
DICE_PROBS_NP = []
for r1 in range(1, 7):
    for r2 in range(1, r1 + 1):
        DICE_OUTCOMES.append((r1, r2))
        DICE_PROBS_NP.append(1/36 if r1 == r2 else 2/36)
DICE_PROBS_NP = np.array(DICE_PROBS_NP, dtype=np.float32)


if __name__ == "__main__":
    # Test the networks
    print("Testing Stochastic MuZero Network...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create network
    net = StochasticMuZeroNetwork().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test initial inference
    batch_size = 4
    board = torch.randn(batch_size, BOARD_LENGTH, CONV_INPUT_CHANNELS).to(device)
    aux = torch.randn(batch_size, AUX_INPUT_SIZE).to(device)
    
    latent, policy, value = net.initial_inference(board, aux)
    print(f"Initial inference:")
    print(f"  Latent shape: {latent.shape}")
    print(f"  Policy shape: {policy.shape}")
    print(f"  Value shape: {value.shape}")
    
    # Test recurrent inference
    action = torch.randint(0, POLICY_SIZE, (batch_size,)).to(device)
    chance = torch.randint(0, NUM_DICE_OUTCOMES, (batch_size,)).to(device)
    
    next_latent, reward, next_policy, next_value = net.recurrent_inference(latent, action, chance)
    print(f"\nRecurrent inference:")
    print(f"  Next latent shape: {next_latent.shape}")
    print(f"  Reward shape: {reward.shape}")
    print(f"  Next policy shape: {next_policy.shape}")
    print(f"  Next value shape: {next_value.shape}")
    
    # Test afterstate and chance
    afterstate, chance_probs = net.get_afterstate(latent, action)
    print(f"\nAfterstate inference:")
    print(f"  Afterstate shape: {afterstate.shape}")
    print(f"  Chance probs shape: {chance_probs.shape}")
    print(f"  Chance probs sum: {chance_probs[0].sum().item():.4f}")
    
    print("\n✓ All tests passed!")
