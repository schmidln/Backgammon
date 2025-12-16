"""
Agent 2 OPTIMIZED: TD(λ) with batched neural network evaluations.

Key optimizations:
1. Batch all afterstate evaluations in move selection (single GPU call)
2. Minimize CPU-GPU transfers
3. Keep tensors on GPU when possible
4. Use torch.no_grad() consistently for inference

This should be 5-10x faster than the non-batched version.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import time
import os

# Import the game engine
import sys; sys.path.append(".."); from backgammon_engine import (
    _new_game, _roll_dice, _actions, _apply_move, _reward,
    _to_canonical,
    W_BAR, B_BAR, W_OFF, B_OFF, NUM_POINTS, NUM_CHECKERS,
    int8
)

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

BOARD_LENGTH = 24
CONV_INPUT_CHANNELS = 15
AUX_INPUT_SIZE = 6
FILTERS = 128
NUM_RESIDUAL_BLOCKS = 9
MAX_SCORE = 3.0


# =============================================================================
# OPTIMIZED FEATURE ENCODING (Vectorized)
# =============================================================================

def encode_board_features_batch(states: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode multiple states at once (vectorized for speed).
    """
    batch_size = len(states)
    board_features = np.zeros((batch_size, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
    aux_features = np.zeros((batch_size, AUX_INPUT_SIZE), dtype=np.float32)
    
    for b, state in enumerate(states):
        # Board features
        for i in range(BOARD_LENGTH):
            point_idx = i + 1
            count = state[point_idx]
            
            if count == 0:
                board_features[b, i, 0] = 1.0
            elif count > 0:
                idx = min(count, 6) * 2 - 1  # 1->1, 2->3, 3->5, 4->7, 5->9, 6->11
                board_features[b, i, idx] = 1.0
                if count > 6:
                    board_features[b, i, 13] = (count - 6) / 9.0
            else:
                black_count = -count
                idx = min(black_count, 6) * 2  # 1->2, 2->4, 3->6, 4->8, 5->10, 6->12
                board_features[b, i, idx] = 1.0
                if black_count > 6:
                    board_features[b, i, 14] = (black_count - 6) / 9.0
        
        # Auxiliary features
        white_bar = state[W_BAR]
        black_bar = -state[B_BAR]
        aux_features[b, 0] = 1.0 if white_bar > 0 else 0.0
        aux_features[b, 1] = 1.0 if black_bar > 0 else 0.0
        aux_features[b, 2] = white_bar / 15.0
        aux_features[b, 3] = black_bar / 15.0
        aux_features[b, 4] = state[W_OFF] / 15.0
        aux_features[b, 5] = -state[B_OFF] / 15.0
    
    return board_features, aux_features


def encode_single_state(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Encode a single state."""
    board_features = np.zeros((BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
    aux_features = np.zeros(AUX_INPUT_SIZE, dtype=np.float32)
    
    for i in range(BOARD_LENGTH):
        point_idx = i + 1
        count = state[point_idx]
        
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
    
    white_bar = state[W_BAR]
    black_bar = -state[B_BAR]
    aux_features[0] = 1.0 if white_bar > 0 else 0.0
    aux_features[1] = 1.0 if black_bar > 0 else 0.0
    aux_features[2] = white_bar / 15.0
    aux_features[3] = black_bar / 15.0
    aux_features[4] = state[W_OFF] / 15.0
    aux_features[5] = -state[B_OFF] / 15.0
    
    return board_features, aux_features


# =============================================================================
# NEURAL NETWORK
# =============================================================================

class ResidualBlockV2(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.ln1 = nn.LayerNorm([channels, BOARD_LENGTH])
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.ln2 = nn.LayerNorm([channels, BOARD_LENGTH])
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding='same')
    
    def forward(self, x):
        residual = x
        out = F.relu(self.ln1(x))
        out = self.conv1(out)
        out = F.relu(self.ln2(out))
        out = self.conv2(out)
        return residual + out


class BackgammonValueNet(nn.Module):
    def __init__(self, num_res_blocks: int = NUM_RESIDUAL_BLOCKS, filters: int = FILTERS):
        super().__init__()
        self.initial_conv = nn.Conv1d(CONV_INPUT_CHANNELS, filters, kernel_size=7, padding='same')
        self.res_blocks = nn.ModuleList([
            ResidualBlockV2(filters, kernel_size=3) for _ in range(num_res_blocks)
        ])
        self.fc1 = nn.Linear(filters + AUX_INPUT_SIZE, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, board_features: torch.Tensor, aux_features: torch.Tensor) -> torch.Tensor:
        x = board_features.transpose(1, 2)
        x = F.relu(self.initial_conv(x))
        for block in self.res_blocks:
            x = block(x)
        x = x.mean(dim=2)
        x = torch.cat([x, aux_features], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.tanh(x) * MAX_SCORE


# =============================================================================
# OPTIMIZED TD(λ) AGENT
# =============================================================================

class OptimizedNeuralTDAgent:
    """
    Optimized TD(λ) agent with batched evaluations.
    """
    
    def __init__(self, 
                 alpha: float = 0.0001,
                 gamma: float = 1.0,
                 lambda_: float = 0.8,
                 epsilon: float = 0.1,
                 device: str = None):
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.network = BackgammonValueNet().to(self.device)
        self.network.eval()  # Default to eval mode
        
        # Pre-allocate tensors for common batch sizes to avoid repeated allocation
        self._tensor_cache = {}
        
        self.traces = {name: torch.zeros_like(param) 
                       for name, param in self.network.named_parameters()}
    
    def reset_traces(self):
        for name in self.traces:
            self.traces[name].zero_()
    
    def _get_cached_tensors(self, batch_size):
        """Get or create cached tensors for given batch size."""
        if batch_size not in self._tensor_cache:
            self._tensor_cache[batch_size] = {
                'board': torch.zeros((batch_size, BOARD_LENGTH, CONV_INPUT_CHANNELS), 
                                    dtype=torch.float32, device=self.device),
                'aux': torch.zeros((batch_size, AUX_INPUT_SIZE), 
                                  dtype=torch.float32, device=self.device)
            }
        return self._tensor_cache[batch_size]
    
    @torch.no_grad()
    def batch_value_fast(self, canonical_states: List[np.ndarray]) -> np.ndarray:
        """
        Fast batched value computation - single GPU call for all states.
        """
        if len(canonical_states) == 0:
            return np.array([], dtype=np.float32)
        
        # Encode all states at once
        board_np, aux_np = encode_board_features_batch(canonical_states)
        
        # Transfer to GPU
        board_tensor = torch.from_numpy(board_np).to(self.device)
        aux_tensor = torch.from_numpy(aux_np).to(self.device)
        
        # Single forward pass for all states
        values = self.network(board_tensor, aux_tensor)
        
        return values.cpu().numpy().flatten()
    
    @torch.no_grad()
    def value(self, state: np.ndarray) -> float:
        """Single state value (for compatibility)."""
        board_feat, aux_feat = encode_single_state(state)
        board_tensor = torch.from_numpy(board_feat).unsqueeze(0).to(self.device)
        aux_tensor = torch.from_numpy(aux_feat).unsqueeze(0).to(self.device)
        value = self.network(board_tensor, aux_tensor)
        return float(value.item())
    
    def select_move_batched(self, state, player, dice):
        """
        Select best move using BATCHED evaluation of all afterstates.
        This is the key optimization - one GPU call instead of N calls.
        """
        moves, afterstates = _actions(state, player, dice)
        
        if len(moves) == 0:
            return moves[0]
        
        if len(moves) == 1:
            return moves[0]
        
        # Convert all afterstates to canonical form
        canonical_states = [_to_canonical(afterstates[i], player) for i in range(len(moves))]
        
        # BATCH evaluate all at once (single GPU call!)
        values = self.batch_value_fast(canonical_states)
        
        # Select best
        best_idx = np.argmax(values)
        return moves[best_idx]
    
    def select_move_epsilon_greedy(self, state, player, dice):
        """Epsilon-greedy move selection with batched evaluation."""
        if np.random.rand() < self.epsilon:
            moves, _ = _actions(state, player, dice)
            return moves[np.random.randint(len(moves))]
        return self.select_move_batched(state, player, dice)
    
    def select_move_2ply(self, state, player, dice):
        """
        2-ply search: For each of our moves, consider all 21 opponent dice rolls,
        assume opponent plays optimally (minimizes our value), and pick the move
        with best expected value.
        
        This is the method specified in the project requirements.
        """
        moves, afterstates = _actions(state, player, dice)
        
        if len(moves) == 0:
            return moves[0]
        
        if len(moves) == 1:
            return moves[0]
        
        # All 21 possible dice rolls with probabilities
        dice_rolls = []
        dice_probs = []
        for r1 in range(1, 7):
            for r2 in range(1, r1 + 1):
                dice_rolls.append(np.array([r1, r2], dtype=np.int8))
                dice_probs.append(1/36 if r1 == r2 else 2/36)
        
        move_values = np.zeros(len(moves))
        
        for m_idx in range(len(moves)):
            afterstate = afterstates[m_idx]
            expected_value = 0.0
            
            # For each possible opponent dice roll
            for d_idx, (opp_dice, prob) in enumerate(zip(dice_rolls, dice_probs)):
                # Get all opponent moves from this afterstate
                opp_moves, opp_afterstates = _actions(afterstate, -player, opp_dice)
                
                if len(opp_moves) == 0:
                    # Opponent has no moves, evaluate afterstate
                    canonical = _to_canonical(afterstate, player)
                    val = self.value(canonical)
                    expected_value += prob * val
                else:
                    # Batch evaluate all opponent afterstates
                    opp_canonicals = [_to_canonical(opp_afterstates[i], player) 
                                     for i in range(len(opp_moves))]
                    opp_values = self.batch_value_fast(opp_canonicals)
                    
                    # Opponent minimizes our value
                    min_val = np.min(opp_values)
                    expected_value += prob * min_val
            
            move_values[m_idx] = expected_value
        
        # Select move with highest expected value
        best_idx = np.argmax(move_values)
        return moves[best_idx]
    
    def select_move_2ply_epsilon_greedy(self, state, player, dice):
        """Epsilon-greedy with 2-ply search."""
        if np.random.rand() < self.epsilon:
            moves, _ = _actions(state, player, dice)
            return moves[np.random.randint(len(moves))]
        return self.select_move_2ply(state, player, dice)
    
    def update(self, state: np.ndarray, td_target: float):
        """TD(λ) update with eligibility traces."""
        self.network.train()
        
        board_feat, aux_feat = encode_single_state(state)
        board_tensor = torch.from_numpy(board_feat).unsqueeze(0).to(self.device)
        aux_tensor = torch.from_numpy(aux_feat).unsqueeze(0).to(self.device)
        
        self.network.zero_grad()
        value = self.network(board_tensor, aux_tensor)
        value.backward()
        
        td_error = td_target - value.item()
        
        with torch.no_grad():
            for name, param in self.network.named_parameters():
                if param.grad is not None:
                    self.traces[name] = self.gamma * self.lambda_ * self.traces[name] + param.grad
                    param.add_(self.alpha * td_error * self.traces[name])
        
        self.network.eval()
    
    def save(self, filepath: str):
        torch.save({
            'model_state_dict': self.network.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])


# =============================================================================
# OPTIMIZED TRAINING
# =============================================================================

def play_single_game_optimized(agent: OptimizedNeuralTDAgent, 
                                use_2ply: bool = False) -> Tuple[int, int]:
    """
    Play a single game with optimized batched move selection.
    """
    player, dice, state = _new_game()
    starting_player = player
    num_moves = 0
    
    agent.reset_traces()
    prev_canonical = {1: None, -1: None}
    
    while True:
        canonical = _to_canonical(state, player)
        current_value = agent.value(canonical)
        
        # TD update for previous state
        if prev_canonical[player] is not None:
            td_target = agent.gamma * (-current_value)
            agent.update(prev_canonical[player], td_target)
        
        # Select move (1-ply or 2-ply)
        if use_2ply:
            move = agent.select_move_2ply_epsilon_greedy(state, player, dice)
        else:
            move = agent.select_move_epsilon_greedy(state, player, dice)
        
        # Apply move
        new_state = _apply_move(state, player, move)
        reward = _reward(new_state, player)
        num_moves += 1
        
        if reward != 0:
            agent.update(canonical, float(reward))
            opponent = -player
            if prev_canonical[opponent] is not None:
                agent.update(prev_canonical[opponent], float(-reward))
            
            final_reward = reward * player * starting_player
            return final_reward, num_moves
        
        prev_canonical[player] = canonical.copy()
        state = new_state
        player = -player
        dice = _roll_dice()


def train_optimized(agent: OptimizedNeuralTDAgent,
                    num_games: int = 1000,
                    print_every: int = 100,
                    use_2ply: bool = False,
                    epsilon_decay: float = 0.9999,
                    min_epsilon: float = 0.01,
                    checkpoint_every: int = None,
                    checkpoint_path: str = "agent2_opt_checkpoint") -> Dict[int, int]:
    """
    Optimized training loop.
    """
    total_moves = 0
    results = {-3: 0, -2: 0, -1: 0, 1: 0, 2: 0, 3: 0}
    start_time = time.time()
    
    for game_num in range(num_games):
        reward, moves = play_single_game_optimized(agent, use_2ply=use_2ply)
        total_moves += moves
        results[reward] = results.get(reward, 0) + 1
        
        agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)
        
        if (game_num + 1) % print_every == 0:
            elapsed = time.time() - start_time
            avg_moves = total_moves / (game_num + 1)
            games_per_sec = (game_num + 1) / elapsed
            
            _, _, test_state = _new_game()
            test_value = agent.value(_to_canonical(test_state, 1))
            
            print(f"Game {game_num + 1}/{num_games} ({games_per_sec:.2f} games/sec)")
            print(f"  Avg moves: {avg_moves:.1f}, Epsilon: {agent.epsilon:.4f}")
            print(f"  Results: {results}")
            print(f"  Sample start value: {test_value:.4f}")
        
        if checkpoint_every and (game_num + 1) % checkpoint_every == 0:
            agent.save(f"{checkpoint_path}_{game_num + 1}.pt")
            print(f"  Checkpoint saved: {checkpoint_path}_{game_num + 1}.pt")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Optimized Backgammon Agent 2')
    parser.add_argument('--games', type=int, default=1000, help='Number of games')
    parser.add_argument('--alpha', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--lambda_', type=float, default=0.8, help='Eligibility trace decay')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Initial exploration rate')
    parser.add_argument('--checkpoint-every', type=int, default=None, help='Save every N games')
    parser.add_argument('--load', type=str, default=None, help='Load model from file')
    parser.add_argument('--save', type=str, default='agent2_optimized.pt', help='Save model to file')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--use-2ply', action='store_true', help='Use 2-ply search (slower but stronger)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Agent 2 OPTIMIZED: Batched Neural Network TD(λ)")
    print("=" * 60)
    print(f"Games: {args.games}")
    print(f"Alpha: {args.alpha}, Lambda: {args.lambda_}")
    print(f"Epsilon: {args.epsilon}")
    print(f"2-ply search: {args.use_2ply}")
    print("=" * 60)
    
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    device = 'cpu' if args.cpu else None
    agent = OptimizedNeuralTDAgent(
        alpha=args.alpha,
        gamma=1.0,
        lambda_=args.lambda_,
        epsilon=args.epsilon,
        device=device
    )
    
    if args.load and os.path.exists(args.load):
        agent.load(args.load)
        print(f"Loaded model from {args.load}")
    
    # Warmup
    print("\nWarming up...")
    t0 = time.time()
    reward, moves = play_single_game_optimized(agent)
    print(f"Warmup: {moves} moves in {time.time()-t0:.2f}s")
    
    # Benchmark
    print("\nBenchmarking (5 games)...")
    t0 = time.time()
    for _ in range(5):
        play_single_game_optimized(agent)
    elapsed = time.time() - t0
    print(f"Benchmark: {5/elapsed:.2f} games/sec ({elapsed/5:.2f}s per game)")
    
    # Train
    print(f"\nStarting training...")
    print("-" * 60)
    
    results = train_optimized(
        agent,
        num_games=args.games,
        print_every=max(1, args.games // 10),
        use_2ply=args.use_2ply,
        checkpoint_every=args.checkpoint_every
    )
    
    print("-" * 60)
    print("Training complete!")
    print(f"Final results: {results}")
    
    agent.save(args.save)
    print(f"\nModel saved to {args.save}")