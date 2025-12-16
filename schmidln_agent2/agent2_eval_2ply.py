"""
Agent 2 FULLY OPTIMIZED: TD(λ) with FULLY batched 2-ply search.

Key optimization from spec:
"build up an array of all states appearing in the RHS of the formula, 
transfer this array to the GPU all at once, compute all their values in parallel"

This batches ALL opponent afterstates across ALL moves into a SINGLE GPU call.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import time
import os
import gc

from backgammon_engine import (
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

# Pre-compute all 21 dice rolls and probabilities
DICE_ROLLS = []
DICE_PROBS = []
for r1 in range(1, 7):
    for r2 in range(1, r1 + 1):
        DICE_ROLLS.append(np.array([r1, r2], dtype=np.int8))
        DICE_PROBS.append(1/36 if r1 == r2 else 2/36)
DICE_PROBS = np.array(DICE_PROBS, dtype=np.float32)


# =============================================================================
# FEATURE ENCODING
# =============================================================================

def encode_board_features_batch(states: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    batch_size = len(states)
    board_features = np.zeros((batch_size, BOARD_LENGTH, CONV_INPUT_CHANNELS), dtype=np.float32)
    aux_features = np.zeros((batch_size, AUX_INPUT_SIZE), dtype=np.float32)

    for b, state in enumerate(states):
        # board planes
        for i in range(BOARD_LENGTH):
            point_idx = i + 1
            count = int(state[point_idx])

            if count == 0:
                board_features[b, i, 0] = 1.0
            elif count > 0:
                idx = min(count, 6) * 2 - 1
                board_features[b, i, idx] = 1.0
                if count > 6:
                    board_features[b, i, 13] = (count - 6) / 9.0
            else:
                black_count = -count
                idx = min(black_count, 6) * 2
                board_features[b, i, idx] = 1.0
                if black_count > 6:
                    board_features[b, i, 14] = (black_count - 6) / 9.0

        # aux
        white_bar = int(state[W_BAR])
        black_bar = -int(state[B_BAR])   # B_BAR stored negative in this engine
        aux_features[b, 0] = 1.0 if white_bar > 0 else 0.0
        aux_features[b, 1] = 1.0 if black_bar > 0 else 0.0
        aux_features[b, 2] = white_bar / 15.0
        aux_features[b, 3] = black_bar / 15.0
        aux_features[b, 4] = int(state[W_OFF]) / 15.0
        aux_features[b, 5] = int(state[B_OFF]) / 15.0  # B_OFF stored positive in this engine

    return board_features, aux_features


def encode_single_state(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    aux_features[5] = int(state[B_OFF]) / 15.0  # <-- fixed

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
# OPTIMIZED TD(λ) AGENT WITH FULLY BATCHED 2-PLY
# =============================================================================

class OptimizedNeuralTDAgent:
    
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
        self.network.eval()
        
        self.traces = {name: torch.zeros_like(param) 
                       for name, param in self.network.named_parameters()}
    
    def reset_traces(self):
        for name in self.traces:
            self.traces[name].zero_()
    
    @torch.no_grad()
    def batch_value_fast(self, canonical_states: List[np.ndarray], max_batch_size: int = 512) -> np.ndarray:
        """
        Batched value computation with chunking for memory efficiency.
        """
        if len(canonical_states) == 0:
            return np.array([], dtype=np.float32)
        
        # If small enough, do in one batch
        if len(canonical_states) <= max_batch_size:
            board_np, aux_np = encode_board_features_batch(canonical_states)
            board_tensor = torch.from_numpy(board_np).to(self.device)
            aux_tensor = torch.from_numpy(aux_np).to(self.device)
            values = self.network(board_tensor, aux_tensor)
            return values.cpu().numpy().flatten()
        
        # Otherwise, chunk into smaller batches
        all_values = []
        for i in range(0, len(canonical_states), max_batch_size):
            chunk = canonical_states[i:i + max_batch_size]
            board_np, aux_np = encode_board_features_batch(chunk)
            board_tensor = torch.from_numpy(board_np).to(self.device)
            aux_tensor = torch.from_numpy(aux_np).to(self.device)
            values = self.network(board_tensor, aux_tensor)
            all_values.append(values.cpu().numpy().flatten())
        
        return np.concatenate(all_values)
    
    @torch.no_grad()
    def value(self, state: np.ndarray) -> float:
        board_feat, aux_feat = encode_single_state(state)
        board_tensor = torch.from_numpy(board_feat).unsqueeze(0).to(self.device)
        aux_tensor = torch.from_numpy(aux_feat).unsqueeze(0).to(self.device)
        value = self.network(board_tensor, aux_tensor)
        return float(value.item())
    
    def select_move_1ply(self, state, player, dice):
        """1-ply greedy search (fast, for training)."""
        moves, afterstates = _actions(state, player, dice)
        
        if len(moves) <= 1:
            return moves[0]
        
        canonical_states = [_to_canonical(afterstates[i], player) for i in range(len(moves))]
        values = self.batch_value_fast(canonical_states)
        best_idx = np.argmax(values)
        return moves[best_idx]
    
    def select_move_2ply_efficient(self, state, player, dice):
        """
        Memory-efficient 2-ply search.
        Processes one move at a time to avoid OOM.
        Still batches opponent responses per move.
        """
        moves, afterstates = _actions(state, player, dice)
        
        if len(moves) == 0:
            return moves[0]
        if len(moves) == 1:
            return moves[0]
        
        move_values = np.zeros(len(moves), dtype=np.float32)
        
        for m_idx in range(len(moves)):
            afterstate = afterstates[m_idx]
            expected_value = 0.0
            
            # Process each dice roll
            for d_idx, opp_dice in enumerate(DICE_ROLLS):
                opp_moves, opp_afterstates = _actions(afterstate, -player, opp_dice)
                
                if len(opp_moves) == 0:
                    canonical = _to_canonical(afterstate, player)
                    val = self.value(canonical)
                else:
                    # Batch evaluate opponent moves for this dice roll
                    opp_canonicals = [_to_canonical(opp_afterstates[i], player) 
                                     for i in range(len(opp_moves))]
                    opp_values = self.batch_value_fast(opp_canonicals)
                    val = np.min(opp_values)
                
                expected_value += DICE_PROBS[d_idx] * val
            
            move_values[m_idx] = expected_value
        
        best_idx = np.argmax(move_values)
        return moves[best_idx]
    
    def select_move_2ply_fully_batched(self, state, player, dice):
        """
        FULLY BATCHED 2-ply search
        
        Collects ALL opponent afterstates across ALL moves into a single batch,
        evaluates them in ONE GPU call, then computes expected values.
        """
        moves, afterstates = _actions(state, player, dice)
        
        if len(moves) == 0:
            return moves[0]
        if len(moves) == 1:
            return moves[0]
        
        # Collect ALL states to evaluate in a single list
        all_states_to_eval = []
        # Track structure: (move_idx, dice_idx, num_opp_moves, is_no_move_case)
        eval_structure = []
        
        for m_idx in range(len(moves)):
            afterstate = afterstates[m_idx]
            
            for d_idx, opp_dice in enumerate(DICE_ROLLS):
                opp_moves, opp_afterstates = _actions(afterstate, -player, opp_dice)
                
                if len(opp_moves) == 0:
                    # No opponent moves - evaluate the afterstate itself
                    canonical = _to_canonical(afterstate, player)
                    all_states_to_eval.append(canonical)
                    eval_structure.append((m_idx, d_idx, 1, True))
                else:
                    # Add all opponent afterstates
                    start_idx = len(all_states_to_eval)
                    for i in range(len(opp_moves)):
                        canonical = _to_canonical(opp_afterstates[i], player)
                        all_states_to_eval.append(canonical)
                    eval_structure.append((m_idx, d_idx, len(opp_moves), False))
        
        # SINGLE GPU CALL for all states
        all_values = self.batch_value_fast(all_states_to_eval)
        
        # Now compute expected values per move
        move_values = np.zeros(len(moves), dtype=np.float32)
        value_idx = 0
        
        for (m_idx, d_idx, num_states, is_no_move) in eval_structure:
            if is_no_move:
                # Just the afterstate value
                val = all_values[value_idx]
                value_idx += 1
            else:
                # Opponent picks min value
                opp_values = all_values[value_idx:value_idx + num_states]
                val = np.min(opp_values)
                value_idx += num_states
            
            # Weight by dice probability
            move_values[m_idx] += DICE_PROBS[d_idx] * val
        
        best_idx = np.argmax(move_values)
        return moves[best_idx]
    
    def select_move_epsilon_greedy(self, state, player, dice, use_2ply=False):
        """Epsilon-greedy move selection."""
        if np.random.rand() < self.epsilon:
            moves, _ = _actions(state, player, dice)
            return moves[np.random.randint(len(moves))]
        
        if use_2ply:
            return self.select_move_2ply_efficient(state, player, dice)
        else:
            return self.select_move_1ply(state, player, dice)
    
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
                    # In-place operations to avoid memory allocation
                    self.traces[name].mul_(self.gamma * self.lambda_).add_(param.grad)
                    param.add_(self.traces[name], alpha=self.alpha * td_error)
        
        # Clear gradients and free memory
        self.network.zero_grad(set_to_none=True)
        self.network.eval()
        del board_tensor, aux_tensor, value
    
    def save(self, filepath: str):
        torch.save({
            'model_state_dict': self.network.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint['model_state_dict'])


# =============================================================================
# TRAINING
# =============================================================================

def play_single_game(agent: OptimizedNeuralTDAgent, 
                     use_2ply: bool = False) -> Tuple[int, int]:
    player, dice, state = _new_game()
    starting_player = player
    num_moves = 0
    
    agent.reset_traces()
    prev_canonical = {1: None, -1: None}
    
    while True:
        canonical = _to_canonical(state, player)
        current_value = agent.value(canonical)
        
        if prev_canonical[player] is not None:
            td_target = agent.gamma * (-current_value)
            agent.update(prev_canonical[player], td_target)
        
        move = agent.select_move_epsilon_greedy(state, player, dice, use_2ply=use_2ply)
        
        new_state = _apply_move(state, player, move)
        reward = _reward(new_state, player)
        num_moves += 1
        
        # Memory cleanup every 10 moves to prevent OOM
        if num_moves % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        if reward != 0:
            agent.update(canonical, float(reward))
            opponent = -player
            if prev_canonical[opponent] is not None:
                agent.update(prev_canonical[opponent], float(-reward))
            
            final_reward = reward * player * starting_player
            torch.cuda.empty_cache()
            gc.collect()
            return final_reward, num_moves
        
        prev_canonical[player] = canonical.copy()
        state = new_state
        player = -player
        dice = _roll_dice()


def train(agent: OptimizedNeuralTDAgent,
          num_games: int = 1000,
          print_every: int = 100,
          use_2ply: bool = False,
          epsilon_decay: float = 0.9999,
          min_epsilon: float = 0.01,
          checkpoint_every: int = None,
          checkpoint_path: str = "agent2_checkpoint") -> Dict[int, int]:
    
    total_moves = 0
    results = {-3: 0, -2: 0, -1: 0, 1: 0, 2: 0, 3: 0}
    start_time = time.time()
    
    for game_num in range(num_games):
        reward, moves = play_single_game(agent, use_2ply=use_2ply)
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
    
    parser = argparse.ArgumentParser(description='Train Backgammon Agent 2')
    parser.add_argument('--games', type=int, default=1000, help='Number of games')
    parser.add_argument('--alpha', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--lambda_', type=float, default=0.8, help='Eligibility trace decay')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Initial exploration rate')
    parser.add_argument('--checkpoint-every', type=int, default=None, help='Save every N games')
    parser.add_argument('--load', type=str, default=None, help='Load model from file')
    parser.add_argument('--save', type=str, default='agent2_model.pt', help='Save model to file')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--use-2ply', action='store_true', help='Use 2-ply search (required by spec, but slower)')
    parser.add_argument('--use-1ply', action='store_true', help='Use 1-ply search (faster for initial training)')
    
    args = parser.parse_args()
    
    # Default to 2-ply as per spec, unless --use-1ply is specified
    use_2ply = not args.use_1ply
    
    print("=" * 60)
    print("Agent 2: Neural Network TD(λ) with 2-ply Search")
    print("=" * 60)
    print(f"Games: {args.games}")
    print(f"Alpha: {args.alpha}, Lambda: {args.lambda_}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Search: {'2-ply (spec)' if use_2ply else '1-ply (fast)'}")
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
    
    # Warmup and benchmark
    print("\nWarming up...")
    t0 = time.time()
    reward, moves = play_single_game(agent, use_2ply=use_2ply)
    print(f"Warmup: {moves} moves in {time.time()-t0:.2f}s")
    
    print("\nBenchmarking (3 games)...")
    t0 = time.time()
    for _ in range(3):
        play_single_game(agent, use_2ply=use_2ply)
    elapsed = time.time() - t0
    print(f"Benchmark: {3/elapsed:.2f} games/sec ({elapsed/3:.2f}s per game)")
    
    # Train
    print(f"\nStarting training...")
    print("-" * 60)
    
    results = train(
        agent,
        num_games=args.games,
        print_every=max(1, args.games // 10),
        use_2ply=use_2ply,
        checkpoint_every=args.checkpoint_every
    )
    
    print("-" * 60)
    print("Training complete!")
    print(f"Final results: {results}")
    
    agent.save(args.save)
    print(f"\nModel saved to {args.save}")

    from backgammon_engine import _new_game, _reward, W_OFF, B_OFF
    import numpy as np

    _, _, s = _new_game()

    # White win terminal
    s1 = s.copy()
    s1[W_OFF] = 15
    s1[B_OFF] = 0
    print("White win:", _reward(s1, 1), _reward(s1, -1))

    # Black win terminal (note negative encoding for black off)
    s2 = s.copy()
    s2[W_OFF] = 0
    s2[B_OFF] = -15
    print("Black win:", _reward(s2, 1), _reward(s2, -1))
