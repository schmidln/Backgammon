"""
Agent 4: Stochastic MuZero Training (PATCHED)

Training loop for Stochastic MuZero agent using:
1. Self-play to generate game trajectories
2. MCTS for move selection during self-play
3. Combined loss for value, policy, and dynamics networks

PATCH NOTES:
- Fixed compute_losses() to actually train the dynamics network
- Added reward prediction training
- dynamics_loss is now computed instead of being set to 0.0

Reference: Antonoglou-Schrittwieser-Ozair-Hubert-Silver 2022
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import time
import argparse
import os
from dataclasses import dataclass
from collections import deque
import sys
sys.path.append("..")

from backgammon_engine import (
    _new_game, _roll_dice, _actions, _apply_move, _reward,
    _to_canonical, W_OFF, B_OFF
)
from agent4_network import (
    StochasticMuZeroNetwork, encode_board_features, encode_board_features_batch,
    DICE_OUTCOMES, DICE_PROBS_NP, NUM_DICE_OUTCOMES,
    action_to_index, dice_roll_to_index, LATENT_SIZE, POLICY_SIZE
)
from agent4_mcts import SimpleMCTS, MCTSConfig


# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Learning
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs_per_iteration: int = 10
    
    # Self-play
    games_per_iteration: int = 100
    mcts_simulations: int = 10  # Reduced for speed
    
    # Replay buffer
    buffer_size: int = 100000
    min_buffer_size: int = 1000
    
    # Loss weights
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    dynamics_loss_weight: float = 0.5
    reward_loss_weight: float = 0.5  # ADDED: weight for reward prediction
    
    # Unrolling
    num_unroll_steps: int = 5  # How many steps to unroll dynamics
    td_steps: int = 10         # TD-n for value targets
    
    # Training schedule
    temperature_init: float = 1.0
    temperature_final: float = 0.1
    temperature_decay_steps: int = 50000


# =============================================================================
# REPLAY BUFFER
# =============================================================================

@dataclass
class GameTrajectory:
    """Stores a single game's trajectory data."""
    states: List[np.ndarray]           # Board states
    players: List[int]                  # Current player at each state
    actions: List[int]                  # Action indices taken
    dice_rolls: List[Tuple[int, int]]   # Dice rolls at each step
    rewards: List[float]                # Rewards received
    policies: List[np.ndarray]          # MCTS visit distributions
    values: List[float]                 # MCTS values (not used during self-play)
    
    def __len__(self):
        return len(self.states)


class ReplayBuffer:
    """Stores game trajectories for training."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)
    
    def add(self, trajectory: GameTrajectory) -> None:
        """Add a game trajectory to the buffer."""
        for i in range(len(trajectory)):
            self.buffer.append({
                'state': trajectory.states[i],
                'player': trajectory.players[i],
                'action': trajectory.actions[i] if i < len(trajectory.actions) else 0,
                'dice': trajectory.dice_rolls[i] if i < len(trajectory.dice_rolls) else (1, 1),
                'reward': trajectory.rewards[i] if i < len(trajectory.rewards) else 0.0,
                'policy': trajectory.policies[i] if i < len(trajectory.policies) else None,
                'trajectory': trajectory,
                'position': i
            })
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample a batch of positions."""
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


# =============================================================================
# SELF-PLAY
# =============================================================================

def play_game(network: StochasticMuZeroNetwork, mcts: SimpleMCTS, 
              temperature: float = 1.0, verbose: bool = False) -> GameTrajectory:
    """
    Play a single game using MCTS.
    
    Returns a trajectory with states, actions, policies, and rewards.
    """
    # _new_game returns (player, dice, state)
    player, dice, state = _new_game()
    
    trajectory = GameTrajectory(
        states=[], players=[], actions=[], dice_rolls=[],
        rewards=[], policies=[], values=[]
    )
    
    max_moves = 500  # Prevent infinite games
    move_count = 0
    
    if verbose:
        print(f"    Starting game, player={player}, dice={dice}")
    
    while move_count < max_moves:
        # Check for terminal state
        r = _reward(state, player)
        if r != 0:
            # Game over
            trajectory.states.append(state.copy())
            trajectory.players.append(player)
            trajectory.rewards.append(float(r))
            trajectory.actions.append(0)
            trajectory.dice_rolls.append((int(dice[0]), int(dice[1])))
            trajectory.policies.append(np.array([1.0]))
            if verbose:
                print(f"    Game ended after {move_count} moves, reward={r}")
            break
        
        # Get dice roll for this turn (use existing dice for first move)
        if move_count > 0:
            dice = _roll_dice()
        r1, r2 = int(dice[0]), int(dice[1])
        if r1 < r2:
            r1, r2 = r2, r1
        
        # Get legal moves
        legal_moves, _ = _actions(state, player, dice)
        
        if len(legal_moves) == 0:
            # No legal moves, switch player
            player = -player
            continue
        
        if verbose and move_count % 10 == 0:
            print(f"      Move {move_count}, {len(legal_moves)} legal moves...")
        
        # Run MCTS
        probs, action_idx = mcts.search(state, player, dice)
        
        # Store trajectory data
        trajectory.states.append(state.copy())
        trajectory.players.append(player)
        trajectory.dice_rolls.append((r1, r2))
        trajectory.policies.append(probs)
        trajectory.actions.append(action_idx)
        trajectory.rewards.append(0.0)  # Intermediate reward
        
        # Apply action with temperature-based selection
        if temperature > 0 and np.random.random() < temperature * 0.3:
            # Exploration: sample from policy
            action_idx = np.random.choice(len(probs), p=probs)
        
        move = legal_moves[action_idx]
        state = _apply_move(state, player, move)
        
        # Switch player
        player = -player
        move_count += 1
    
    # Compute value targets using TD(n) or Monte Carlo
    compute_value_targets(trajectory)
    
    return trajectory


def compute_value_targets(trajectory: GameTrajectory, 
                          discount: float = 1.0, td_steps: int = 10) -> None:
    """Compute value targets for each position in the trajectory."""
    n = len(trajectory)
    if n == 0:
        return
    
    # Get terminal reward
    terminal_reward = trajectory.rewards[-1] if trajectory.rewards else 0.0
    
    # Backward pass to compute discounted returns
    trajectory.values = [0.0] * n
    
    # Start from the end
    current_value = terminal_reward
    
    for i in range(n - 1, -1, -1):
        player = trajectory.players[i]
        
        # Value from this player's perspective
        if i == n - 1:
            trajectory.values[i] = current_value * player  # Normalize by player
        else:
            # TD(n) bootstrap
            lookahead = min(td_steps, n - 1 - i)
            
            # Sum of discounted rewards
            value = 0.0
            for j in range(lookahead):
                value += (discount ** j) * trajectory.rewards[i + j]
            
            # Bootstrap from future value
            if i + lookahead < n:
                value += (discount ** lookahead) * trajectory.values[i + lookahead]
            
            trajectory.values[i] = value


# =============================================================================
# LOSS FUNCTIONS (PATCHED - this is the key fix)
# =============================================================================

def compute_losses(network: StochasticMuZeroNetwork, batch: List[Dict],
                   config: TrainingConfig, device: torch.device) -> Tuple[torch.Tensor, Dict]:
    """
    Compute combined loss for training.
    
    Losses:
    1. Value loss: MSE between predicted and target values
    2. Policy loss: Cross-entropy between predicted and MCTS policies
    3. Dynamics loss: Prediction error for dynamics network (NOW IMPLEMENTED!)
    4. Reward loss: Prediction error for reward head (NEW!)
    """
    batch_size = len(batch)
    
    # Prepare batch data
    states = []
    next_states = []  # ADDED: for dynamics training
    target_values = []
    target_policies = []
    actions = []
    dice_indices = []
    rewards = []  # ADDED: for reward prediction
    has_next = []  # ADDED: track which samples have a next state
    
    for sample in batch:
        state = sample['state']
        player = sample['player']
        
        # Convert to canonical form
        canonical = _to_canonical(state, player)
        states.append(canonical)
        
        # Get targets from trajectory
        trajectory = sample['trajectory']
        pos = sample['position']
        
        # Value target
        if pos < len(trajectory.values):
            target_values.append(trajectory.values[pos])
        else:
            target_values.append(0.0)
        
        # Policy target
        if pos < len(trajectory.policies) and trajectory.policies[pos] is not None:
            policy = trajectory.policies[pos]
            # Create uniform policy over legal moves, padded to POLICY_SIZE
            padded = np.zeros(POLICY_SIZE, dtype=np.float32)
            # Only use as many entries as we have, up to POLICY_SIZE
            n_actions = min(len(policy), POLICY_SIZE)
            padded[:n_actions] = policy[:n_actions]
            # Renormalize if needed
            if padded.sum() > 0:
                padded = padded / padded.sum()
            else:
                padded[0] = 1.0  # Fallback
            target_policies.append(padded)
        else:
            target_policies.append(np.ones(POLICY_SIZE, dtype=np.float32) / POLICY_SIZE)
        
        # Action taken
        actions.append(sample['action'])
        
        # Dice roll index
        dice = sample['dice']
        dice_indices.append(dice_roll_to_index(dice[0], dice[1]))
        
        # Reward at this step
        rewards.append(sample['reward'])
        
        # ADDED: Get next state for dynamics training
        next_pos = pos + 1
        if next_pos < len(trajectory.states):
            next_state = trajectory.states[next_pos]
            next_player = trajectory.players[next_pos]
            next_canonical = _to_canonical(next_state, next_player)
            next_states.append(next_canonical)
            has_next.append(True)
        else:
            # No next state (terminal) - use current state as placeholder
            next_states.append(canonical)
            has_next.append(False)
    
    # Encode states
    board_features, aux_features = encode_board_features_batch(states)
    next_board_features, next_aux_features = encode_board_features_batch(next_states)
    
    # To tensors
    board_tensor = torch.tensor(board_features, dtype=torch.float32, device=device)
    aux_tensor = torch.tensor(aux_features, dtype=torch.float32, device=device)
    next_board_tensor = torch.tensor(next_board_features, dtype=torch.float32, device=device)
    next_aux_tensor = torch.tensor(next_aux_features, dtype=torch.float32, device=device)
    
    target_values_t = torch.tensor(target_values, dtype=torch.float32, device=device).unsqueeze(1)
    target_policies_t = torch.tensor(np.array(target_policies), dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    dice_tensor = torch.tensor(dice_indices, dtype=torch.long, device=device)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    has_next_tensor = torch.tensor(has_next, dtype=torch.bool, device=device)
    
    # Forward pass - initial inference
    latent, policy_logits, value = network.initial_inference(board_tensor, aux_tensor)
    
    # Value loss (MSE)
    value_loss = F.mse_loss(value, target_values_t)
    
    # Policy loss (cross-entropy / KL divergence)
    policy_probs = F.softmax(policy_logits, dim=-1)
    target_probs = target_policies_t / (target_policies_t.sum(dim=-1, keepdim=True) + 1e-8)
    policy_loss = F.kl_div(
        policy_probs.log(),
        target_probs,
        reduction='batchmean'
    )
    
    # =========================================================================
    # PATCHED: DYNAMICS LOSS (this was the missing piece!)
    # =========================================================================
    
    # Get actual next state value (ground truth) - detached so we don't backprop through it
    with torch.no_grad():
        _, _, next_value_true = network.initial_inference(next_board_tensor, next_aux_tensor)
    
    # Predict next state using dynamics network
    # Step 1: Get afterstate (after action, before dice)
    afterstate = network.dynamics.afterstate(latent, actions_tensor)
    
    # Step 2: Apply chance (dice roll) to get predicted next state
    next_latent_pred = network.dynamics.chance_transition(afterstate, dice_tensor)
    
    # Step 3: Get predicted value from predicted next state
    _, next_value_pred = network.prediction(next_latent_pred)
    
    # Dynamics loss: predicted next-state value should match actual next-state value
    # Only compute for non-terminal states
    if has_next_tensor.any():
        dynamics_loss = F.mse_loss(
            next_value_pred[has_next_tensor], 
            next_value_true[has_next_tensor]
        )
    else:
        dynamics_loss = torch.tensor(0.0, device=device)
    
    # =========================================================================
    # PATCHED: REWARD LOSS
    # =========================================================================
    
    # Predict reward from afterstate
    reward_pred = network.dynamics.predict_reward(afterstate)
    reward_loss = F.mse_loss(reward_pred, rewards_tensor)
    
    # =========================================================================
    # Combined loss
    # =========================================================================
    
    total_loss = (config.value_loss_weight * value_loss + 
                  config.policy_loss_weight * policy_loss +
                  config.dynamics_loss_weight * dynamics_loss +
                  config.reward_loss_weight * reward_loss)
    
    metrics = {
        'total_loss': total_loss.item(),
        'value_loss': value_loss.item(),
        'policy_loss': policy_loss.item(),
        'dynamics_loss': dynamics_loss.item() if isinstance(dynamics_loss, torch.Tensor) else dynamics_loss,
        'reward_loss': reward_loss.item(),
        'mean_value': value.mean().item(),
        'target_value_mean': target_values_t.mean().item()
    }
    
    return total_loss, metrics


# =============================================================================
# TRAINING LOOP
# =============================================================================

class MuZeroTrainer:
    """Trainer for Stochastic MuZero agent."""
    
    def __init__(self, config: TrainingConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Create network
        self.network = StochasticMuZeroNetwork().to(device)
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-5
        )
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Create MCTS
        self.mcts = SimpleMCTS(self.network, device, config.mcts_simulations)
        
        # Training state
        self.iteration = 0
        self.games_played = 0
        self.temperature = config.temperature_init
    
    def self_play(self, num_games: int) -> Dict:
        """Generate games through self-play."""
        self.network.eval()
        
        total_moves = 0
        total_rewards = 0
        game_lengths = []
        
        for g in range(num_games):
            print(f"  Playing game {g+1}/{num_games}...", flush=True)
            trajectory = play_game(self.network, self.mcts, self.temperature, verbose=(g==0))
            self.replay_buffer.add(trajectory)
            
            total_moves += len(trajectory)
            if trajectory.rewards:
                total_rewards += abs(trajectory.rewards[-1])
            game_lengths.append(len(trajectory))
            print(f"    Game {g+1} done: {len(trajectory)} moves", flush=True)
            
            self.games_played += 1
        
        return {
            'games': num_games,
            'avg_length': np.mean(game_lengths),
            'avg_reward': total_rewards / num_games,
            'buffer_size': len(self.replay_buffer)
        }
    
    def train_step(self) -> Dict:
        """Perform one training iteration."""
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return {'status': 'filling_buffer', 'buffer_size': len(self.replay_buffer)}
        
        self.network.train()
        
        total_metrics = {
            'total_loss': 0.0,
            'value_loss': 0.0,
            'policy_loss': 0.0,
            'dynamics_loss': 0.0,
            'reward_loss': 0.0  # ADDED
        }
        num_batches = 0
        
        for _ in range(self.config.epochs_per_iteration):
            # Sample batch
            batch = self.replay_buffer.sample(self.config.batch_size)
            
            # Compute loss
            self.optimizer.zero_grad()
            loss, metrics = compute_losses(self.network, batch, self.config, self.device)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
            
            # Update
            self.optimizer.step()
            
            # Accumulate metrics
            for key in total_metrics:
                total_metrics[key] += metrics.get(key, 0.0)
            num_batches += 1
        
        # Average metrics
        for key in total_metrics:
            total_metrics[key] /= num_batches
        
        # Update scheduler
        self.scheduler.step()
        total_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
        
        # Update temperature
        self.temperature = max(
            self.config.temperature_final,
            self.temperature * 0.995
        )
        total_metrics['temperature'] = self.temperature
        
        self.iteration += 1
        
        return total_metrics
    
    def train(self, num_iterations: int, checkpoint_every: int = 10, 
              save_path: str = 'agent4_checkpoint.pt') -> None:
        """Main training loop."""
        print("=" * 60)
        print("Agent 4: Stochastic MuZero Training (PATCHED)")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Games per iteration: {self.config.games_per_iteration}")
        print(f"MCTS simulations: {self.config.mcts_simulations}")
        print(f"Batch size: {self.config.batch_size}")
        print("=" * 60)
        
        start_time = time.time()
        
        for i in range(num_iterations):
            iter_start = time.time()
            
            print(f"\n--- Iteration {i+1}/{num_iterations} ---", flush=True)
            
            # Self-play phase
            print("Self-play phase...", flush=True)
            self_play_stats = self.self_play(self.config.games_per_iteration)
            
            # Training phase
            print("Training phase...", flush=True)
            train_stats = self.train_step()
            
            iter_time = time.time() - iter_start
            
            # Print progress (UPDATED to show dynamics and reward loss)
            if (i + 1) % 5 == 0 or i == 0:
                print(f"\nIteration {i + 1}/{num_iterations} ({iter_time:.1f}s)")
                print(f"  Games played: {self.games_played}, Buffer: {len(self.replay_buffer)}")
                print(f"  Avg game length: {self_play_stats['avg_length']:.1f}")
                if 'total_loss' in train_stats:
                    print(f"  Loss: {train_stats['total_loss']:.4f} "
                          f"(V: {train_stats['value_loss']:.4f}, "
                          f"P: {train_stats['policy_loss']:.4f}, "
                          f"D: {train_stats['dynamics_loss']:.4f}, "
                          f"R: {train_stats['reward_loss']:.4f})")
                    print(f"  Temperature: {train_stats.get('temperature', self.temperature):.3f}")
            
            # Checkpoint
            if (i + 1) % checkpoint_every == 0:
                self.save_checkpoint(f'agent4_iter_{i+1}.pt')
                print(f"  Checkpoint saved: agent4_iter_{i+1}.pt")
        
        # Final save
        self.save_checkpoint(save_path)
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Training complete!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Total games: {self.games_played}")
        print(f"Model saved: {save_path}")
        print("=" * 60)
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'iteration': self.iteration,
            'games_played': self.games_played,
            'temperature': self.temperature,
            'config': self.config
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.iteration = checkpoint['iteration']
        self.games_played = checkpoint['games_played']
        self.temperature = checkpoint['temperature']
        print(f"Loaded checkpoint from {path}")
        print(f"  Iteration: {self.iteration}, Games: {self.games_played}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Stochastic MuZero Agent')
    parser.add_argument('--iterations', type=int, default=100, help='Training iterations')
    parser.add_argument('--games-per-iter', type=int, default=50, help='Games per iteration')
    parser.add_argument('--mcts-sims', type=int, default=10, help='MCTS simulations per move')
    parser.add_argument('--batch-size', type=int, default=128, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint-every', type=int, default=20, help='Checkpoint frequency')
    parser.add_argument('--load', type=str, default=None, help='Load checkpoint')
    parser.add_argument('--save', type=str, default='agent4_trained.pt', help='Save path')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    args = parser.parse_args()
    
    # Setup device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}")
    
    # Create config
    config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        games_per_iteration=args.games_per_iter,
        mcts_simulations=args.mcts_sims
    )
    
    # Create trainer
    trainer = MuZeroTrainer(config, device)
    
    # Load checkpoint if specified
    if args.load:
        trainer.load_checkpoint(args.load)
    
    # Train
    trainer.train(
        num_iterations=args.iterations,
        checkpoint_every=args.checkpoint_every,
        save_path=args.save
    )


if __name__ == '__main__':
    main()
    
