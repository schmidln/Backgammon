"""
Agent 3: PPO Training Loop

Trains the PPO agent using self-play.

Training approach:
1. Collect trajectories through self-play
2. Compute advantages using GAE
3. Update policy and value networks using PPO objective
4. Repeat

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time
import argparse
import os
import gc
import sys

# Ensure we can import from the current directory or parent
sys.path.append("..")
sys.path.append(".")

from backgammon_engine import (
    _new_game, _roll_dice, _actions, _apply_move, _reward,
    _to_canonical, W_OFF, B_OFF
)
from agent3_network import (
    PPOAgent, PPOActorCriticNet, 
    encode_board_features_batch, encode_single_state,
    DICE_ROLLS, DICE_PROBS, MAX_SCORE
)


# =============================================================================
# TRAJECTORY STORAGE
# =============================================================================

@dataclass
class Transition:
    """Single transition in a trajectory."""
    state: np.ndarray           # Canonical state
    action_idx: int             # Index in legal moves list
    num_moves: int              # Number of legal moves at this state
    log_prob: float             # Log probability of action taken
    value: float                # Value estimate at this state
    reward: float               # Reward received
    done: bool                  # Episode terminated


class RolloutBuffer:
    """Buffer for storing trajectories collected during rollout."""
    
    def __init__(self):
        self.transitions: List[Transition] = []
    
    def add(self, state: np.ndarray, action_idx: int, num_moves: int,
            log_prob: float, value: float, reward: float, done: bool):
        self.transitions.append(Transition(
            state=state.copy(),
            action_idx=action_idx,
            num_moves=num_moves,
            log_prob=log_prob,
            value=value,
            reward=reward,
            done=done
        ))
    
    def clear(self):
        self.transitions.clear()
    
    def __len__(self):
        return len(self.transitions)


# =============================================================================
# PPO TRAINER
# =============================================================================

class PPOTrainer:
    """
    PPO Trainer for Agent 3.
    
    Implements the PPO algorithm with:
    - GAE for advantage estimation
    - Clipped surrogate objective
    - Value function clipping
    - Entropy bonus
    """
    
    def __init__(self,
                 agent: PPOAgent,
                 lr: float = 3e-4,
                 gamma: float = 1.0,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 4,
                 minibatch_size: int = 64):
        """
        Args:
            agent: PPO agent to train
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clip parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO epochs per update
            minibatch_size: Minibatch size for updates
        """
        self.agent = agent
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        
        # Update agent parameters
        self.agent.gamma = gamma
        self.agent.gae_lambda = gae_lambda
        self.agent.clip_epsilon = clip_epsilon
        self.agent.value_coef = value_coef
        self.agent.entropy_coef = entropy_coef
        
        # Statistics
        self.total_games = 0
        self.total_steps = 0
    
    def collect_rollout(self, num_games: int, use_1ply_training: bool = True) -> Tuple[RolloutBuffer, Dict]:
        """
        Collect trajectories through self-play.
        
        Args:
            num_games: Number of games to play
            use_1ply_training: Use 1-ply search during training (faster)
        
        Returns:
            buffer: RolloutBuffer containing collected transitions
            stats: Dictionary with rollout statistics
        """
        buffer = RolloutBuffer()
        total_reward = 0
        total_moves = 0
        results = {-3: 0, -2: 0, -1: 0, 1: 0, 2: 0, 3: 0}
        
        for _ in range(num_games):
            # Play one game and collect transitions
            game_reward, game_moves = self._play_training_game(buffer, use_1ply_training)
            total_reward += game_reward
            total_moves += game_moves
            results[game_reward] = results.get(game_reward, 0) + 1
            self.total_games += 1
        
        self.total_steps += len(buffer)
        
        stats = {
            'games': num_games,
            'avg_reward': total_reward / num_games,
            'avg_moves': total_moves / num_games,
            'transitions': len(buffer),
            'results': results
        }
        
        return buffer, stats
    
    def _play_training_game(self, buffer: RolloutBuffer, 
                            use_1ply: bool) -> Tuple[int, int]:
        """
        Play one training game and add transitions to buffer.
        
        Both players use the same agent (self-play).
        Transitions are stored from perspective of starting player.
        """
        player, dice, state = _new_game()
        starting_player = player
        num_moves = 0
        
        # Track transitions for each player separately
        player_transitions = {1: [], -1: []}
        
        while True:
            moves, afterstates = _actions(state, player, dice)
            
            if len(moves) == 0:
                # No legal moves, skip turn
                player = -player
                dice = _roll_dice()
                continue
            
            # Get canonical state from current player's perspective
            canonical = _to_canonical(state, player)
            
            # Select action
            if use_1ply:
                # During training, use simple policy + 1-ply
                action_idx, log_prob, value = self.agent.select_action(
                    state, player, dice, deterministic=False
                )
            else:
                # Full policy-pruned 2-ply (slower)
                action_idx = self.agent.select_move_policy_pruned_2ply(
                    state, player, dice, top_k=5
                )
                # Still need log_prob and value for PPO
                canonical = _to_canonical(state, player)
                probs = self.agent.get_action_probs(canonical, len(moves))
                log_prob = np.log(probs[action_idx] + 1e-8)
                value = self.agent.get_value(canonical)
            
            # Store transition (reward and done will be filled in later)
            player_transitions[player].append({
                'state': canonical.copy(),
                'action_idx': action_idx,
                'num_moves': len(moves),
                'log_prob': log_prob,
                'value': value,
                'player': player
            })
            
            # Apply action
            move = moves[action_idx]
            new_state = _apply_move(state, player, move)
            reward = _reward(new_state, player)
            num_moves += 1
            
            if reward != 0:
                # Game over - add all transitions with proper rewards
                # Reward is from perspective of starting player
                final_reward = reward * player * starting_player
                
                # Add transitions for winning player (positive reward at end)
                winner = player if reward > 0 else -player
                last_idx = len(player_transitions[winner]) - 1
                for i, trans in enumerate(player_transitions[winner]):
                    # Intermediate rewards are 0, final gets the terminal reward
                    is_last = (i == last_idx)
                    buffer.add(
                        state=trans['state'],
                        action_idx=trans['action_idx'],
                        num_moves=trans['num_moves'],
                        log_prob=trans['log_prob'],
                        value=trans['value'],
                        reward=float(abs(reward)) if is_last else 0.0,
                        done=is_last
                    )
                
                # Add transitions for losing player (negative reward at end)
                loser = -winner
                last_idx = len(player_transitions[loser]) - 1
                for i, trans in enumerate(player_transitions[loser]):
                    is_last = (i == last_idx)
                    buffer.add(
                        state=trans['state'],
                        action_idx=trans['action_idx'],
                        num_moves=trans['num_moves'],
                        log_prob=trans['log_prob'],
                        value=trans['value'],
                        reward=float(-abs(reward)) if is_last else 0.0,
                        done=is_last
                    )
                
                return final_reward, num_moves
            
            state = new_state
            player = -player
            dice = _roll_dice()
        
        return 0, num_moves
    
    def compute_advantages(self, buffer: RolloutBuffer) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantages using GAE.
        
        Returns:
            advantages: Normalized advantages for each transition
            returns: Target returns for value function
        """
        transitions = buffer.transitions
        n = len(transitions)
        
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)
        
        last_gae = 0.0
        
        # Process in reverse order
        for t in reversed(range(n)):
            trans = transitions[t]
            
            if trans.done:
                # Terminal state - no next value
                delta = trans.reward - trans.value
                last_gae = delta
            else:
                # Non-terminal
                next_value = transitions[t + 1].value if t + 1 < n else 0.0
                delta = trans.reward + self.gamma * next_value - trans.value
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            
            advantages[t] = last_gae
            returns[t] = advantages[t] + trans.value
        
        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std
        
        return advantages, returns
    
    def ppo_update(self, buffer: RolloutBuffer) -> Dict:
        """
        Perform PPO update on the collected buffer.
        
        Returns:
            metrics: Dictionary with training metrics
        """
        transitions = buffer.transitions
        n = len(transitions)
        
        if n == 0:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}
        
        # Compute advantages
        advantages, returns = self.compute_advantages(buffer)
        
        # Prepare data
        states = np.array([t.state for t in transitions])
        action_indices = np.array([t.action_idx for t in transitions])
        num_moves_list = np.array([t.num_moves for t in transitions])
        old_log_probs = np.array([t.log_prob for t in transitions], dtype=np.float32)
        old_values = np.array([t.value for t in transitions], dtype=np.float32)
        
        # Encode states
        board_np, aux_np = encode_board_features_batch(list(states))
        board_tensor = torch.from_numpy(board_np).to(self.agent.device)
        aux_tensor = torch.from_numpy(aux_np).to(self.agent.device)
        
        advantages_tensor = torch.from_numpy(advantages).to(self.agent.device)
        returns_tensor = torch.from_numpy(returns).to(self.agent.device)
        old_log_probs_tensor = torch.from_numpy(old_log_probs).to(self.agent.device)
        old_values_tensor = torch.from_numpy(old_values).to(self.agent.device)
        
        # PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        indices = np.arange(n)
        
        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, n, self.minibatch_size):
                end = min(start + self.minibatch_size, n)
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_board = board_tensor[batch_indices]
                batch_aux = aux_tensor[batch_indices]
                batch_actions = action_indices[batch_indices]
                batch_num_moves = num_moves_list[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_old_values = old_values_tensor[batch_indices]
                
                # Forward pass
                self.agent.network.train()
                
                # Create action masks for batch
                max_moves = int(batch_num_moves.max())
                action_masks = torch.zeros(len(batch_indices), self.agent.max_actions, 
                                          dtype=torch.bool, device=self.agent.device)
                for i, nm in enumerate(batch_num_moves):
                    action_masks[i, :nm] = True
                
                policy_logits, values = self.agent.network(batch_board, batch_aux, action_masks)
                values = values.squeeze(-1)
                
                # Compute new log probs and entropy
                new_log_probs = []
                entropies = []
                
                for i in range(len(batch_indices)):
                    nm = int(batch_num_moves[i])
                    logits = policy_logits[i, :nm]
                    probs = F.softmax(logits, dim=0)
                    
                    action_idx = batch_actions[i]
                    log_prob = torch.log(probs[action_idx] + 1e-8)
                    new_log_probs.append(log_prob)
                    
                    # Entropy
                    entropy = -(probs * torch.log(probs + 1e-8)).sum()
                    entropies.append(entropy)
                
                new_log_probs = torch.stack(new_log_probs)
                entropies = torch.stack(entropies)
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                values_clipped = batch_old_values + torch.clamp(
                    values - batch_old_values, -self.clip_epsilon, self.clip_epsilon
                )
                value_loss1 = (values - batch_returns).pow(2)
                value_loss2 = (values_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                # Entropy bonus
                entropy_loss = -entropies.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.agent.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.network.parameters(), self.max_grad_norm)
                self.agent.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropies.mean().item()
                num_updates += 1
        
        self.agent.network.eval()
        
        metrics = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'num_updates': num_updates
        }
        
        return metrics
    
    def train(self, 
              num_iterations: int,
              games_per_iteration: int = 100,
              use_1ply_training: bool = True,
              checkpoint_every: int = 20,
              save_path: str = 'agent3_trained.pt'):
        """
        Main training loop.
        
        Args:
            num_iterations: Number of training iterations
            games_per_iteration: Games to play per iteration
            use_1ply_training: Use 1-ply during training (faster)
            checkpoint_every: Save checkpoint every N iterations
            save_path: Final model save path
        """
        print("=" * 60)
        print("Agent 3: PPO Training")
        print("=" * 60)
        print(f"Device: {self.agent.device}")
        print(f"Iterations: {num_iterations}")
        print(f"Games per iteration: {games_per_iteration}")
        print(f"PPO epochs: {self.ppo_epochs}")
        print(f"Minibatch size: {self.minibatch_size}")
        print(f"Training search: {'1-ply (fast)' if use_1ply_training else '2-ply (slow)'}")
        print("=" * 60)
        
        start_time = time.time()
        
        for iteration in range(num_iterations):
            iter_start = time.time()
            
            # Collect rollout
            buffer, rollout_stats = self.collect_rollout(
                games_per_iteration, use_1ply_training
            )
            
            # PPO update
            update_metrics = self.ppo_update(buffer)
            
            # Clear buffer
            buffer.clear()
            gc.collect()
            torch.cuda.empty_cache()
            
            iter_time = time.time() - iter_start
            
            # Print progress
            if (iteration + 1) % 5 == 0 or iteration == 0:
                wins = sum(rollout_stats['results'].get(r, 0) for r in [1, 2, 3])
                print(f"\nIteration {iteration + 1}/{num_iterations} ({iter_time:.1f}s)")
                print(f"  Games: {self.total_games}, Transitions: {rollout_stats['transitions']}")
                print(f"  Avg reward: {rollout_stats['avg_reward']:.3f}, Avg moves: {rollout_stats['avg_moves']:.1f}")
                print(f"  Policy loss: {update_metrics['policy_loss']:.4f}, "
                      f"Value loss: {update_metrics['value_loss']:.4f}, "
                      f"Entropy: {update_metrics['entropy']:.4f}")
            
            # Checkpoint
            if (iteration + 1) % checkpoint_every == 0:
                self.agent.save(f'agent3_iter_{iteration + 1}.pt')
                print(f"  Checkpoint saved: agent3_iter_{iteration + 1}.pt")
        
        # Final save
        self.agent.save(save_path)
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Total games: {self.total_games}")
        print(f"Model saved: {save_path}")
        print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Agent 3 (PPO)')
    parser.add_argument('--iterations', type=int, default=100, help='Training iterations')
    parser.add_argument('--games-per-iter', type=int, default=50, help='Games per iteration')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--ppo-epochs', type=int, default=4, help='PPO epochs per update')
    parser.add_argument('--minibatch-size', type=int, default=64, help='Minibatch size')
    parser.add_argument('--checkpoint-every', type=int, default=20, help='Checkpoint frequency')
    parser.add_argument('--load', type=str, default=None, help='Load checkpoint')
    parser.add_argument('--save', type=str, default='agent3_trained.pt', help='Save path')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    parser.add_argument('--use-2ply', action='store_true', help='Use 2-ply during training (slower)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.cpu:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Create agent
    agent = PPOAgent(lr=args.lr, device=device)
    
    # Load checkpoint if specified
    if args.load and os.path.exists(args.load):
        agent.load(args.load)
    
    # Create trainer
    trainer = PPOTrainer(
        agent=agent,
        lr=args.lr,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size
    )
    
    # Warmup
    print("\nWarming up...")
    t0 = time.time()
    buffer, _ = trainer.collect_rollout(2, use_1ply_training=True)
    print(f"Warmup: {len(buffer)} transitions in {time.time()-t0:.2f}s")
    buffer.clear()
    
    # Train
    trainer.train(
        num_iterations=args.iterations,
        games_per_iteration=args.games_per_iter,
        use_1ply_training=not args.use_2ply,
        checkpoint_every=args.checkpoint_every,
        save_path=args.save
    )


if __name__ == '__main__':
    main()
