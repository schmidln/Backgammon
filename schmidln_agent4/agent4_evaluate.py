"""
Agent 4: Stochastic MuZero Evaluation

Evaluate trained agent against random opponent or other agents.
"""

import numpy as np
import torch
import argparse
import time
import sys
from typing import Tuple, Optional
sys.path.append("..")

from backgammon_engine import (
    _new_game, _roll_dice, _actions, _apply_move, _reward,
    _to_canonical
)
from agent4_network import StochasticMuZeroNetwork, encode_board_features
from agent4_mcts import SimpleMCTS, MCTSConfig


# =============================================================================
# AGENT CLASS
# =============================================================================

class MuZeroAgent:
    """Agent 4: Stochastic MuZero agent for game play."""
    
    def __init__(self, model_path: str, device: torch.device, 
                 mcts_simulations: int = 20):
        self.device = device
        self.mcts_simulations = mcts_simulations
        
        # Load network
        self.network = StochasticMuZeroNetwork().to(device)
        self._load_model(model_path)
        self.network.eval()
        
        # Create MCTS
        self.mcts = SimpleMCTS(self.network, device, mcts_simulations)
    
    def _load_model(self, path: str) -> None:
        """Load model weights."""
        import sys
        # Add a dummy TrainingConfig to handle pickle loading
        class TrainingConfig:
            pass
        # Temporarily add to module namespace for unpickling
        current_module = sys.modules[__name__]
        if not hasattr(current_module, 'TrainingConfig'):
            current_module.TrainingConfig = TrainingConfig
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if 'network_state_dict' in checkpoint:
            self.network.load_state_dict(checkpoint['network_state_dict'])
        else:
            self.network.load_state_dict(checkpoint)
        print(f"Loaded model from {path}")
    
    def select_move(self, state: np.ndarray, dice: np.ndarray, 
                    player: int) -> Tuple[int, np.ndarray]:
        """
        Select a move using MCTS.
        
        Args:
            state: Current game state
            dice: Current dice roll
            player: Current player (+1 or -1)
            
        Returns:
            move_idx: Index of selected move
            move: The actual move
        """
        legal_moves, _ = _actions(state, player, dice)
        
        if len(legal_moves) == 0:
            return -1, None
        
        if len(legal_moves) == 1:
            return 0, legal_moves[0]
        
        with torch.no_grad():
            probs, action_idx = self.mcts.search(state, player, dice)
        
        return action_idx, legal_moves[action_idx]


def random_agent_move(state: np.ndarray, dice: np.ndarray, player: int):
    """Random agent: select uniformly from legal moves."""
    legal_moves, _ = _actions(state, player, dice)
    if len(legal_moves) == 0:
        return -1, None
    idx = np.random.randint(len(legal_moves))
    return idx, legal_moves[idx]


# =============================================================================
# EVALUATION
# =============================================================================

def play_evaluation_game(agent: MuZeroAgent, agent_is_white: bool = True, 
                         verbose: bool = False) -> dict:
    """
    Play one evaluation game: agent vs random.
    
    Returns dict with game statistics.
    """
    # _new_game returns (player, dice, state)
    player, dice, state = _new_game()
    agent_player = 1 if agent_is_white else -1
    
    move_count = 0
    max_moves = 500
    
    while move_count < max_moves:
        # Check terminal
        reward = _reward(state, player)
        if reward != 0:
            # Game over
            agent_reward = reward * agent_player
            return {
                'winner': 'agent' if agent_reward > 0 else 'random',
                'reward': agent_reward,
                'moves': move_count,
                'agent_color': 'white' if agent_is_white else 'black'
            }
        
        # Roll dice (except first move uses dice from _new_game)
        if move_count > 0:
            dice = _roll_dice()
        
        # Get legal moves
        legal_moves, _ = _actions(state, player, dice)
        
        if len(legal_moves) == 0:
            player = -player
            continue
        
        # Select move
        if player == agent_player:
            idx, move = agent.select_move(state, dice, player)
        else:
            idx, move = random_agent_move(state, dice, player)
        
        if move is None:
            player = -player
            continue
        
        # Apply move
        state = _apply_move(state, player, move)
        
        if verbose and player == agent_player:
            print(f"Move {move_count}: Agent plays {move}, dice={dice}")
        
        player = -player
        move_count += 1
    
    # Max moves reached - draw
    return {
        'winner': 'draw',
        'reward': 0,
        'moves': move_count,
        'agent_color': 'white' if agent_is_white else 'black'
    }


def evaluate_agent(agent: MuZeroAgent, num_games: int = 100, 
                   verbose: bool = False) -> dict:
    """
    Evaluate agent over multiple games.
    
    Args:
        agent: The MuZero agent to evaluate
        num_games: Number of games to play
        verbose: Print individual game results
        
    Returns:
        Statistics dictionary
    """
    wins = 0
    losses = 0
    draws = 0
    total_reward = 0
    total_moves = 0
    
    results = {-3: 0, -2: 0, -1: 0, 0: 0, 1: 0, 2: 0, 3: 0}
    
    start_time = time.time()
    
    for i in range(num_games):
        # Alternate colors
        agent_is_white = (i % 2 == 0)
        
        result = play_evaluation_game(agent, agent_is_white, verbose)
        
        reward = result['reward']
        total_reward += reward
        total_moves += result['moves']
        
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
        
        # Clamp to valid range
        reward_key = max(-3, min(3, int(reward)))
        results[reward_key] += 1
        
        if verbose or (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            games_per_sec = (i + 1) / elapsed
            print(f"Game {i+1}/{num_games}: {result['winner']}, "
                  f"reward={reward}, moves={result['moves']}, "
                  f"({games_per_sec:.2f} games/sec)")
    
    elapsed = time.time() - start_time
    
    stats = {
        'games': num_games,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': wins / num_games,
        'avg_reward': total_reward / num_games,
        'avg_moves': total_moves / num_games,
        'results': results,
        'time': elapsed,
        'games_per_sec': num_games / elapsed
    }
    
    return stats


def print_results(stats: dict) -> None:
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Games played: {stats['games']}")
    print(f"Wins: {stats['wins']} ({stats['win_rate']*100:.1f}%)")
    print(f"Losses: {stats['losses']} ({stats['losses']/stats['games']*100:.1f}%)")
    print(f"Draws: {stats['draws']}")
    print(f"Average reward: {stats['avg_reward']:.3f}")
    print(f"Average moves: {stats['avg_moves']:.1f}")
    print(f"Time: {stats['time']:.1f}s ({stats['games_per_sec']:.2f} games/sec)")
    print("\nResult distribution:")
    for key in sorted(stats['results'].keys()):
        count = stats['results'][key]
        pct = count / stats['games'] * 100
        label = {-3: 'Backgammon loss', -2: 'Gammon loss', -1: 'Normal loss',
                 0: 'Draw', 1: 'Normal win', 2: 'Gammon win', 3: 'Backgammon win'}
        print(f"  {label[key]:18s}: {count:4d} ({pct:5.1f}%)")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate Stochastic MuZero Agent')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--games', type=int, default=100, help='Number of games')
    parser.add_argument('--mcts-sims', type=int, default=20, help='MCTS simulations')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--cpu', action='store_true', help='Force CPU')
    args = parser.parse_args()
    
    # Setup device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("Agent 4: Stochastic MuZero Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Games: {args.games}")
    print(f"MCTS simulations: {args.mcts_sims}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Create agent
    agent = MuZeroAgent(args.model, device, args.mcts_sims)
    
    # Warmup
    print("\nWarming up...")
    warmup_stats = evaluate_agent(agent, num_games=2, verbose=False)
    print(f"Warmup complete: {warmup_stats['games_per_sec']:.2f} games/sec")
    
    # Run evaluation
    print(f"\nEvaluating over {args.games} games...")
    stats = evaluate_agent(agent, num_games=args.games, verbose=args.verbose)
    
    # Print results
    print_results(stats)
    
    # Summary
    if stats['win_rate'] > 0.9:
        print("\n✓ Agent is MUCH stronger than random!")
    elif stats['win_rate'] > 0.7:
        print("\n✓ Agent is stronger than random.")
    elif stats['win_rate'] > 0.5:
        print("\n~ Agent is slightly better than random.")
    else:
        print("\n✗ Agent needs more training.")


if __name__ == '__main__':
    main()