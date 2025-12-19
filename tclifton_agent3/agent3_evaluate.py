"""
Agent 3: Evaluation Script

Evaluates the trained PPO agent against:
1. Random player
2. Itself (should be ~50/50)

Uses policy-pruned 2-ply search for evaluation (as per spec).
"""

import numpy as np
import torch
import time
import argparse
from typing import Dict, Tuple
import sys
sys.path.append("..")

from backgammon_engine import (
    _new_game, _roll_dice, _actions, _apply_move, _reward,
    _to_canonical
)
from agent3_network import PPOAgent


def random_player_move(state: np.ndarray, player: int, dice: np.ndarray) -> int:
    """Random player - selects a random legal move."""
    moves, _ = _actions(state, player, dice)
    if len(moves) == 0:
        return 0
    return np.random.randint(len(moves))


def play_evaluation_game(agent: PPOAgent, 
                         opponent_type: str = 'random',
                         agent_plays_white: bool = True,
                         use_policy_pruned_2ply: bool = True,
                         top_k: int = 5) -> Tuple[int, int]:
    """
    Play a single evaluation game (no learning).
    
    Args:
        agent: The trained agent
        opponent_type: 'random' or 'self'
        agent_plays_white: If True, agent plays as white (+1)
        use_policy_pruned_2ply: Use policy-pruned 2-ply search (spec requirement)
        top_k: Number of moves to consider in pruned search
    
    Returns:
        result: Reward from agent's perspective
        num_moves: Total moves in game
    """
    player, dice, state = _new_game()
    num_moves = 0
    
    agent_color = 1 if agent_plays_white else -1
    
    while True:
        moves, _ = _actions(state, player, dice)
        
        if len(moves) == 0:
            # No legal moves, switch player
            player = -player
            dice = _roll_dice()
            continue
        
        # Determine who moves
        if player == agent_color:
            # Agent's turn
            if use_policy_pruned_2ply:
                move_idx = agent.select_move_policy_pruned_2ply(
                    state, player, dice, top_k=top_k
                )
            else:
                move_idx = agent.select_move_1ply(state, player, dice)
        else:
            # Opponent's turn
            if opponent_type == 'random':
                move_idx = random_player_move(state, player, dice)
            else:  # self-play
                if use_policy_pruned_2ply:
                    move_idx = agent.select_move_policy_pruned_2ply(
                        state, player, dice, top_k=top_k
                    )
                else:
                    move_idx = agent.select_move_1ply(state, player, dice)
        
        # Apply move
        move = moves[move_idx]
        new_state = _apply_move(state, player, move)
        reward = _reward(new_state, player)
        num_moves += 1
        
        if reward != 0:
            # Game over
            if player == agent_color:
                agent_reward = reward
            else:
                agent_reward = -reward
            return agent_reward, num_moves
        
        state = new_state
        player = -player
        dice = _roll_dice()


def evaluate_agent(agent: PPOAgent,
                   num_games: int = 100,
                   opponent_type: str = 'random',
                   use_policy_pruned_2ply: bool = True,
                   top_k: int = 5,
                   verbose: bool = True) -> Dict:
    """
    Evaluate agent over multiple games.
    
    Returns:
        stats: Dictionary with win rate, backgammon rate, etc.
    """
    results = {-3: 0, -2: 0, -1: 0, 1: 0, 2: 0, 3: 0}
    total_moves = 0
    
    start_time = time.time()
    
    for game_num in range(num_games):
        # Alternate colors for fairness
        agent_plays_white = (game_num % 2 == 0)
        
        result, moves = play_evaluation_game(
            agent, 
            opponent_type=opponent_type,
            agent_plays_white=agent_plays_white,
            use_policy_pruned_2ply=use_policy_pruned_2ply,
            top_k=top_k
        )
        
        results[result] = results.get(result, 0) + 1
        total_moves += moves
        
        if verbose and (game_num + 1) % 10 == 0:
            elapsed = time.time() - start_time
            wins = results[1] + results[2] + results[3]
            losses = results[-1] + results[-2] + results[-3]
            print(f"Game {game_num + 1}/{num_games} - W:{wins} L:{losses} "
                  f"({(game_num+1)/elapsed:.2f} games/sec)")
    
    # Calculate statistics
    wins = results[1] + results[2] + results[3]
    losses = results[-1] + results[-2] + results[-3]
    
    win_rate = wins / num_games * 100
    avg_moves = total_moves / num_games
    
    # Points per game
    agent_points = results[1] + 2*results[2] + 3*results[3]
    opp_points = results[-1] + 2*results[-2] + 3*results[-3]
    ppg = (agent_points - opp_points) / num_games
    
    stats = {
        'win_rate': win_rate,
        'wins': wins,
        'losses': losses,
        'results': results,
        'avg_moves': avg_moves,
        'agent_backgammons': results[3],
        'agent_gammons': results[2],
        'opp_backgammons': results[-3],
        'opp_gammons': results[-2],
        'points_per_game': ppg,
        'games_per_sec': num_games / (time.time() - start_time)
    }
    
    return stats


def print_stats(stats: Dict, title: str = "Evaluation Results"):
    """Pretty print evaluation statistics."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"Win Rate: {stats['win_rate']:.1f}% ({stats['wins']} wins, {stats['losses']} losses)")
    print(f"Points per Game: {stats['points_per_game']:+.3f}")
    print(f"Average Game Length: {stats['avg_moves']:.1f} moves")
    print(f"Games per Second: {stats['games_per_sec']:.2f}")
    print()
    print("Result Distribution:")
    print(f"  Agent Backgammons (+3): {stats['agent_backgammons']}")
    print(f"  Agent Gammons (+2):     {stats['agent_gammons']}")
    print(f"  Agent Normal Wins (+1): {stats['results'][1]}")
    print(f"  Opponent Normal Wins:   {stats['results'][-1]}")
    print(f"  Opponent Gammons:       {stats['opp_gammons']}")
    print(f"  Opponent Backgammons:   {stats['opp_backgammons']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Agent 3 (PPO)')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--games', type=int, default=100, help='Number of games')
    parser.add_argument('--opponent', type=str, default='random', 
                        choices=['random', 'self'], help='Opponent type')
    parser.add_argument('--top-k', type=int, default=5, help='Top-K moves for pruned search')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--use-1ply', action='store_true', help='Use 1-ply instead of 2-ply')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    use_policy_pruned_2ply = not args.use_1ply
    
    print("=" * 60)
    print("Agent 3 (PPO) Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Games: {args.games}")
    print(f"Opponent: {args.opponent}")
    print(f"Search: {'Policy-pruned 2-ply (K={})'.format(args.top_k) if use_policy_pruned_2ply else '1-ply'}")
    print("=" * 60)
    
    # Setup device
    device = 'cpu' if args.cpu else None
    
    # Load agent
    agent = PPOAgent(device=device)
    agent.load(args.model)
    
    # Warmup
    print("\nWarming up...")
    play_evaluation_game(agent, opponent_type='random', use_policy_pruned_2ply=use_policy_pruned_2ply)
    
    # Evaluate
    print(f"\nEvaluating against {args.opponent} player...")
    stats = evaluate_agent(
        agent, 
        num_games=args.games, 
        opponent_type=args.opponent,
        use_policy_pruned_2ply=use_policy_pruned_2ply,
        top_k=args.top_k,
        verbose=args.verbose
    )
    
    print_stats(stats, f"Agent 3 vs {args.opponent.capitalize()}")
    
    # Interpretation
    if args.opponent == 'random':
        print("\nInterpretation:")
        if stats['win_rate'] > 90:
            print("  ? Excellent! Agent is much stronger than random.")
        elif stats['win_rate'] > 75:
            print("  ? Good. Agent has learned meaningful strategy.")
        elif stats['win_rate'] > 60:
            print("  ~ Moderate. Agent is somewhat better than random.")
        else:
            print("  ? Weak. Agent needs more training.")


if __name__ == '__main__':
    main()

