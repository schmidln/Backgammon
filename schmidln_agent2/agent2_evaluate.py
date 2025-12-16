"""
Evaluation script for Backgammon Agent 2.

Tests agent strength by:
1. Playing against a random player
2. Playing against itself (should be ~50/50)
3. Measuring win rates, backgammon rates, average game length
"""

import numpy as np
import torch
import time
import argparse
from typing import Dict, Tuple

from backgammon_engine import (
    _new_game, _roll_dice, _actions, _apply_move, _reward,
    _to_canonical
)
from agent2_eval_2ply import OptimizedNeuralTDAgent


def random_player_move(state, player, dice):
    """Random player - selects a random legal move."""
    moves, _ = _actions(state, player, dice)
    return moves[np.random.randint(len(moves))]


def play_evaluation_game(agent, opponent_type='random', agent_plays_white=True, use_2ply=True):
    """
    Play a single evaluation game (no learning).
    
    Args:
        agent: The trained agent
        opponent_type: 'random' or 'self'
        agent_plays_white: If True, agent plays as white (+1)
        use_2ply: Use 2-ply search (required by spec for evaluation)
    
    Returns:
        result: 1 if agent wins, -1 if agent loses, with magnitude for gammon/backgammon
        num_moves: Total moves in game
    """
    player, dice, state = _new_game()
    num_moves = 0
    
    agent_color = 1 if agent_plays_white else -1
    
    while True:
        # Determine who moves
        if player == agent_color:
            # Agent's turn - use 2-ply if specified
            if use_2ply:
                move = agent.select_move_2ply_efficient(state, player, dice)
            else:
                move = agent.select_move_1ply(state, player, dice)
        else:
            # Opponent's turn
            if opponent_type == 'random':
                move = random_player_move(state, player, dice)
            else:  # self-play
                if use_2ply:
                    move = agent.select_move_2ply_efficient(state, player, dice)
                else:
                    move = agent.select_move_1ply(state, player, dice)
        
        # Apply move
        new_state = _apply_move(state, player, move)
        reward = _reward(new_state, player)
        num_moves += 1
        
        if reward != 0:
            # Game over
            # reward is from perspective of player who just moved
            if player == agent_color:
                agent_reward = reward
            else:
                agent_reward = -reward
            return agent_reward, num_moves
        
        state = new_state
        player = -player
        dice = _roll_dice()


def evaluate_agent(agent, num_games=100, opponent_type='random', use_2ply=True, verbose=True):
    """
    Evaluate agent over multiple games.
    
    Returns:
        stats: Dictionary with win rate, backgammon rate, etc.
    """
    results = {-3: 0, -2: 0, -1: 0, 1: 0, 2: 0, 3: 0}
    total_moves = 0
    
    start_time = time.time()
    
    # Print every game for 2-ply (slow), every 10 for 1-ply (fast)
    print_every = 1 if use_2ply else 10
    
    for game_num in range(num_games):
        # Alternate colors for fairness
        agent_plays_white = (game_num % 2 == 0)
        
        result, moves = play_evaluation_game(
            agent, 
            opponent_type=opponent_type,
            agent_plays_white=agent_plays_white,
            use_2ply=use_2ply
        )
        
        results[result] = results.get(result, 0) + 1
        total_moves += moves
        
        if verbose and (game_num + 1) % print_every == 0:
            elapsed = time.time() - start_time
            wins = results[1] + results[2] + results[3]
            losses = results[-1] + results[-2] + results[-3]
            print(f"Game {game_num + 1}/{num_games} - W:{wins} L:{losses} ({(game_num+1)/elapsed:.2f} games/sec)")
    
    # Calculate statistics
    wins = results[1] + results[2] + results[3]
    losses = results[-1] + results[-2] + results[-3]
    
    agent_backgammons = results[3]
    agent_gammons = results[2]
    opp_backgammons = results[-3]
    opp_gammons = results[-2]
    
    win_rate = wins / num_games * 100
    avg_moves = total_moves / num_games
    
    # Points per game (backgammon=3, gammon=2, normal=1)
    agent_points = results[1] + 2*results[2] + 3*results[3]
    opp_points = results[-1] + 2*results[-2] + 3*results[-3]
    ppg = (agent_points - opp_points) / num_games
    
    stats = {
        'win_rate': win_rate,
        'wins': wins,
        'losses': losses,
        'results': results,
        'avg_moves': avg_moves,
        'agent_backgammons': agent_backgammons,
        'agent_gammons': agent_gammons,
        'opp_backgammons': opp_backgammons,
        'opp_gammons': opp_gammons,
        'points_per_game': ppg,
        'games_per_sec': num_games / (time.time() - start_time)
    }
    
    return stats


def print_stats(stats, title="Evaluation Results"):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Backgammon Agent 2')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--games', type=int, default=100, help='Number of games to play')
    parser.add_argument('--opponent', type=str, default='random', choices=['random', 'self'],
                        help='Opponent type')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--use-1ply', action='store_true', help='Use 1-ply search (faster but weaker)')
    
    args = parser.parse_args()
    
    use_2ply = not args.use_1ply
    
    print("=" * 60)
    print("Backgammon Agent 2 Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Games: {args.games}")
    print(f"Opponent: {args.opponent}")
    print(f"Search: {'2-ply (spec)' if use_2ply else '1-ply (fast)'}")
    print("=" * 60)
    
    # Load agent
    device = 'cpu' if args.cpu else None
    agent = OptimizedNeuralTDAgent(device=device)
    agent.load(args.model)
    agent.epsilon = 0.0  # No exploration during evaluation
    print(f"Loaded model from {args.model}")
    
    # Warmup
    print("\nWarming up...")
    play_evaluation_game(agent, opponent_type='random', use_2ply=use_2ply)
    
    # Evaluate vs random
    print(f"\nEvaluating against {args.opponent} player...")
    stats = evaluate_agent(agent, num_games=args.games, opponent_type=args.opponent, use_2ply=use_2ply)
    print_stats(stats, f"Agent vs {args.opponent.capitalize()}")
    
    # If vs random, also show expected values
    if args.opponent == 'random':
        print("\nInterpretation:")
        if stats['win_rate'] > 90:
            print("  Excellent! Agent is much stronger than random.")
        elif stats['win_rate'] > 75:
            print("  Good. Agent has learned meaningful strategy.")
        elif stats['win_rate'] > 60:
            print("  Moderate. Agent is somewhat better than random.")
        else:
            print("  Weak. Agent needs more training or has issues.")