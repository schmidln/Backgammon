"""
Tournament Evaluation for Agent 2

This script compares different checkpoints of Agent 2 by having them
play against each other, or evaluates them against a random player.

Usage:
    python3 tournament.py --mode tournament    # Head-to-head between checkpoints
    python3 tournament.py --mode vs-random     # All checkpoints vs random player
    python3 tournament.py --mode both          # Run both evaluations

The tournament mode runs head-to-head matches between:
- Original trained weights (9000 games) vs Checkpoint 250
- Checkpoint 250 vs Checkpoint 500
- Checkpoint 500 vs Checkpoint 1000

This helps visualize the agent's learning progression - later checkpoints
should generally beat earlier ones as the agent learns better strategies.
"""

import sys
sys.path.append("..")
import argparse

from agent2_eval_2ply import OptimizedNeuralTDAgent
from agent2_evaluate import play_evaluation_game, evaluate_agent


def tournament(model1_path, model2_path, num_games=10):
    """Run head-to-head matches between two models."""
    agent1 = OptimizedNeuralTDAgent()
    agent1.load(model1_path)
    agent1.epsilon = 0.0
    
    agent2 = OptimizedNeuralTDAgent()
    agent2.load(model2_path)
    agent2.epsilon = 0.0
    
    wins1, wins2 = 0, 0
    for i in range(num_games):
        if i % 2 == 0:
            result, _ = play_evaluation_game(agent1, opponent_type='self', agent_plays_white=True, use_2ply=False)
        else:
            result, _ = play_evaluation_game(agent1, opponent_type='self', agent_plays_white=False, use_2ply=False)
        if result > 0:
            wins1 += 1
        else:
            wins2 += 1
        print(f"Game {i+1}: {'Model1' if result > 0 else 'Model2'} wins")
    print(f'\n{model1_path}: {wins1} wins')
    print(f'{model2_path}: {wins2} wins')


def run_tournament_mode():
    """Run head-to-head tournaments between checkpoints."""
    print('=== Original (9000 games) vs Checkpoint 250 ===')
    tournament('agent2_trained_weights.pt', 'agent2_opt_checkpoint_250.pt', 10)
    print()
    print('=== Checkpoint 250 vs Checkpoint 500 ===')
    tournament('agent2_opt_checkpoint_250.pt', 'agent2_opt_checkpoint_500.pt', 10)
    print()
    print('=== Checkpoint 500 vs Checkpoint 1000 ===')
    tournament('agent2_opt_checkpoint_500.pt', 'agent2_opt_checkpoint_1000.pt', 10)


def run_vs_random_mode():
    """Evaluate all checkpoints against random player."""
    models = [
        ('Original (9000 games)', 'agent2_trained_weights.pt'),
        ('Checkpoint 250', 'agent2_opt_checkpoint_250.pt'),
        ('Checkpoint 500', 'agent2_opt_checkpoint_500.pt'),
        ('Checkpoint 1000', 'agent2_opt_checkpoint_1000.pt'),
    ]
    
    print('=== All Checkpoints vs Random Player (50 games each) ===\n')
    for name, path in models:
        agent = OptimizedNeuralTDAgent()
        agent.load(path)
        agent.epsilon = 0.0
        stats = evaluate_agent(agent, num_games=50, use_2ply=False, verbose=False)
        print(f'{name}: {stats["win_rate"]:.0f}% win rate, {stats["points_per_game"]:+.2f} PPG')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tournament evaluation for Agent 2')
    parser.add_argument('--mode', type=str, default='both', 
                        choices=['tournament', 'vs-random', 'both'],
                        help='Evaluation mode: tournament, vs-random, or both')
    args = parser.parse_args()
    
    if args.mode == 'tournament':
        run_tournament_mode()
    elif args.mode == 'vs-random':
        run_vs_random_mode()
    else:  # both
        run_tournament_mode()
        print('\n' + '='*60 + '\n')
        run_vs_random_mode()