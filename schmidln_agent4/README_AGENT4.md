# Agent 4: Stochastic MuZero for Backgammon

## Overview

This agent implements **Stochastic MuZero**, a model-based reinforcement learning algorithm that combines Monte Carlo Tree Search (MCTS) with a learned world model to handle the stochastic nature of backgammon (dice rolls).

Based on: *Antonoglou, Schrittwieser, Ozair, Hubert, Silver (2022) - "Planning in Stochastic Environments with a Learned Model"*

## Author

**Lucas Schmidt** (schmidln)  
Boston College Student (2026)  
GitHub: [schmidln](https://github.com/schmidln)  
LinkedIn: [Lucas Schmidt](https://www.linkedin.com/in/lucasschmidt33/)  

CS Course: Reinforcement Learning  
December 2025

Developed as part of the final project for Professor McTague's Reinforcement Learning course.

## Architecture

### Neural Network Components

The network consists of three main components:

1. **Representation Network** (`h`): Encodes the raw board state into a latent representation
   - Input: Board features (4 channels × 28 positions) + auxiliary features (4)
   - Output: 256-dimensional latent state
   - Architecture: 3-layer ResNet with batch normalization

2. **Dynamics Network** (`g`): Predicts the next latent state given an action
   - Splits into afterstate transition (deterministic) and chance transition (stochastic)
   - Handles dice roll outcomes as chance events
   - Architecture: 2-layer MLP with residual connection

3. **Prediction Network** (`f`): Outputs policy and value from latent state
   - Policy head: Distribution over possible moves (625 actions)
   - Value head: Expected game outcome [-1, 1]
   - Architecture: Separate MLPs for policy and value

### MCTS Implementation

The MCTS search handles stochasticity through:

- **UCB Selection**: Balances exploration vs exploitation using Upper Confidence Bounds
- **Visit Counts**: Tracks simulation statistics for each move
- **Batch Evaluation**: Efficiently evaluates all legal moves in a single GPU call
- **Exploration Noise**: Adds controlled randomness for diverse training data

```
For each move decision:
1. Get all legal moves and resulting afterstates
2. Batch evaluate afterstates using the neural network
3. Run MCTS simulations with UCB selection
4. Select move based on visit counts
```

## Files

| File | Description |
|------|-------------|
| `agent4_network.py` | Neural network architecture (representation, dynamics, prediction) |
| `agent4_mcts.py` | Monte Carlo Tree Search with chance nodes |
| `agent4_train.py` | Self-play training loop with replay buffer |
| `agent4_evaluate.py` | Evaluation against random opponent |
| `README_AGENT4.md` | This file |

## Requirements

```
python >= 3.10
pytorch >= 2.0
numpy >= 1.23
numba >= 0.57
```

**Note**: Requires `numba==0.57.1` and `numpy==1.23.5` for compatibility with the backgammon engine.

## Usage

### Training

```bash
# Basic training (quick test)
PYTHONPATH=.. python agent4_train.py --iterations 50 --games-per-iter 10

# Full training (recommended for strong agent)
PYTHONPATH=.. python agent4_train.py --iterations 500 --games-per-iter 50 --mcts-sims 15

# With checkpointing
PYTHONPATH=.. python agent4_train.py --iterations 200 --games-per-iter 20 \
    --save agent4_trained.pt --checkpoint-every 50
```

#### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--iterations` | 100 | Number of training iterations |
| `--games-per-iter` | 10 | Self-play games per iteration |
| `--mcts-sims` | 10 | MCTS simulations per move |
| `--batch-size` | 128 | Training batch size |
| `--lr` | 0.001 | Learning rate |
| `--save` | `agent4_trained.pt` | Output model path |
| `--checkpoint-every` | 0 | Save checkpoint every N iterations |

### Evaluation

```bash
# Evaluate against random opponent
PYTHONPATH=.. python agent4_evaluate.py --model agent4_trained.pt --games 100

# With more MCTS simulations (stronger but slower)
PYTHONPATH=.. python agent4_evaluate.py --model agent4_trained.pt --games 100 --mcts-sims 20
```

## Training Results

### Current Training Run (500 games, ~55 minutes)

```
Iteration  1: Loss 6.38 (V: 3.70, P: 2.68)
Iteration 10: Loss 3.09 (V: 2.15, P: 0.94)
Iteration 25: Loss 2.82 (V: 1.99, P: 0.83)
Iteration 50: Loss 2.75 (V: 2.00, P: 0.75)
```

### Evaluation Results (vs Random)

```
Games played: 100
Win rate: 50%
Average reward: 0.22
```

## Current Limitations & Future Work

### Training Time Constraints

**Important**: Due to time constraints, the current model was only trained for **500 self-play games** (~55 minutes). This is significantly less than what Stochastic MuZero typically requires for strong performance.

For reference:

- Current training: 500 games
- Ideal minimum: 5,000-10,000 games
- AlphaZero/MuZero papers: 100,000+ games

The 50% win rate against random reflects this limited training. With more training time, the agent should:

- Achieve 70-80%+ win rate against random
- Learn basic backgammon strategy (building points, hitting blots)
- Develop endgame bearing-off efficiency

### Ideal Future Training for Strong Agent

```bash
# Overnight trainings (+10 Hours)
PYTHONPATH=.. python agent4_train.py \
    --iterations 1000 \
    --games-per-iter 50 \
    --mcts-sims 15 \
    --save agent4_strong.pt \
    --checkpoint-every 100
```

### Potential Improvements

1. **Longer Training**: 10,000+ self-play games
2. **Curriculum Learning**: Start against weaker opponents
3. **Network Size**: Larger hidden dimensions and more residual blocks
4. **MCTS Depth**: More simulations (25-50) for better move selection
5. **Opponent Modeling**: Include opponent response in MCTS rollouts

## Algorithm Details

### Self-Play Training Loop

```
for iteration in range(num_iterations):
    # 1. Self-play phase
    for game in range(games_per_iteration):
        trajectory = play_game_with_mcts(network)
        replay_buffer.add(trajectory)
    
    # 2. Training phase
    batch = replay_buffer.sample()
    loss = compute_loss(network, batch)
    optimizer.step()
    
    # 3. Temperature annealing
    temperature *= 0.995
```

### Loss Function

The network is trained with a combined loss:

```
L = L_value + L_policy

L_value = MSE(predicted_value, actual_game_outcome)
L_policy = CrossEntropy(predicted_policy, MCTS_visit_distribution)
```

### Handling Stochasticity

Backgammon's dice rolls create a stochastic environment. The network handles this by:

1. **Afterstate Representation**: The dynamics network first computes the afterstate (board after our move, before dice roll)
2. **Chance Transitions**: A separate network component models the distribution over possible dice outcomes
3. **Expected Values**: MCTS averages over sampled dice outcomes when evaluating moves

## Performance Notes

- **Training Speed**: ~10 games/minute on NVIDIA A100 GPU
- **Evaluation Speed**: ~0.2 games/second (limited by MCTS)
- **Memory Usage**: ~500MB GPU memory during training
- **Model Size**: ~2.4M parameters

## References

1. Antonoglou, I., Schrittwieser, J., Ozair, S., Hubert, T., & Silver, D. (2022). *Planning in Stochastic Environments with a Learned Model*. ICLR 2022.

2. Schrittwieser, J., et al. (2020). *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model*. Nature.

3. Silver, D., et al. (2018). *A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go Through Self-Play*. Science.
