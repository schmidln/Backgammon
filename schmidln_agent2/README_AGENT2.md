# Agent 2: Neural Network TD(λ) with 2-Ply Search

A backgammon-playing agent using a deep residual convolutional neural network trained with TD(λ) and eligibility traces.



## Author

**Lucas Schmidt** (schmidln)  
Boston College Student (2026)  
GitHub: [schmidln](https://github.com/schmidln)  
LinkedIn: [Lucas Schmidt](https://www.linkedin.com/in/lucasschmidt33/)  

CS Course: Reinforcement Learning  
December 2025

Developed as part of the final project for Professor McTague's Reinforcement Learning course.



## Performance

- **Win Rate vs Random**: 95-96% with 2-ply search
- **Points per Game**: +1.9 to +2.6
- **Training**: ~9,000 self-play games

## Files

| File | Description |
|------|-------------|
| `agent2_train_1ply.py` | Training script using 1-ply search for speed (~1.4 games/sec) |
| `agent2_eval_2ply.py` | Agent implementation with both 1-ply and 2-ply search methods |
| `agent2_evaluate.py` | Evaluation script to test agent against random opponent |
| `tournament.py` | Tournament script to compare checkpoints head-to-head or evaluate all against random |
| `agent2_trained_weights.pt` | Pre-trained model weights (~9,000 games) |
| `README_AGENT2` | This file |

## Architecture

The neural network follows the project specification:

- **Input**: 15 feature planes × 24 board points + 6 auxiliary features
- **Initial Conv**: Conv1D (15 → 128 filters, kernel=7)
- **Residual Blocks**: 9 × ResNet v2 blocks (128 filters, kernel=3, pre-activation with LayerNorm)
- **Output Head**: Global Average Pooling → Dense(128+6 → 64) → Dense(64 → 1) → Tanh × 3
- **Output Range**: [-3, +3] (backgammon/gammon/normal win/loss)

### Feature Encoding (15 planes per point)

| Planes | Description |
|--------|-------------|
| 0 | Empty point |
| 1-2 | Blot (1 checker) - white/black |
| 3-4 | Made point (2 checkers) |
| 5-6 | Builder (3 checkers) |
| 7-8 | Basic anchor (4 checkers) |
| 9-10 | Deep anchor (5 checkers) |
| 11-12 | Permanent anchor (6 checkers) |
| 13-14 | Overflow (>6 checkers, normalized) |

### Auxiliary Features (6)

- Bar indicator (binary) × 2 players
- Bar count (normalized /15) × 2 players  
- Borne-off count (normalized /15) × 2 players

## Training Algorithm

**TD(λ) with Eligibility Traces**

- Learning rate (α): 0.0001
- Trace decay (λ): 0.8
- Discount factor (γ): 1.0
- Exploration (ε): 0.1 with decay

The agent learns by playing against itself. After each move, it updates the value function using temporal difference learning with eligibility traces for credit assignment across multiple moves.

**Training uses 1-ply search** for speed (~1.4 games/sec), while **evaluation uses 2-ply search** as required by the project specification.

## 2-Ply Search

For each candidate move, the agent:
1. Considers all 21 possible opponent dice rolls
2. For each roll, assumes opponent plays optimally (minimizes agent's value)
3. Computes expected value weighted by dice probabilities
4. Selects the move maximizing expected value

```
a* = argmax_a Σ_j P(d_j) × min_{a_opp} V(S_{a,j,a_opp})
```

## Usage

### Evaluate the Pre-trained Agent

```bash
# Quick test (2 games)
python agent2_evaluate.py --model agent2_trained_weights.pt --games 2

# Full evaluation with 2-ply (slow, ~100 sec/game)
python agent2_evaluate.py --model agent2_trained_weights.pt --games 50

# Fast evaluation with 1-ply
python agent2_evaluate.py --model agent2_trained_weights.pt --games 100 --use-1ply
```

### Train a New Agent

```bash
# Train from scratch (1000 games)
python agent2_train_1ply.py --games 1000 --save my_agent.pt

# Continue training from checkpoint
python agent2_train_1ply.py --games 5000 --load agent2_trained_weights.pt --save my_agent_continued.pt --checkpoint-every 1000
```

### Command Line Arguments

**Training (`agent2_train_1ply.py`)**:
- `--games`: Number of games to train (default: 1000)
- `--save`: Output filename for trained model
- `--load`: Load existing weights to continue training
- `--alpha`: Learning rate (default: 0.0001)
- `--checkpoint-every`: Save checkpoints every N games

**Evaluation (`agent2_evaluate.py`)**:
- `--model`: Path to model weights file (required)
- `--games`: Number of games to play (default: 100)
- `--opponent`: Opponent type - 'random' or 'self' (default: random)
- `--use-1ply`: Use faster 1-ply search instead of 2-ply

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Numba
- CUDA-capable GPU (recommended)

## Notes

- The `backgammon_engine.py` file must be in the parent directory
- 2-ply search is slow (~0.01 games/sec) but produces stronger play
- Training was performed on NVIDIA A100 GPUs via NSF ACCESS Delta

