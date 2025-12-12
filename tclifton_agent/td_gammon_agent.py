import numpy as np
import numba
from numba import njit, prange
import backgammon_engine as bg
import os
import time

# --- OPTIMIZATION: BATCH SIZE ---
# Delta CPU nodes have 128 cores. We want to use them all.
BATCH_SIZE = 512  
NUM_EPISODES = 100000  # Thorough training (takes ~1-2 hours on Delta)

# [cite_start]--- Feature Extraction [cite: 69-92] ---
@njit
def get_features(state):
    """ Extracts handcrafted features from a CANONICAL state. """
    features = np.zeros(198, dtype=np.float32)
    f_idx = 0
    
    # Normalize board state
    for i in range(28):
        features[f_idx] = state[i] / 15.0
        f_idx += 1

    board = state[1:25] # P1 to P24

    # 1. Blot Counts
    my_blots = 0
    opp_blots = 0
    for i in range(24):
        if board[i] == 1: my_blots += 1
        elif board[i] == -1: opp_blots += 1
    features[f_idx] = my_blots/15.0; f_idx += 1
    features[f_idx] = opp_blots/15.0; f_idx += 1

    # 2. Primes and Blockades
    my_prime = 0
    opp_prime = 0
    cur_m = 0
    cur_o = 0
    for i in range(24):
        if board[i] >= 2: cur_m += 1
        else:
            if cur_m > my_prime: my_prime = cur_m
            cur_m = 0
        if board[i] <= -2: cur_o += 1
        else:
            if cur_o > opp_prime: opp_prime = cur_o
            cur_o = 0
    if cur_m > my_prime: my_prime = cur_m
    if cur_o > opp_prime: opp_prime = cur_o
    
    features[f_idx] = my_prime/6.0; f_idx += 1
    features[f_idx] = opp_prime/6.0; f_idx += 1

    # 3. Checkers in home (Race progress)
    my_home = 0
    for i in range(18, 24):
        if board[i] > 0: my_home += board[i]
    features[f_idx] = my_home/15.0; f_idx += 1
    
    # 4. Pip Count Diff
    my_pips = 0
    for i in range(24):
        if board[i] > 0: my_pips += board[i] * (24 - i)
    if state[0] > 0: my_pips += state[0] * 25
    
    opp_pips = 0
    for i in range(24):
        if board[i] < 0: opp_pips += abs(board[i]) * (i + 1)
    if state[25] < 0: opp_pips += abs(state[25]) * 25

    features[f_idx] = (my_pips - opp_pips) / 100.0; f_idx += 1

    return features[:f_idx]

# --- PARALLELIZED FEATURE EXTRACTION ---
@njit(parallel=True)
def _compute_batch_features_numba(flat_states_list, players, counts):
    total_states = len(flat_states_list)
    dummy = np.zeros(28, dtype=np.int8)
    d_feat = get_features(dummy)
    D = len(d_feat)
    
    feature_matrix = np.zeros((total_states, D), dtype=np.float32)
    num_games = len(players)
    
    for g in prange(num_games):  # PARALLEL LOOP
        start_idx = counts[g]
        end_idx = counts[g+1]
        p = players[g]
        if start_idx == end_idx: continue
        
        for k in range(start_idx, end_idx):
            raw_state = np.array(flat_states_list[k], dtype=np.int8)
            if p == 1:
                feat = get_features(raw_state)
            else:
                can_state = bg._to_canonical(raw_state, -1)
                feat = get_features(can_state)
            feature_matrix[k] = feat
    return feature_matrix

class TDGammonAgent:
    def __init__(self, load_path=None, alpha=0.1, lam=0.7):
        dummy = np.zeros(28, dtype=np.int8)
        self.num_features = len(get_features(dummy))
        if load_path and os.path.exists(load_path):
            self.weights = np.load(load_path)
            print(f"Loaded weights from {load_path}")
        else:
            self.weights = np.zeros(self.num_features, dtype=np.float32)
        self.alpha = alpha
        self.lam = lam
        self.gamma = 1.0

    def evaluate_batch(self, feature_matrix):
        return feature_matrix @ self.weights

    def save(self, path):
        np.save(path, self.weights)
        print(f"Saved weights to {path}")

def select_action(agent, states, players, dices):
    (state_buffer, offsets_list, afterstate_dicts, 
     counts) = bg._vectorized_collect_search_data(states, players, dices)
    feature_matrix = _compute_batch_features_numba(state_buffer, players, counts)
    values = agent.evaluate_batch(feature_matrix)
    best_moves = bg._vectorized_select_optimal_move(values, offsets_list, afterstate_dicts, counts)
    return best_moves

def train(num_episodes=NUM_EPISODES, batch_size=BATCH_SIZE):
    print(f"Starting Cluster Training: {num_episodes} updates, {batch_size} parallel games")
    start_time = time.time()
    
    agent = TDGammonAgent()
    states, players, dices = bg._vectorized_new_game(batch_size)
    z = np.zeros((batch_size, agent.num_features), dtype=np.float32)
    
    # Initial V_t
    can_states = np.zeros((batch_size, 28), dtype=np.int8)
    for i in range(batch_size):
        can_states[i] = bg._to_canonical(states[i], players[i])
    # Fast initial features (not parallel needed for 512, but fine)
    phi_t = np.array([get_features(s) for s in can_states])
    v_t = phi_t @ agent.weights
    
    for episode in range(num_episodes):
        actions = select_action(agent, states, players, dices)
        new_states = bg._vectorized_apply_move(states, players, actions)
        
        rewards = np.zeros(batch_size, dtype=np.float32)
        game_overs = np.zeros(batch_size, dtype=bool)
        for i in range(batch_size):
            r = bg._reward(new_states[i], players[i])
            if r != 0:
                rewards[i] = r
                game_overs[i] = True
        
        next_players = -players
        phi_next = np.zeros_like(phi_t)
        
        # We need next state features for TD error
        # Vectorize this loop if possible, but standard loop is okay for 512
        for i in range(batch_size):
            if not game_overs[i]:
                can_s = bg._to_canonical(new_states[i], next_players[i])
                phi_next[i] = get_features(can_s)
        
        v_next_opp = phi_next @ agent.weights
        targets = np.where(game_overs, rewards, -agent.gamma * v_next_opp)
        deltas = targets - v_t
        
        # Update Traces & Weights
        grad_accum = np.zeros_like(agent.weights)
        z = agent.gamma * agent.lam * z + phi_t
        
        # Accumulate gradients (Vectorized)
        # grad_accum = sum( delta * z )
        # Using matrix multiplication for speed: (1, B) @ (B, F) -> (1, F)
        grad_accum = deltas @ z
        
        # Reset traces for finished games
        # Using boolean indexing for speed
        if np.any(game_overs):
            z[game_overs] = 0
            
        agent.weights += agent.alpha * (grad_accum / batch_size)
        
        # Handle Game Overs (Resets)
        if np.any(game_overs):
            idxs = np.where(game_overs)[0]
            for idx in idxs:
                p, d, s = bg._new_game()
                new_states[idx] = s
                next_players[idx] = p
        
        # Handle Dice
        new_dices = bg._vectorized_roll_dice(batch_size)
        if np.any(game_overs):
             idxs = np.where(game_overs)[0]
             for idx in idxs:
                 p, d, s = bg._new_game()
                 new_states[idx] = s
                 next_players[idx] = p
                 new_dices[idx] = d

        states = new_states
        players = next_players
        dices = new_dices
        phi_t = phi_next
        
        # Re-compute features for completely new games
        if np.any(game_overs):
            idxs = np.where(game_overs)[0]
            for idx in idxs:
                can_s = bg._to_canonical(states[idx], players[idx])
                phi_t[idx] = get_features(can_s)
                
        v_t = phi_t @ agent.weights
        
        if episode % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Ep {episode}: Mean Val {np.mean(np.abs(v_t)):.3f}, Max W {np.max(np.abs(agent.weights)):.3f}, Time: {elapsed:.1f}s")
            # Autosave periodically
            agent.save("td_gammon_weights.npy")

    agent.save("td_gammon_weights.npy")
    print("Training Complete.")

if __name__ == "__main__":
    train()
