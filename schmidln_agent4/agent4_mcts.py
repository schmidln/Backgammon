"""
Agent 4: Monte Carlo Tree Search (MCTS) for Stochastic MuZero

This implements MCTS with chance nodes for stochastic games like backgammon.
The tree has three types of nodes:
1. Decision nodes (player chooses action)
2. Afterstate nodes (action applied, waiting for chance)
3. Chance nodes (dice roll determines next state)

Reference: Antonoglou-Schrittwieser-Ozair-Hubert-Silver 2022
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import math
import sys
sys.path.append("..")

from backgammon_engine import _actions, _apply_move, _to_canonical, _new_game, _roll_dice
from agent4_network import (
    StochasticMuZeroNetwork, encode_board_features, encode_board_features_batch,
    DICE_OUTCOMES, DICE_PROBS_NP, NUM_DICE_OUTCOMES,
    action_to_index, index_to_action, dice_roll_to_index
)


# =============================================================================
# MCTS HYPERPARAMETERS
# =============================================================================

@dataclass
class MCTSConfig:
    """Configuration for MCTS."""
    num_simulations: int = 50          # Number of MCTS simulations per move
    c_puct: float = 1.5                # Exploration constant
    c_init: float = 1.25               # Initial exploration bonus
    c_base: float = 19652              # Base for exploration scaling
    dirichlet_alpha: float = 0.3       # Dirichlet noise parameter
    root_dirichlet_fraction: float = 0.25  # Fraction of noise at root
    discount: float = 1.0              # Discount factor (γ)
    temperature: float = 1.0           # Temperature for action selection
    pb_c_init: float = 1.25            # UCB exploration coefficient
    pb_c_base: float = 19652           # UCB exploration base


# =============================================================================
# MCTS NODE CLASSES
# =============================================================================

@dataclass
class MCTSNode:
    """Base class for MCTS nodes."""
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    
    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass
class DecisionNode(MCTSNode):
    """
    Decision node where player chooses an action.
    Contains the latent state and children (afterstate nodes).
    """
    latent_state: Optional[torch.Tensor] = None
    policy_logits: Optional[torch.Tensor] = None
    children: Dict[int, 'AfterstateNode'] = field(default_factory=dict)
    legal_actions: Optional[List[int]] = None  # Legal action indices
    is_terminal: bool = False
    terminal_value: float = 0.0
    
    def expanded(self) -> bool:
        return len(self.children) > 0 or self.is_terminal


@dataclass
class AfterstateNode(MCTSNode):
    """
    Afterstate node after action is taken, before chance outcome.
    Represents the intermediate state waiting for dice roll.
    """
    afterstate: Optional[torch.Tensor] = None
    reward: float = 0.0
    children: Dict[int, 'DecisionNode'] = field(default_factory=dict)  # Keyed by chance outcome
    chance_probs: Optional[np.ndarray] = None  # Probability distribution over dice rolls
    
    def expanded(self) -> bool:
        return len(self.children) > 0


# =============================================================================
# MCTS IMPLEMENTATION
# =============================================================================

class MCTS:
    """
    Monte Carlo Tree Search for Stochastic MuZero.
    
    The tree alternates between:
    1. Decision nodes (select action using UCB)
    2. Afterstate nodes (sample or enumerate chance outcomes)
    3. Back to decision nodes
    """
    
    def __init__(self, network: StochasticMuZeroNetwork, config: MCTSConfig, device: torch.device):
        self.network = network
        self.config = config
        self.device = device
    
    def search(self, root_state: np.ndarray, player: int, 
               dice_roll: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Run MCTS from the given state and return action probabilities.
        
        Args:
            root_state: Current game state (28-element array)
            player: Current player (+1 white, -1 black)
            dice_roll: Current dice roll
            
        Returns:
            action_probs: Probability distribution over legal moves
            selected_action: Index of selected action in the legal moves list
        """
        # Get legal moves for current state
        legal_moves, _ = _actions(root_state, player, dice_roll)
        if len(legal_moves) == 0:
            return np.array([1.0]), 0  # No legal moves, pass
        
        # Convert state to canonical form (always from white's perspective)
        canonical_state = _to_canonical(root_state, player)
        
        # Initialize root node
        root = self._create_root_node(canonical_state, legal_moves)
        
        # Add exploration noise at root
        self._add_dirichlet_noise(root)
        
        # Run simulations
        for _ in range(self.config.num_simulations):
            self._simulate(root, legal_moves, player)
        
        # Get action probabilities from visit counts
        action_probs = self._get_action_probs(root, legal_moves)
        
        # Select action
        if self.config.temperature == 0:
            selected_action = np.argmax(action_probs)
        else:
            selected_action = np.random.choice(len(action_probs), p=action_probs)
        
        return action_probs, selected_action
    
    def _create_root_node(self, state: np.ndarray, legal_moves: List) -> DecisionNode:
        """Create and expand the root decision node."""
        # Encode state
        board_features, aux_features = encode_board_features(state)
        board_tensor = torch.tensor(board_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        aux_tensor = torch.tensor(aux_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get initial inference
        with torch.no_grad():
            latent, policy_logits, value = self.network.initial_inference(board_tensor, aux_tensor)
        
        # Create root node
        root = DecisionNode(
            latent_state=latent,
            policy_logits=policy_logits.squeeze(0),
            legal_actions=list(range(len(legal_moves))),
            visit_count=1,
            value_sum=value.item()
        )
        
        # Expand root
        self._expand_decision_node(root, legal_moves)
        
        return root
    
    def _expand_decision_node(self, node: DecisionNode, legal_moves: List) -> None:
        """Expand a decision node by creating children for each legal action."""
        if node.policy_logits is None:
            return
        
        # Compute action probabilities from policy logits
        # For backgammon moves, we need to map moves to our policy representation
        policy_probs = self._compute_move_probs(node.policy_logits, legal_moves)
        
        # Create afterstate children
        for i, move in enumerate(legal_moves):
            child = AfterstateNode(
                prior=policy_probs[i],
                chance_probs=DICE_PROBS_NP.copy()
            )
            node.children[i] = child
    
    def _compute_move_probs(self, policy_logits: torch.Tensor, legal_moves: List) -> np.ndarray:
        """
        Compute probabilities for legal moves from policy logits.
        
        The policy outputs a 25x25 grid for submove (source, dest) pairs.
        We need to aggregate these for complete moves.
        """
        # Simple approach: use first submove of each move to index into policy
        logits = policy_logits.cpu().numpy()
        logits_2d = logits.reshape(25, 25)
        
        move_logits = []
        for move in legal_moves:
            if len(move) == 0:
                # Pass move
                move_logits.append(0.0)
            else:
                # Use first submove
                first_submove = move[0]
                src = first_submove[0]  # Source point
                dst = first_submove[1]  # Destination point
                # Clamp to valid range
                src = min(max(src, 0), 24)
                dst = min(max(dst, 0), 24)
                move_logits.append(logits_2d[src, dst])
        
        move_logits = np.array(move_logits, dtype=np.float32)
        
        # Softmax
        max_logit = np.max(move_logits)
        exp_logits = np.exp(move_logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)
        
        return probs
    
    def _add_dirichlet_noise(self, node: DecisionNode) -> None:
        """Add Dirichlet noise to prior probabilities at root."""
        if len(node.children) == 0:
            return
        
        num_actions = len(node.children)
        noise = np.random.dirichlet([self.config.dirichlet_alpha] * num_actions)
        
        for i, (action, child) in enumerate(node.children.items()):
            child.prior = (1 - self.config.root_dirichlet_fraction) * child.prior + \
                          self.config.root_dirichlet_fraction * noise[i]
    
    def _simulate(self, root: DecisionNode, legal_moves: List, player: int) -> None:
        """Run one simulation from root to leaf."""
        node = root
        search_path = [node]
        actions_taken = []
        
        # Selection phase: traverse tree until we reach an unexpanded node
        while node.expanded() and not node.is_terminal:
            action, child = self._select_action(node)
            actions_taken.append((action, legal_moves[action] if action < len(legal_moves) else None))
            search_path.append(child)
            
            # From afterstate, sample or select chance outcome
            if isinstance(child, AfterstateNode):
                chance_idx = self._select_chance_outcome(child)
                
                if chance_idx in child.children:
                    # Already expanded this chance outcome
                    node = child.children[chance_idx]
                    search_path.append(node)
                else:
                    # Need to expand this chance outcome
                    break
            else:
                node = child
        
        # Expansion and evaluation
        value = self._expand_and_evaluate(search_path, actions_taken, legal_moves, player)
        
        # Backup
        self._backup(search_path, value)
    
    def _select_action(self, node: DecisionNode) -> Tuple[int, AfterstateNode]:
        """Select action using PUCT formula."""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        sqrt_total = math.sqrt(node.visit_count)
        
        for action, child in node.children.items():
            # PUCT formula
            pb_c = math.log((node.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base) + self.config.pb_c_init
            prior_score = pb_c * child.prior * sqrt_total / (1 + child.visit_count)
            
            # Value from child's perspective
            value_score = -child.value if child.visit_count > 0 else 0
            
            score = value_score + prior_score
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def _select_chance_outcome(self, node: AfterstateNode) -> int:
        """Select chance outcome (dice roll) based on probabilities."""
        # Use visit counts to guide selection, but weighted by true probabilities
        if len(node.children) == 0:
            # First time: sample from true distribution
            return np.random.choice(NUM_DICE_OUTCOMES, p=node.chance_probs)
        
        # UCB-style selection weighted by true probabilities
        best_score = -float('inf')
        best_idx = 0
        
        for idx in range(NUM_DICE_OUTCOMES):
            if idx in node.children:
                child = node.children[idx]
                visits = child.visit_count
            else:
                visits = 0
            
            # Weighted exploration bonus
            prob = node.chance_probs[idx]
            exploration = prob * math.sqrt(node.visit_count + 1) / (1 + visits)
            
            score = exploration
            if score > best_score:
                best_score = score
                best_idx = idx
        
        return best_idx
    
    def _expand_and_evaluate(self, search_path: List, actions_taken: List, 
                             legal_moves: List, player: int) -> float:
        """Expand leaf node and return value estimate."""
        leaf = search_path[-1]
        
        if isinstance(leaf, DecisionNode):
            if leaf.is_terminal:
                return leaf.terminal_value
            
            # Evaluate with network
            if leaf.latent_state is not None:
                with torch.no_grad():
                    _, value = self.network.prediction(leaf.latent_state)
                return value.item()
            return 0.0
        
        elif isinstance(leaf, AfterstateNode):
            # Need to expand through chance node
            parent_decision = search_path[-2] if len(search_path) >= 2 else None
            
            if parent_decision is not None and isinstance(parent_decision, DecisionNode):
                # Get afterstate from dynamics network
                with torch.no_grad():
                    action_tensor = torch.zeros(1, dtype=torch.long, device=self.device)
                    afterstate, _ = self.network.get_afterstate(
                        parent_decision.latent_state, action_tensor
                    )
                leaf.afterstate = afterstate
            
            # Sample a chance outcome and create child decision node
            chance_idx = np.random.choice(NUM_DICE_OUTCOMES, p=leaf.chance_probs)
            
            # Create new decision node for this chance outcome
            if leaf.afterstate is not None:
                with torch.no_grad():
                    chance_tensor = torch.tensor([chance_idx], device=self.device)
                    next_state, policy, value = self.network.apply_chance(
                        leaf.afterstate, chance_tensor
                    )
                
                child = DecisionNode(
                    latent_state=next_state,
                    policy_logits=policy.squeeze(0),
                    visit_count=0,
                    value_sum=0.0
                )
                leaf.children[chance_idx] = child
                
                return value.item()
            
            return 0.0
        
        return 0.0
    
    def _backup(self, search_path: List, value: float) -> None:
        """Backup value through the search path."""
        # Alternate sign as we go up (opponent's perspective)
        current_value = value
        
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += current_value
            
            # Flip value for opponent at decision nodes
            if isinstance(node, DecisionNode):
                current_value = -current_value
    
    def _get_action_probs(self, root: DecisionNode, legal_moves: List) -> np.ndarray:
        """Get action probabilities from visit counts."""
        visit_counts = np.zeros(len(legal_moves), dtype=np.float32)
        
        for action, child in root.children.items():
            if action < len(legal_moves):
                visit_counts[action] = child.visit_count
        
        if self.config.temperature == 0:
            # Greedy
            probs = np.zeros_like(visit_counts)
            probs[np.argmax(visit_counts)] = 1.0
        else:
            # Apply temperature
            visit_counts = visit_counts ** (1.0 / self.config.temperature)
            total = np.sum(visit_counts)
            if total > 0:
                probs = visit_counts / total
            else:
                probs = np.ones_like(visit_counts) / len(visit_counts)
        
        return probs


# =============================================================================
# SIMPLIFIED MCTS FOR INITIAL TRAINING
# =============================================================================

class SimpleMCTS:
    """
    Simplified MCTS for Stochastic MuZero.
    
    Uses the network to evaluate positions and builds a shallow search tree.
    Handles stochasticity by sampling dice outcomes.
    """
    
    def __init__(self, network: StochasticMuZeroNetwork, device: torch.device, 
                 num_simulations: int = 10):
        self.network = network
        self.device = device
        self.num_simulations = num_simulations
    
    def search(self, state: np.ndarray, player: int, 
               dice_roll: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        MCTS search with chance nodes for dice.
        
        For efficiency, we:
        1. Batch evaluate all our possible afterstates
        2. For top moves, sample a few opponent dice rolls
        3. Use visit counts to select final move
        """
        legal_moves, afterstates = _actions(state, player, dice_roll)
        
        if len(legal_moves) == 0:
            return np.array([1.0]), 0
        
        if len(legal_moves) == 1:
            return np.array([1.0]), 0
        
        n_moves = len(legal_moves)
        
        # Step 1: Batch evaluate all afterstates to get initial values
        afterstate_values = self._batch_evaluate_afterstates(afterstates, player)
        
        # Initialize visit counts and value sums
        visit_counts = np.ones(n_moves, dtype=np.float32)  # Start with 1 visit
        value_sums = afterstate_values.copy()
        
        # Step 2: Run MCTS simulations
        for sim in range(self.num_simulations):
            # UCB selection
            ucb_scores = value_sums / visit_counts + 1.5 * np.sqrt(np.log(sim + 2) / visit_counts)
            selected_idx = np.argmax(ucb_scores)
            
            # Get afterstate for selected move
            afterstate = np.array(afterstates[selected_idx], dtype=np.int8)
            
            # Sample opponent dice roll
            dice_idx = np.random.choice(NUM_DICE_OUTCOMES, p=DICE_PROBS_NP)
            opp_dice = np.array(DICE_OUTCOMES[dice_idx], dtype=np.int8)
            
            # Get opponent moves and evaluate
            opp_moves, opp_afterstates = _actions(afterstate, -player, opp_dice)
            
            if len(opp_moves) == 0:
                # Opponent can't move, use current afterstate value
                value = afterstate_values[selected_idx]
            elif len(opp_moves) == 1:
                # Only one opponent move
                opp_result = np.array(opp_afterstates[0], dtype=np.int8)
                value = -self._evaluate_single(opp_result, -player)
            else:
                # Evaluate opponent's best response (batch)
                opp_values = self._batch_evaluate_afterstates(opp_afterstates, -player)
                best_opp_value = np.max(opp_values)
                value = -best_opp_value  # Opponent minimizes our value
            
            # Update statistics
            visit_counts[selected_idx] += 1
            value_sums[selected_idx] += value
        
        # Select move based on visit counts
        total_visits = np.sum(visit_counts)
        probs = visit_counts / total_visits
        
        # Select most visited
        best_action = np.argmax(visit_counts)
        
        return probs, best_action
    
    def _batch_evaluate_afterstates(self, afterstates, player: int) -> np.ndarray:
        """Batch evaluate afterstates using the network."""
        n = len(afterstates)
        
        # Convert to list of numpy arrays in canonical form
        states_list = []
        for i in range(n):
            afterstate = np.array(afterstates[i], dtype=np.int8)
            canonical = _to_canonical(afterstate, player)
            states_list.append(canonical)
        
        # Batch encode
        board_features, aux_features = encode_board_features_batch(states_list)
        
        # To tensors
        board_tensor = torch.tensor(board_features, dtype=torch.float32, device=self.device)
        aux_tensor = torch.tensor(aux_features, dtype=torch.float32, device=self.device)
        
        # Get values
        with torch.no_grad():
            _, _, values = self.network.initial_inference(board_tensor, aux_tensor)
        
        return values.cpu().numpy().flatten()
    
    def _evaluate_single(self, state: np.ndarray, player: int) -> float:
        """Evaluate a single state."""
        canonical = _to_canonical(state, player)
        board_features, aux_features = encode_board_features(canonical)
        
        board_tensor = torch.tensor(board_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        aux_tensor = torch.tensor(aux_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, _, value = self.network.initial_inference(board_tensor, aux_tensor)
        
        return value.item()


if __name__ == "__main__":
    print("Testing MCTS implementation...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create network and MCTS
    network = StochasticMuZeroNetwork().to(device)
    mcts = SimpleMCTS(network, device, num_simulations=5)
    
    # Create a test state - _new_game returns (player, dice, state)
    player, dice, state = _new_game()
    
    print(f"\nTest state created")
    print(f"Player: {player}, Dice roll: {dice}")
    
    # Time the search
    import time
    start = time.time()
    probs, action = mcts.search(state, player, dice)
    elapsed = time.time() - start
    
    print(f"\nSearch results:")
    print(f"  Action probabilities shape: {probs.shape}")
    print(f"  Selected action: {action}")
    print(f"  Max probability: {np.max(probs):.3f}")
    print(f"  Time: {elapsed:.3f}s")
    
    print("\n✓ MCTS test passed!")