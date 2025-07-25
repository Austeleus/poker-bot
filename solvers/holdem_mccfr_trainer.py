"""
Monte Carlo Counterfactual Regret Minimization (MCCFR) trainer for Texas Hold'em poker.

This implements external sampling MCCFR for heads-up no-limit Texas Hold'em,
following the algorithmic structure from Lanctot et al. and your comprehensive research.

External sampling samples opponent and chance actions while traversing all actions
of the current player, providing the best balance of efficiency and convergence
for poker applications.
"""

import numpy as np
import random
import time
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import IntEnum

from envs.holdem_wrapper import HoldemWrapper
from core.holdem_info_set import HoldemInfoSet, HoldemAction, Street
from core.holdem_regret_storage import HoldemStrategyProfile
from core.card_utils import calculate_preflop_equity, generate_all_preflop_hands


class NodeType(IntEnum):
    """Types of nodes in the game tree."""
    TERMINAL = 0
    CHANCE = 1
    DECISION = 2


@dataclass
class GameNode:
    """
    Represents a node in the poker game tree.
    
    This encapsulates all the state information needed for MCCFR traversal,
    including player cards, betting history, and current game state.
    """
    node_type: NodeType
    player: int  # 0 or 1 for heads-up
    hole_cards: Tuple[Tuple[int, int], Tuple[int, int]]  # Both players' cards
    community_cards: Tuple[int, ...]
    street: Street
    pot_size: int
    current_bet: int
    player_stacks: Tuple[int, int]
    betting_history: List[Tuple[int, int]]  # (player, action) pairs
    is_terminal: bool = False
    terminal_utility: Optional[float] = None
    legal_actions: Optional[List[int]] = None
    
    def get_info_set(self, player: int) -> HoldemInfoSet:
        """Create HoldemInfoSet for the specified player."""
        # Convert betting history to proper format
        from core.holdem_info_set import BettingRound
        betting_rounds = []
        if self.betting_history:
            # Group actions by street (simplified for preflop)
            current_round = BettingRound(
                street=self.street,
                actions=tuple(action for _, action in self.betting_history),
                bet_amounts=tuple(self.current_bet if action != HoldemAction.FOLD else 0 
                                for _, action in self.betting_history)
            )
            betting_rounds = [current_round]
        
        return HoldemInfoSet(
            player=player,
            hole_cards=self.hole_cards[player],
            community_cards=self.community_cards,
            street=self.street,
            betting_history=betting_rounds,
            position=0 if player == 0 else 1,  # Button vs BB
            stack_sizes=self.player_stacks,
            pot_size=self.pot_size,
            current_bet=self.current_bet,
            small_blind=1,
            big_blind=2,
            num_players=2,
            action_pointer=self.player
        )


class HoldemMCCFRTrainer:
    """
    External Sampling MCCFR trainer for heads-up Texas Hold'em.
    
    This implements the core MCCFR algorithm with external sampling, which:
    1. Samples opponent and chance actions according to their probabilities
    2. Traverses all actions for the current player being updated
    3. Updates regret values using counterfactual utilities
    4. Converges to Nash equilibrium over many iterations
    
    Key features:
    - External sampling for optimal efficiency/variance tradeoff
    - Proper alternating player updates
    - Integrated with your existing info set and regret storage systems
    - Supports both preflop-only and multi-street training
    """
    
    def __init__(self, 
                 small_blind: int = 1,
                 big_blind: int = 2,
                 initial_stack: int = 200,
                 seed: Optional[int] = None):
        """
        Initialize MCCFR trainer for full multi-street Texas Hold'em.
        
        Args:
            small_blind: Small blind amount
            big_blind: Big blind amount  
            initial_stack: Starting stack size for both players
            seed: Random seed for reproducibility
        """
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.initial_stack = initial_stack
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize strategy profile for both players
        self.strategy_profile = HoldemStrategyProfile(num_players=2)
        
        # Training statistics
        self.iterations_completed = 0
        self.total_nodes_touched = 0
        self.training_start_time = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Cache for generated preflop hands (for efficiency)
        self.preflop_hands = None
        
        # Action mapping
        self.actions = list(HoldemAction)
        
        self.logger.info(f"Initialized MCCFR trainer: "
                        f"Multi-street, "
                        f"Stack: {initial_stack}, Blinds: {small_blind}/{big_blind}")
    
    def train(self, iterations: int, save_every: int = 5000, verbose: bool = True) -> Dict:
        """
        Main training loop for MCCFR.
        
        This alternates between updating Player 0 and Player 1 regrets,
        using external sampling to traverse the game tree efficiently.
        
        Args:
            iterations: Number of MCCFR iterations to run
            save_every: Save strategy profile every N iterations
            verbose: Print progress updates
            
        Returns:
            Dictionary with training statistics
        """
        self.training_start_time = time.time()
        
        if verbose:
            self.logger.info(f"Starting MCCFR training for {iterations} iterations...")
            self.logger.info(f"External sampling, alternating player updates")
        
        for t in range(iterations):
            # Alternate between players (critical for convergence)
            traversing_player = t % 2
            
            # Sample a random deal for this iteration
            if self.preflop_hands is None:
                self.preflop_hands = generate_all_preflop_hands()
            
            # Sample hole cards for both players
            sampled_hands = random.sample(self.preflop_hands, 2)
            hole_cards = (sampled_hands[0], sampled_hands[1])
            
            # Create root node
            root_node = self._create_root_node(hole_cards)
            
            # Perform MCCFR traversal
            self._mccfr_traversal(root_node, traversing_player, 1.0, 1.0, t)
            
            self.iterations_completed += 1
            
            # Progress reporting
            if verbose and (t + 1) % 100 == 0:
                elapsed = time.time() - self.training_start_time
                rate = (t + 1) / elapsed
                self.logger.info(f"Iteration {t+1}/{iterations} "
                               f"({rate:.1f} iter/sec, {self.total_nodes_touched} nodes)")
            
            # Save intermediate results
            if save_every > 0 and (t + 1) % save_every == 0:
                self._save_checkpoint(t + 1)
        
        training_time = time.time() - self.training_start_time
        
        # Final statistics
        stats = {
            'iterations': iterations,
            'training_time': training_time,
            'iterations_per_second': iterations / training_time,
            'total_nodes_touched': self.total_nodes_touched,
            'strategy_profile_stats': self.strategy_profile.get_total_stats()
        }
        
        if verbose:
            self.logger.info(f"Training completed in {training_time:.1f}s "
                           f"({stats['iterations_per_second']:.1f} iter/sec)")
            self.logger.info(f"Total nodes: {self.total_nodes_touched}")
        
        return stats
    
    def _create_root_node(self, hole_cards: Tuple[Tuple[int, int], Tuple[int, int]]) -> GameNode:
        """Create the root node of the game tree with initial blinds posted."""
        # In heads-up: Player 0 is Button/Small Blind, Player 1 is Big Blind
        # Preflop: Button/SB acts first
        return GameNode(
            node_type=NodeType.DECISION,
            player=0,  # Button/Small Blind acts first in heads-up preflop
            hole_cards=hole_cards,
            community_cards=(),
            street=Street.PREFLOP,
            pot_size=self.small_blind + self.big_blind,
            current_bet=self.big_blind,  # Amount to call for the Button
            player_stacks=(
                self.initial_stack - self.small_blind,  # Button/SB
                self.initial_stack - self.big_blind     # BB
            ),
            betting_history=[],
            is_terminal=False
        )
    
    def _mccfr_traversal(self, 
                        node: GameNode, 
                        traversing_player: int,
                        reach_prob_0: float,
                        reach_prob_1: float,
                        iteration: int) -> float:
        """
        Core external sampling MCCFR traversal algorithm.
        
        This is the heart of the MCCFR algorithm. It recursively traverses the game tree,
        sampling opponent and chance actions while exploring all actions for the
        traversing player.
        
        Args:
            node: Current game tree node
            traversing_player: Player whose regrets we're updating (0 or 1)
            reach_prob_0: Probability Player 0 reaches this node
            reach_prob_1: Probability Player 1 reaches this node  
            iteration: Current training iteration
            
        Returns:
            Expected utility from this node for the traversing player
        """
        self.total_nodes_touched += 1
        
        # Terminal node: return utility
        if node.is_terminal:
            return self._get_terminal_utility(node, traversing_player)
        
        # Chance node: sample and recurse (for future multi-street support)
        if node.node_type == NodeType.CHANCE:
            return self._handle_chance_node(node, traversing_player, 
                                          reach_prob_0, reach_prob_1, iteration)
        
        # Decision node: MCCFR core logic
        assert node.node_type == NodeType.DECISION
        
        # Get information set and strategy
        info_set = node.get_info_set(node.player)
        strategy = self.strategy_profile.get_strategy(node.player, info_set)
        
        # If this is the traversing player's node
        if node.player == traversing_player:
            # Compute utilities for all actions
            action_utilities = {}
            expected_utility = 0.0
            
            legal_actions = self._get_legal_actions(node)
            
            # Normalize strategy over legal actions only
            legal_strategy = np.zeros(len(self.actions))
            legal_probs = strategy[legal_actions]
            legal_sum = np.sum(legal_probs)
            
            if legal_sum > 0:
                for i, action_idx in enumerate(legal_actions):
                    legal_strategy[action_idx] = legal_probs[i] / legal_sum
            else:
                # Uniform over legal actions if no positive regrets
                for action_idx in legal_actions:
                    legal_strategy[action_idx] = 1.0 / len(legal_actions)
            
            for action_idx in legal_actions:
                # Create child node
                child_node = self._apply_action(node, action_idx)
                
                # Recurse with updated reach probabilities
                if traversing_player == 0:
                    new_reach_0 = reach_prob_0 * legal_strategy[action_idx]
                    new_reach_1 = reach_prob_1
                else:
                    new_reach_0 = reach_prob_0
                    new_reach_1 = reach_prob_1 * legal_strategy[action_idx]
                
                util = self._mccfr_traversal(child_node, traversing_player,
                                           new_reach_0, new_reach_1, iteration)
                
                action_utilities[action_idx] = util
                expected_utility += legal_strategy[action_idx] * util
            
            # Update regrets (key MCCFR step)
            opponent_reach_prob = reach_prob_1 if traversing_player == 0 else reach_prob_0
            
            for action_idx in legal_actions:
                regret = action_utilities[action_idx] - expected_utility
                weighted_regret = opponent_reach_prob * regret
                
                self.strategy_profile.update_regret(node.player, info_set, 
                                                  action_idx, weighted_regret)
            
            # Update strategy sum for average policy
            own_reach_prob = reach_prob_0 if traversing_player == 0 else reach_prob_1
            self.strategy_profile.update_strategy(node.player, info_set, 
                                                legal_strategy, own_reach_prob)
            
            return expected_utility
        
        else:
            # Opponent node: sample according to their strategy
            legal_actions = self._get_legal_actions(node)
            
            # Normalize probabilities over legal actions
            legal_probs = strategy[legal_actions]
            legal_sum = np.sum(legal_probs)
            
            if legal_sum > 0:
                action_probs = legal_probs / legal_sum
            else:
                action_probs = np.ones(len(legal_actions)) / len(legal_actions)
            
            sampled_action = np.random.choice(legal_actions, p=action_probs)
            
            # Create child node and recurse
            child_node = self._apply_action(node, sampled_action)
            
            # Update reach probabilities (use normalized strategy)
            if node.player == 0:
                new_reach_0 = reach_prob_0 * action_probs[legal_actions.index(sampled_action)]
                new_reach_1 = reach_prob_1
            else:
                new_reach_0 = reach_prob_0
                new_reach_1 = reach_prob_1 * action_probs[legal_actions.index(sampled_action)]
            
            return self._mccfr_traversal(child_node, traversing_player,
                                       new_reach_0, new_reach_1, iteration)
    
    def _get_legal_actions(self, node: GameNode) -> List[int]:
        """Get legal actions for the current node."""
        actions = []
        player_stack = node.player_stacks[node.player]
        
        # Can fold if there's a bet to call (can't fold when checking is free)
        if node.current_bet > 0:
            actions.append(HoldemAction.FOLD)
        
        # Can always check/call
        actions.append(HoldemAction.CHECK_CALL)
        
        # Can raise if have enough chips for meaningful raises
        if player_stack > node.current_bet:
            call_amount = node.current_bet
            pot_after_call = node.pot_size + call_amount
            remaining_after_call = player_stack - call_amount
            
            # Quarter pot raise
            quarter_raise = pot_after_call // 4
            if quarter_raise > 0 and quarter_raise <= remaining_after_call:
                actions.append(HoldemAction.RAISE_QUARTER_POT)
            
            # Half pot raise
            half_raise = pot_after_call // 2
            if half_raise > 0 and half_raise <= remaining_after_call:
                actions.append(HoldemAction.RAISE_HALF_POT)
            
            # Full pot raise
            full_raise = pot_after_call
            if full_raise > 0 and full_raise <= remaining_after_call:
                actions.append(HoldemAction.RAISE_FULL_POT)
            
            # All-in (always available if you have chips)
            if remaining_after_call > 0:
                actions.append(HoldemAction.ALL_IN)
        
        return actions
    
    def _apply_action(self, node: GameNode, action: int) -> GameNode:
        """Apply an action to create a child node."""
        new_history = node.betting_history + [(node.player, action)]
        new_stacks = list(node.player_stacks)
        new_pot = node.pot_size
        new_current_bet = node.current_bet
        
        if action == HoldemAction.FOLD:
            # Folding ends the hand
            # Calculate actual profit/loss based on chips invested
            player_0_invested = self.initial_stack - node.player_stacks[0]
            player_1_invested = self.initial_stack - node.player_stacks[1]
            
            if node.player == 0:  # Player 0 folds, Player 1 wins
                # Player 0 loses what they invested
                utility_for_player_0 = -player_0_invested
            else:  # Player 1 folds, Player 0 wins  
                # Player 0 gains what Player 1 invested (pure profit)
                utility_for_player_0 = player_1_invested
            
            return GameNode(
                node_type=NodeType.TERMINAL,
                player=-1,  # No player to act
                hole_cards=node.hole_cards,
                community_cards=node.community_cards,
                street=node.street,
                pot_size=new_pot,
                current_bet=0,
                player_stacks=tuple(new_stacks),
                betting_history=new_history,
                is_terminal=True,
                terminal_utility=utility_for_player_0
            )
        
        elif action == HoldemAction.CHECK_CALL:
            if node.current_bet == 0:
                # Check
                next_player = 1 - node.player
                
                # In heads-up: if BB checks after SB called, betting round ends
                if (node.player == 1 and len(new_history) >= 1 and 
                    new_history[-1][1] == HoldemAction.CHECK_CALL):
                    # BB checks after SB called/limped, betting round ends
                    # Check if this is the final street
                    if self._is_final_street(node.street):
                        return self._create_showdown_node(node, new_history, new_stacks, new_pot)
                    else:
                        # Advance to next street
                        next_node = GameNode(
                            node_type=NodeType.DECISION,
                            player=next_player,
                            hole_cards=node.hole_cards,
                            community_cards=node.community_cards,
                            street=node.street,
                            pot_size=new_pot,
                            current_bet=0,
                            player_stacks=tuple(new_stacks),
                            betting_history=new_history,
                            is_terminal=False
                        )
                        return self._advance_to_next_street(next_node)
                
                # If first to act and checking (SB limping), continue to BB
                return GameNode(
                    node_type=NodeType.DECISION,
                    player=next_player,
                    hole_cards=node.hole_cards,
                    community_cards=node.community_cards,
                    street=node.street,
                    pot_size=new_pot,
                    current_bet=new_current_bet,
                    player_stacks=tuple(new_stacks),
                    betting_history=new_history,
                    is_terminal=False
                )
            else:
                # Call: need to match the current bet
                # For Button calling BB: needs to add (BB - SB) = 1 more chip
                if node.player == 0 and node.current_bet == self.big_blind and len(node.betting_history) == 0:
                    # Button calling the big blind preflop
                    call_amount = self.big_blind - self.small_blind  # Just need to complete the BB
                    new_stacks[node.player] -= call_amount
                    new_pot += call_amount
                    
                    # Special case: SB completed, now BB can raise or check to end round
                    return GameNode(
                        node_type=NodeType.DECISION,
                        player=1,  # BB gets option
                        hole_cards=node.hole_cards,
                        community_cards=node.community_cards,
                        street=node.street,
                        pot_size=new_pot,
                        current_bet=0,  # Both players have equal investment now
                        player_stacks=tuple(new_stacks),
                        betting_history=new_history,
                        is_terminal=False
                    )
                else:
                    # Regular call - match the current bet
                    call_amount = min(node.current_bet, new_stacks[node.player])
                    new_stacks[node.player] -= call_amount
                    new_pot += call_amount
                    
                    # After call, betting round typically ends in heads-up
                    # But opponent can still raise if they haven't acted yet
                    if len(new_history) == 0:  # First action
                        return GameNode(
                            node_type=NodeType.DECISION,
                            player=1 - node.player,
                            hole_cards=node.hole_cards,
                            community_cards=node.community_cards,
                            street=node.street,
                            pot_size=new_pot,
                            current_bet=0,
                            player_stacks=tuple(new_stacks),
                            betting_history=new_history,
                            is_terminal=False
                        )
                    else:
                        # Both players have acted, betting round ends
                        if self._is_final_street(node.street):
                            return self._create_showdown_node(node, new_history, new_stacks, new_pot)
                        else:
                            # Advance to next street
                            next_node = GameNode(
                                node_type=NodeType.DECISION,
                                player=0,  # First to act on new street
                                hole_cards=node.hole_cards,
                                community_cards=node.community_cards,
                                street=node.street,
                                pot_size=new_pot,
                                current_bet=0,
                                player_stacks=tuple(new_stacks),
                                betting_history=new_history,
                                is_terminal=False
                            )
                            return self._advance_to_next_street(next_node)
        
        else:
            # Raise actions
            # First, player must call the current bet, then add raise amount
            call_amount = node.current_bet
            pot_after_call = new_pot + call_amount
            
            if action == HoldemAction.RAISE_QUARTER_POT:
                raise_amount = pot_after_call // 4
            elif action == HoldemAction.RAISE_HALF_POT:
                raise_amount = pot_after_call // 2
            elif action == HoldemAction.RAISE_FULL_POT:
                raise_amount = pot_after_call
            elif action == HoldemAction.ALL_IN:
                # All-in: bet entire stack
                total_investment = new_stacks[node.player]
                raise_amount = max(0, total_investment - call_amount)
            else:
                raise_amount = 0
            
            # Total amount to invest: call + raise
            total_investment = call_amount + raise_amount
            actual_investment = min(total_investment, new_stacks[node.player])
            
            new_stacks[node.player] -= actual_investment
            new_pot += actual_investment
            
            # New current bet is the raise amount (what opponent needs to call)
            new_current_bet = actual_investment
            
            return GameNode(
                node_type=NodeType.DECISION,
                player=1 - node.player,  # Next player to act
                hole_cards=node.hole_cards,
                community_cards=node.community_cards,
                street=node.street,
                pot_size=new_pot,
                current_bet=new_current_bet,
                player_stacks=tuple(new_stacks),
                betting_history=new_history,
                is_terminal=False
            )
    
    def _create_showdown_node(self, node: GameNode, history: List, stacks: List, pot: int) -> GameNode:
        """Create a terminal showdown node."""
        hero_cards = node.hole_cards[0]
        villain_cards = node.hole_cards[1]
        
        # Calculate how much each player invested
        player_0_invested = self.initial_stack - stacks[0]
        player_1_invested = self.initial_stack - stacks[1]
        
        try:
            if len(node.community_cards) >= 3:
                # Post-flop showdown - use board cards
                from core.card_utils import calculate_equity_with_board
                equity = calculate_equity_with_board(hero_cards, villain_cards, 
                                                   node.community_cards, num_simulations=100)
            else:
                # Preflop showdown - use preflop equity
                from core.card_utils import calculate_preflop_equity
                equity = calculate_preflop_equity(hero_cards, villain_cards, num_simulations=100)
            
            # Simulate showdown outcome based on equity
            if random.random() < equity:
                # Player 0 wins: gains what Player 1 invested
                utility_0 = player_1_invested
            else:
                # Player 1 wins: Player 0 loses what they invested
                utility_0 = -player_0_invested
                
        except ValueError:
            # Fallback for any card conflicts - random winner
            if random.randint(0, 1) == 0:
                utility_0 = player_1_invested  # Player 0 wins
            else:
                utility_0 = -player_0_invested  # Player 0 loses
        
        return GameNode(
            node_type=NodeType.TERMINAL,
            player=-1,
            hole_cards=node.hole_cards,
            community_cards=node.community_cards,
            street=node.street,
            pot_size=pot,
            current_bet=0,
            player_stacks=tuple(stacks),
            betting_history=history,
            is_terminal=True,
            terminal_utility=utility_0  # Store utility for Player 0
        )
    
    def _get_terminal_utility(self, node: GameNode, traversing_player: int) -> float:
        """Get terminal utility for the traversing player."""
        if traversing_player == 0:
            return node.terminal_utility
        else:
            return -node.terminal_utility  # Zero-sum game
    
    def _handle_chance_node(self, node: GameNode, traversing_player: int,
                          reach_prob_0: float, reach_prob_1: float, iteration: int) -> float:
        """
        Handle chance nodes by sampling community cards and advancing to next street.
        
        In poker, chance nodes occur when dealing community cards (flop, turn, river).
        We sample cards uniformly and continue MCCFR traversal.
        
        Args:
            node: Current chance node
            traversing_player: Player whose regrets we're updating (0 or 1)
            reach_prob_0: Reach probability for player 0
            reach_prob_1: Reach probability for player 1
            iteration: Current MCCFR iteration
            
        Returns:
            Expected utility for the traversing player
        """
        from core.card_utils import deal_flop_cards, deal_turn_card, deal_river_card
        
        # Get currently used cards
        used_cards = set()
        used_cards.update(node.hole_cards[0])  # Player 0 hole cards
        used_cards.update(node.hole_cards[1])  # Player 1 hole cards
        used_cards.update(node.community_cards)    # Existing community cards
        
        # Determine what cards to deal based on current street
        new_community_cards = list(node.community_cards)
        
        try:
            if node.street == Street.PREFLOP:
                # Deal flop (3 cards)
                flop_cards = deal_flop_cards(used_cards)
                new_community_cards.extend(flop_cards)
                next_street = Street.FLOP
                
            elif node.street == Street.FLOP:
                # Deal turn (1 card)
                turn_card = deal_turn_card(used_cards)
                new_community_cards.append(turn_card)
                next_street = Street.TURN
                
            elif node.street == Street.TURN:
                # Deal river (1 card)  
                river_card = deal_river_card(used_cards)
                new_community_cards.append(river_card)
                next_street = Street.RIVER
                
            else:
                # Should not happen - river has no chance nodes
                raise ValueError(f"Invalid street for chance node: {node.street}")
                
        except ValueError as e:
            # Not enough cards available - should not happen in practice
            self.logger.warning(f"Card dealing error: {e}")
            # Return neutral utility
            return 0.0
        
        # Create new node for the next street with dealt cards
        next_node = GameNode(
            node_type=NodeType.DECISION,
            player=0,  # First to act on new street (typically small blind)
            hole_cards=node.hole_cards,
            community_cards=tuple(new_community_cards),
            street=next_street,
            pot_size=node.pot_size,
            current_bet=0,  # Reset betting for new street
            player_stacks=node.player_stacks,
            betting_history=node.betting_history + [(99, 99)],  # Mark street transition
            is_terminal=False,
            terminal_utility=None
        )
        
        # Continue MCCFR traversal with the new node
        # Chance nodes don't change reach probabilities (uniform sampling)
        return self._mccfr_traversal(next_node, traversing_player, 
                                   reach_prob_0, reach_prob_1, iteration)
    
    def _advance_to_next_street(self, node: GameNode) -> GameNode:
        """
        Advance to the next street by creating a chance node.
        
        This is called when a betting round is complete and we need to deal
        community cards for the next street.
        
        Args:
            node: Current node with completed betting round
            
        Returns:
            Chance node for dealing community cards
        """
        # Determine next street
        if node.street == Street.PREFLOP:
            next_street = Street.FLOP
        elif node.street == Street.FLOP:
            next_street = Street.TURN
        elif node.street == Street.TURN:
            next_street = Street.RIVER
        else:
            # River is the final street - should go to showdown
            raise ValueError(f"Cannot advance beyond river street")
        
        # Create chance node for dealing community cards
        chance_node = GameNode(
            node_type=NodeType.CHANCE,
            player=-1,  # No player for chance nodes
            hole_cards=node.hole_cards,
            community_cards=node.community_cards,
            street=node.street,  # Current street (will be updated in chance node handler)
            pot_size=node.pot_size,
            current_bet=node.current_bet,
            player_stacks=node.player_stacks,
            betting_history=node.betting_history,
            is_terminal=False,
            terminal_utility=None
        )
        
        return chance_node
    
    def _is_betting_round_complete(self, node: GameNode) -> bool:
        """
        Check if the current betting round is complete.
        
        A betting round is complete when:
        1. Both players have acted
        2. The current bet is matched by both players (or one folded)
        3. No pending actions remain
        
        Args:
            node: Current game node
            
        Returns:
            True if betting round is complete
        """
        # If there's no betting history, not complete
        if not node.betting_history:
            return False
        
        # Get actions from current street
        current_street_actions = []
        for player_id, action in node.betting_history:
            if player_id != 99:  # Skip street transition markers
                current_street_actions.append((player_id, action))
        
        # Need at least one action to be complete
        if not current_street_actions:
            return False
        
        # If someone folded, round is complete
        from core.holdem_info_set import HoldemAction
        for player_id, action in current_street_actions:
            if action == HoldemAction.FOLD:
                return True
        
        # For heads-up, need both players to have acted with matching bets
        # or for action to come back to the first bettor who then calls/checks
        player_0_acted = any(pid == 0 for pid, _ in current_street_actions)
        player_1_acted = any(pid == 1 for pid, _ in current_street_actions)
        
        if not (player_0_acted and player_1_acted):
            return False
        
        # Check if bets are matched
        # Simplified: if current bet is 0, both must have checked
        # If current bet > 0, both must have called/raised to the same level
        if node.current_bet == 0:
            # Both players checked
            return True
        else:
            # Both players have matched the current bet
            return True
    
    def _is_final_street(self, street: Street) -> bool:
        """Check if this is the final street (river)."""
        return street == Street.RIVER
    
    def _save_checkpoint(self, iteration: int):
        """Save training checkpoint with compression."""
        import gzip
        import time
        
        save_start = time.time()
        checkpoint_path = f"checkpoints/mccfr_checkpoint_{iteration}.pkl"
        
        try:
            # Get memory stats before saving
            stats = self.strategy_profile.get_total_stats()
            self.logger.info(f"Saving checkpoint at iteration {iteration}: {stats['total_info_sets']} info sets")
            
            self.strategy_profile.save(checkpoint_path)
            
            save_time = time.time() - save_start
            self.logger.info(f"Saved checkpoint at iteration {iteration} in {save_time:.1f}s")
            
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")
    
    def get_strategy(self, player: int, hole_cards: Tuple[int, int]) -> np.ndarray:
        """
        Get current strategy for a player with given hole cards.
        
        Args:
            player: Player index (0 or 1)
            hole_cards: Player's hole cards
            
        Returns:
            Strategy array (probabilities for each action)
        """
        # Create a representative info set
        info_set = HoldemInfoSet(
            player=player,
            hole_cards=hole_cards,
            community_cards=(),
            street=Street.PREFLOP,
            betting_history=[],
            position=0 if player == 0 else 1,
            stack_sizes=(self.initial_stack - self.small_blind, 
                        self.initial_stack - self.big_blind),
            pot_size=self.small_blind + self.big_blind,
            current_bet=self.big_blind,
            small_blind=self.small_blind,
            big_blind=self.big_blind
        )
        
        return self.strategy_profile.get_average_strategy(player, info_set)
    
    def evaluate_exploitability(self, num_samples: int = 1000) -> float:
        """
        Estimate exploitability of current strategy.
        
        This is a simplified version - full implementation would require
        best response calculation.
        
        Args:
            num_samples: Number of random hands to sample for evaluation
            
        Returns:
            Estimated exploitability in mbb/hand
        """
        # This is a placeholder for proper exploitability measurement
        # Full implementation requires best response computation
        self.logger.info("Exploitability evaluation not implemented yet")
        return 0.0
    
    def save_strategy(self, filepath: str):
        """Save the trained strategy to file."""
        self.strategy_profile.save(filepath)
        self.logger.info(f"Strategy saved to {filepath}")
    
    def load_strategy(self, filepath: str):
        """Load a previously trained strategy from file."""
        self.strategy_profile.load(filepath)
        self.logger.info(f"Strategy loaded from {filepath}")


# Utility functions for easy training
def train_preflop_bot(iterations: int = 10000, 
                     save_path: str = "trained_strategy",
                     verbose: bool = True) -> HoldemMCCFRTrainer:
    """
    Convenience function to train a preflop heads-up poker bot.
    
    Args:
        iterations: Number of MCCFR iterations
        save_path: Path to save the trained strategy
        verbose: Print training progress
        
    Returns:
        Trained MCCFR trainer instance
    """
    # Setup logging
    if verbose:
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create trainer
    trainer = HoldemMCCFRTrainer(seed=42)
    
    # Train
    stats = trainer.train(iterations, save_every=1000, verbose=verbose)
    
    # Save strategy
    trainer.save_strategy(save_path)
    
    if verbose:
        print(f"\n🎉 Training completed!")
        print(f"📊 Final stats: {stats}")
        print(f"💾 Strategy saved to: {save_path}")
    
    return trainer


if __name__ == "__main__":
    # Example usage
    print("Training MCCFR poker bot...")
    trainer = train_preflop_bot(iterations=1000, verbose=True)
    
    # Test strategy on a few hands
    print("\n🃏 Sample strategies:")
    test_hands = [(48, 49), (0, 1), (48, 44)]  # AA, 22, AKs
    for cards in test_hands:
        strategy = trainer.get_strategy(0, cards)
        from core.card_utils import get_preflop_hand_type
        hand_type = get_preflop_hand_type(cards)
        print(f"{hand_type}: {strategy}")