"""
Exploitability evaluator for poker strategies.

This module implements exploitability calculation using best response
computation to measure how much a perfect adversary could exploit
a given strategy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

from envs.kuhn_poker import KuhnPokerEnv, Action
from core.info_set import KuhnInfoSet, InfoSetManager, info_set_from_game_state
from core.regret_storage import StrategyProfile


class BestResponseCalculator:
    """
    Calculates best response to a given strategy in Kuhn poker.
    
    Uses backward induction to find the optimal counter-strategy
    against a fixed opponent strategy.
    """
    
    def __init__(self):
        self.env = KuhnPokerEnv()
        self.info_set_manager = InfoSetManager()
        self.info_set_manager.generate_kuhn_poker_info_sets()
    
    def compute_best_response(self, opponent_strategy: StrategyProfile, 
                            responding_player: int) -> Tuple[Dict[str, Dict[int, float]], float]:
        """
        Compute best response strategy and expected value against opponent.
        
        Args:
            opponent_strategy: Fixed strategy for the opponent
            responding_player: Player computing best response (0 or 1)
            
        Returns:
            Tuple of (best_response_strategy, expected_value)
        """
        opponent_player = 1 - responding_player
        
        # Initialize value table for all information sets of responding player
        info_set_values = {}
        best_response_strategy = {}
        
        # Get all possible card deals
        all_deals = self.env.get_all_possible_deals()
        
        # Compute expected value for each information set by enumerating all possibilities
        for info_set in self.info_set_manager.get_player_info_sets(responding_player):
            info_set_key = info_set.to_string()
            legal_actions = info_set.get_legal_actions()
            
            # Compute expected value for each action
            action_values = {}
            
            for action in legal_actions:
                # Compute expected value of taking this action
                expected_value = self._compute_action_expected_value(
                    info_set, action, opponent_strategy, responding_player, opponent_player
                )
                action_values[action] = expected_value
            
            # Best response: take action with highest expected value
            if action_values:
                best_action = max(action_values.keys(), key=lambda a: action_values[a])
                best_value = action_values[best_action]
                
                # Create pure strategy (probability 1 for best action, 0 for others)
                strategy = {action: 0.0 for action in legal_actions}
                strategy[best_action] = 1.0
                
                best_response_strategy[info_set_key] = strategy
                info_set_values[info_set_key] = best_value
        
        # Compute overall expected value of best response strategy
        overall_expected_value = self._compute_strategy_expected_value(
            best_response_strategy, opponent_strategy, responding_player
        )
        
        return best_response_strategy, overall_expected_value
    
    def _compute_action_expected_value(self, info_set: KuhnInfoSet, action: int,
                                     opponent_strategy: StrategyProfile, 
                                     responding_player: int, opponent_player: int) -> float:
        """Compute expected value of taking a specific action from an information set."""
        
        card = info_set.card
        history = info_set.history
        
        # Get all possible opponent cards
        opponent_cards = [c for c in [1, 2, 3] if c != card]
        
        total_value = 0.0
        total_weight = 0.0
        
        for opponent_card in opponent_cards:
            # Set up game state
            self.env.reset()
            if responding_player == 0:
                self.env.set_cards(card, opponent_card)
            else:
                self.env.set_cards(opponent_card, card)
            
            # Replay history to get to current state
            for action_char in history:
                if action_char == 'C':
                    self.env.step(Action.CHECK_CALL.value)
                elif action_char == 'B':
                    self.env.step(Action.BET_FOLD.value)
                elif action_char == 'F':
                    self.env.step(Action.BET_FOLD.value)
            
            # Take the specified action
            env_copy = self.env.clone()
            env_copy.step(action)
            
            # Compute expected continuation value
            continuation_value = self._compute_continuation_value(
                env_copy, opponent_strategy, responding_player, opponent_player
            )
            
            total_value += continuation_value
            total_weight += 1.0
        
        return total_value / max(total_weight, 1.0)
    
    def _compute_continuation_value(self, env: KuhnPokerEnv, opponent_strategy: StrategyProfile,
                                  responding_player: int, opponent_player: int) -> float:
        """Compute expected value from current state onwards."""
        
        if env.is_terminal_state:
            return env.get_payoff(responding_player)
        
        current_player = env.current_player
        
        if current_player == responding_player:
            # Responding player's turn - we'll handle this in the outer optimization
            # For now, assume they play optimally from here (this is a simplification)
            legal_actions = env.get_legal_actions()
            best_value = float('-inf')
            
            for action in legal_actions:
                env_copy = env.clone()
                env_copy.step(action)
                value = self._compute_continuation_value(
                    env_copy, opponent_strategy, responding_player, opponent_player
                )
                best_value = max(best_value, value)
            
            return best_value if best_value != float('-inf') else 0.0
        
        else:
            # Opponent's turn - use their fixed strategy
            opponent_info_set = info_set_from_game_state(env, current_player)
            opponent_info_set_key = opponent_info_set.to_string()
            opponent_strategy_dict = opponent_strategy.get_strategy(opponent_info_set_key)
            
            if not opponent_strategy_dict:
                # If no strategy available, assume uniform random
                legal_actions = env.get_legal_actions()
                prob = 1.0 / len(legal_actions)
                opponent_strategy_dict = {action: prob for action in legal_actions}
            
            # Compute expected value over opponent's actions
            expected_value = 0.0
            for action, probability in opponent_strategy_dict.items():
                if probability > 0:
                    env_copy = env.clone()
                    env_copy.step(action)
                    value = self._compute_continuation_value(
                        env_copy, opponent_strategy, responding_player, opponent_player
                    )
                    expected_value += probability * value
            
            return expected_value
    
    def _compute_strategy_expected_value(self, best_response_strategy: Dict[str, Dict[int, float]],
                                       opponent_strategy: StrategyProfile, 
                                       responding_player: int) -> float:
        """Compute overall expected value of a strategy."""
        
        total_value = 0.0
        total_weight = 0.0
        
        # Enumerate all possible game scenarios
        all_deals = self.env.get_all_possible_deals()
        
        for p0_card, p1_card in all_deals:
            self.env.reset()
            self.env.set_cards(p0_card, p1_card)
            
            # Simulate game with both strategies
            game_value = self._simulate_game_with_strategies(
                self.env, best_response_strategy, opponent_strategy, responding_player
            )
            
            total_value += game_value
            total_weight += 1.0
        
        return total_value / max(total_weight, 1.0)
    
    def _simulate_game_with_strategies(self, env: KuhnPokerEnv, 
                                     responding_strategy: Dict[str, Dict[int, float]],
                                     opponent_strategy: StrategyProfile, 
                                     responding_player: int) -> float:
        """Simulate a single game with given strategies."""
        
        env_copy = env.clone()
        
        while not env_copy.is_terminal_state:
            current_player = env_copy.current_player
            info_set = info_set_from_game_state(env_copy, current_player)
            info_set_key = info_set.to_string()
            
            if current_player == responding_player:
                # Use best response strategy
                strategy = responding_strategy.get(info_set_key, {})
            else:
                # Use opponent strategy
                strategy = opponent_strategy.get_strategy(info_set_key)
            
            # Sample action according to strategy
            legal_actions = env_copy.get_legal_actions()
            
            if strategy:
                # Weighted random choice
                actions = list(strategy.keys())
                probabilities = [strategy[a] for a in actions]
                if sum(probabilities) > 0:
                    probabilities = [p / sum(probabilities) for p in probabilities]
                    action = np.random.choice(actions, p=probabilities)
                else:
                    action = np.random.choice(legal_actions)
            else:
                # Uniform random
                action = np.random.choice(legal_actions)
            
            env_copy.step(action)
        
        return env_copy.get_payoff(responding_player)


class ExploitabilityEvaluator:
    """
    Evaluates exploitability of poker strategies.
    
    Exploitability is defined as:
    exploitability(σ) = max_σ'[u(σ', σ)] - u(σ, σ)
    
    Where σ is the strategy to evaluate, σ' is any counter-strategy,
    and u(σ1, σ2) is the expected utility of σ1 against σ2.
    """
    
    def __init__(self):
        self.best_response_calc = BestResponseCalculator()
    
    def compute_exploitability(self, strategy: StrategyProfile) -> Dict[str, float]:
        """
        Compute exploitability for both players.
        
        Args:
            strategy: Strategy profile to evaluate
            
        Returns:
            Dictionary with exploitability for each player and total
        """
        results = {}
        
        # Compute best response for each player
        for player in [0, 1]:
            best_response_strategy, best_response_value = self.best_response_calc.compute_best_response(
                strategy, player
            )
            
            # Compute value of original strategy against itself
            # (This should be close to 0 for a Nash equilibrium)
            original_value = self._compute_strategy_value_against_itself(strategy, player)
            
            # Exploitability = best_response_value - original_value
            exploitability = best_response_value - original_value
            
            results[f'player_{player}_exploitability'] = exploitability
            results[f'player_{player}_best_response_value'] = best_response_value
            results[f'player_{player}_original_value'] = original_value
        
        # Total exploitability (sum of both players' exploitabilities)
        total_exploitability = (results['player_0_exploitability'] + 
                              results['player_1_exploitability'])
        results['total_exploitability'] = total_exploitability
        
        # Convert to milliblinds per hand (mbb/h)
        # In Kuhn poker, each hand starts with 1 chip ante per player
        results['exploitability_mbb_per_hand'] = total_exploitability * 1000
        
        return results
    
    def _compute_strategy_value_against_itself(self, strategy: StrategyProfile, 
                                             player: int) -> float:
        """Compute expected value of strategy against itself for given player."""
        
        total_value = 0.0
        total_scenarios = 0
        
        env = KuhnPokerEnv()
        all_deals = env.get_all_possible_deals()
        
        for p0_card, p1_card in all_deals:
            env.reset()
            env.set_cards(p0_card, p1_card)
            
            # Simulate game where both players use the same strategy
            game_value = self._simulate_self_play_game(env, strategy, player)
            total_value += game_value
            total_scenarios += 1
        
        return total_value / max(total_scenarios, 1)
    
    def _simulate_self_play_game(self, env: KuhnPokerEnv, strategy: StrategyProfile, 
                               target_player: int) -> float:
        """Simulate game where both players use the same strategy."""
        
        env_copy = env.clone()
        
        while not env_copy.is_terminal_state:
            current_player = env_copy.current_player
            info_set = info_set_from_game_state(env_copy, current_player)
            info_set_key = info_set.to_string()
            
            # Both players use the same strategy
            strategy_dict = strategy.get_strategy(info_set_key)
            legal_actions = env_copy.get_legal_actions()
            
            if strategy_dict:
                # Sample action according to strategy
                actions = list(strategy_dict.keys())
                probabilities = [strategy_dict[a] for a in actions]
                if sum(probabilities) > 0:
                    probabilities = [p / sum(probabilities) for p in probabilities]
                    action = np.random.choice(actions, p=probabilities)
                else:
                    action = np.random.choice(legal_actions)
            else:
                # Uniform random if no strategy
                action = np.random.choice(legal_actions)
            
            env_copy.step(action)
        
        return env_copy.get_payoff(target_player)
    
    def print_exploitability_report(self, results: Dict[str, float]) -> None:
        """Print a formatted exploitability report."""
        
        print("\nExploitability Report")
        print("=" * 40)
        print(f"Player 0 exploitability: {results['player_0_exploitability']:.6f}")
        print(f"Player 1 exploitability: {results['player_1_exploitability']:.6f}")
        print(f"Total exploitability: {results['total_exploitability']:.6f}")
        print(f"Exploitability (mbb/hand): {results['exploitability_mbb_per_hand']:.3f}")
        
        # Provide interpretation
        mbb_per_hand = results['exploitability_mbb_per_hand']
        if mbb_per_hand < 0.1:
            level = "Professional level (< 0.1 mbb/h)"
        elif mbb_per_hand < 1.0:
            level = "Strong level (< 1.0 mbb/h)"
        elif mbb_per_hand < 10.0:
            level = "Good level (< 10.0 mbb/h)"
        else:
            level = "Needs improvement (> 10.0 mbb/h)"
        
        print(f"Strategy strength: {level}")


def run_exploitability_test():
    """Test the exploitability evaluator with a simple strategy."""
    print("Testing Exploitability Evaluator")
    print("=" * 40)
    
    # Create a simple uniform random strategy for testing
    from core.regret_storage import RegretTable
    from core.info_set import InfoSetManager
    
    info_set_manager = InfoSetManager()
    info_set_manager.generate_kuhn_poker_info_sets()
    regret_table = RegretTable()
    
    # Create uniform strategy (should be highly exploitable)
    all_info_sets = info_set_manager.get_all_info_sets()
    for info_set in all_info_sets:
        info_set_key = info_set.to_string()
        legal_actions = info_set.get_legal_actions()
        uniform_prob = 1.0 / len(legal_actions)
        
        for action in legal_actions:
            regret_table.strategy_sum[info_set_key][action] = uniform_prob
    
    from core.regret_storage import StrategyProfile
    uniform_strategy = StrategyProfile(regret_table, all_info_sets)
    
    # Evaluate exploitability
    evaluator = ExploitabilityEvaluator()
    results = evaluator.compute_exploitability(uniform_strategy)
    evaluator.print_exploitability_report(results)
    
    return evaluator, results


if __name__ == "__main__":
    run_exploitability_test()