"""
Regret storage and strategy computation for CFR algorithms.

This module implements the core regret tracking and strategy computation
used in Counterfactual Regret Minimization (CFR) algorithms.
"""

from typing import Dict, List, Tuple, Optional, DefaultDict
from collections import defaultdict
import numpy as np
from core.info_set import InfoSet


class RegretTable:
    """
    Stores cumulative regrets for all information sets and actions.
    
    Implements the core regret storage mechanism for CFR algorithms,
    including regret updates and strategy computation via regret matching.
    """
    
    def __init__(self):
        # regrets[info_set_key][action] = cumulative regret
        self.regrets: DefaultDict[str, DefaultDict[int, float]] = defaultdict(lambda: defaultdict(float))
        
        # strategy_sum[info_set_key][action] = cumulative strategy weight
        self.strategy_sum: DefaultDict[str, DefaultDict[int, float]] = defaultdict(lambda: defaultdict(float))
        
        # Track number of times each info set was visited
        self.visit_counts: DefaultDict[str, int] = defaultdict(int)
        
        # Track total iterations for average strategy computation
        self.total_iterations = 0
    
    def add_regret(self, info_set_key: str, action: int, regret: float) -> None:
        """Add regret for a specific information set and action."""
        self.regrets[info_set_key][action] += regret
    
    def get_regret(self, info_set_key: str, action: int) -> float:
        """Get cumulative regret for information set and action."""
        return self.regrets[info_set_key][action]
    
    def get_strategy(self, info_set_key: str, legal_actions: List[int]) -> Dict[int, float]:
        """
        Compute current strategy using regret matching.
        
        Regret matching formula:
        - If sum of positive regrets > 0: action_prob = max(regret, 0) / sum_positive_regrets
        - Otherwise: uniform random strategy
        
        Args:
            info_set_key: Information set identifier
            legal_actions: List of legal action indices
            
        Returns:
            Dictionary mapping actions to probabilities
        """
        regret_dict = self.regrets[info_set_key]
        
        # Calculate positive regrets
        positive_regrets = {}
        regret_sum = 0.0
        
        for action in legal_actions:
            positive_regret = max(regret_dict[action], 0.0)
            positive_regrets[action] = positive_regret
            regret_sum += positive_regret
        
        # Compute strategy
        strategy = {}
        
        if regret_sum > 0:
            # Proportional to positive regrets
            for action in legal_actions:
                strategy[action] = positive_regrets[action] / regret_sum
        else:
            # Uniform random strategy
            uniform_prob = 1.0 / len(legal_actions)
            for action in legal_actions:
                strategy[action] = uniform_prob
        
        return strategy
    
    def update_strategy_sum(self, info_set_key: str, strategy: Dict[int, float], 
                           reach_probability: float = 1.0) -> None:
        """
        Update cumulative strategy sum for average strategy computation.
        
        Args:
            info_set_key: Information set identifier
            strategy: Current strategy (action probabilities)
            reach_probability: Probability of reaching this information set
        """
        for action, probability in strategy.items():
            weighted_prob = reach_probability * probability
            self.strategy_sum[info_set_key][action] += weighted_prob
        
        # Increment visit count
        self.visit_counts[info_set_key] += 1
    
    def get_average_strategy(self, info_set_key: str, legal_actions: List[int]) -> Dict[int, float]:
        """
        Compute average strategy over all iterations.
        
        This is the final strategy that converges to Nash equilibrium.
        
        Args:
            info_set_key: Information set identifier
            legal_actions: List of legal action indices
            
        Returns:
            Dictionary mapping actions to average probabilities
        """
        strategy_dict = self.strategy_sum[info_set_key]
        
        # Calculate sum of strategy weights
        total_weight = sum(strategy_dict[action] for action in legal_actions)
        
        if total_weight > 0:
            # Normalize strategy weights
            avg_strategy = {}
            for action in legal_actions:
                avg_strategy[action] = strategy_dict[action] / total_weight
        else:
            # Uniform strategy if no data
            uniform_prob = 1.0 / len(legal_actions)
            avg_strategy = {action: uniform_prob for action in legal_actions}
        
        return avg_strategy
    
    def get_all_average_strategies(self, info_sets: List[InfoSet]) -> Dict[str, Dict[int, float]]:
        """Get average strategies for all information sets."""
        strategies = {}
        for info_set in info_sets:
            info_set_key = info_set.to_string()
            legal_actions = info_set.get_legal_actions()
            strategies[info_set_key] = self.get_average_strategy(info_set_key, legal_actions)
        return strategies
    
    def clear(self) -> None:
        """Clear all stored regrets and strategies."""
        self.regrets.clear()
        self.strategy_sum.clear()
        self.visit_counts.clear()
        self.total_iterations = 0
    
    def increment_iteration(self) -> None:
        """Increment the total iteration counter."""
        self.total_iterations += 1
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about the regret table."""
        num_info_sets = len(self.regrets)
        total_visits = sum(self.visit_counts.values())
        
        # Calculate regret statistics
        all_regrets = []
        for info_set_regrets in self.regrets.values():
            all_regrets.extend(info_set_regrets.values())
        
        regret_stats = {}
        if all_regrets:
            regret_stats = {
                'mean': np.mean(all_regrets),
                'std': np.std(all_regrets),
                'min': np.min(all_regrets),
                'max': np.max(all_regrets)
            }
        
        return {
            'total_iterations': self.total_iterations,
            'num_info_sets': num_info_sets,
            'total_visits': total_visits,
            'avg_visits_per_info_set': total_visits / max(num_info_sets, 1),
            'regret_stats': regret_stats
        }
    
    def print_strategies(self, info_sets: List[InfoSet], strategy_type: str = 'average') -> None:
        """Print strategies for all information sets."""
        print(f"\n{strategy_type.title()} Strategies:")
        print("=" * 50)
        
        for info_set in sorted(info_sets, key=lambda x: x.to_string()):
            info_set_key = info_set.to_string()
            legal_actions = info_set.get_legal_actions()
            
            if strategy_type == 'average':
                strategy = self.get_average_strategy(info_set_key, legal_actions)
            else:
                strategy = self.get_strategy(info_set_key, legal_actions)
            
            print(f"{info_set_key}:")
            for action, prob in strategy.items():
                if hasattr(info_set, 'get_action_meaning'):
                    action_name = info_set.get_action_meaning(action)
                    print(f"  {action_name}: {prob:.4f}")
                else:
                    print(f"  Action {action}: {prob:.4f}")
            
            # Show regrets for debugging
            if strategy_type == 'current':
                regret_dict = self.regrets[info_set_key]
                regret_str = ", ".join(f"{a}: {regret_dict[a]:.2f}" for a in legal_actions)
                print(f"  Regrets: [{regret_str}]")
            
            print()


class StrategyProfile:
    """
    Represents a complete strategy profile for all players.
    
    A strategy profile contains strategies for all information sets
    of all players in the game.
    """
    
    def __init__(self, regret_table: RegretTable, info_sets: List[InfoSet]):
        self.regret_table = regret_table
        self.strategies = {}
        
        # Compute average strategies for all information sets
        for info_set in info_sets:
            info_set_key = info_set.to_string()
            legal_actions = info_set.get_legal_actions()
            self.strategies[info_set_key] = regret_table.get_average_strategy(
                info_set_key, legal_actions
            )
    
    def get_strategy(self, info_set_key: str) -> Dict[int, float]:
        """Get strategy for a specific information set."""
        return self.strategies.get(info_set_key, {})
    
    def get_action_probability(self, info_set_key: str, action: int) -> float:
        """Get probability of taking a specific action."""
        strategy = self.get_strategy(info_set_key)
        return strategy.get(action, 0.0)
    
    def sample_action(self, info_set_key: str, legal_actions: List[int], 
                     rng: Optional[np.random.Generator] = None) -> int:
        """Sample an action according to the strategy."""
        if rng is None:
            rng = np.random.default_rng()
        
        strategy = self.get_strategy(info_set_key)
        
        if not strategy:
            # Uniform random if no strategy
            return rng.choice(legal_actions)
        
        # Create probability array
        probabilities = [strategy.get(action, 0.0) for action in legal_actions]
        
        # Normalize probabilities (in case of numerical errors)
        prob_sum = sum(probabilities)
        if prob_sum > 0:
            probabilities = [p / prob_sum for p in probabilities]
        else:
            # Uniform if all probabilities are zero
            probabilities = [1.0 / len(legal_actions)] * len(legal_actions)
        
        # Sample action
        return rng.choice(legal_actions, p=probabilities)


if __name__ == "__main__":
    # Test the regret storage implementation
    print("Testing Regret Storage and Strategy Computation")
    print("=" * 50)
    
    # Create regret table
    regret_table = RegretTable()
    
    # Test basic functionality
    info_set_key = "1/"
    legal_actions = [0, 1]
    
    # Add some regrets
    regret_table.add_regret(info_set_key, 0, 2.0)
    regret_table.add_regret(info_set_key, 1, -1.0)
    
    print(f"Regrets for {info_set_key}:")
    for action in legal_actions:
        regret = regret_table.get_regret(info_set_key, action)
        print(f"  Action {action}: {regret}")
    
    # Test strategy computation
    strategy = regret_table.get_strategy(info_set_key, legal_actions)
    print(f"\nStrategy for {info_set_key}:")
    for action, prob in strategy.items():
        print(f"  Action {action}: {prob:.4f}")
    
    # Test strategy sum update
    regret_table.update_strategy_sum(info_set_key, strategy, reach_probability=0.5)
    
    # Test average strategy
    avg_strategy = regret_table.get_average_strategy(info_set_key, legal_actions)
    print(f"\nAverage strategy for {info_set_key}:")
    for action, prob in avg_strategy.items():
        print(f"  Action {action}: {prob:.4f}")
    
    # Test statistics
    regret_table.increment_iteration()
    stats = regret_table.get_statistics()
    print(f"\nRegret table statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Regret storage tests completed!")