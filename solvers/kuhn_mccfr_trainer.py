"""
Monte Carlo Counterfactual Regret Minimization (MCCFR) Trainer.

This module implements external sampling MCCFR for Kuhn poker,
following the algorithm from research literature.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Tuple, Optional
import random
import numpy as np
from tqdm import tqdm

from envs.kuhn_poker import KuhnPokerEnv, Action
from core.info_set import KuhnInfoSet, InfoSetManager, info_set_from_game_state
from core.regret_storage import RegretTable, StrategyProfile


class MCCFRTrainer:
    """
    External Sampling Monte Carlo CFR trainer for Kuhn poker.
    
    Implements the external sampling variant where:
    - Opponent actions are sampled
    - Chance actions (card deals) are sampled
    - All actions for the traversing player are considered
    
    This provides a good balance between efficiency and low variance.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize MCCFR trainer.
        
        Args:
            seed: Random seed for reproducible results
        """
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.rng = np.random.default_rng(seed)
        
        # Core components
        self.env = KuhnPokerEnv()
        self.info_set_manager = InfoSetManager()
        self.regret_table = RegretTable()
        
        # Generate all possible information sets for Kuhn poker
        self.info_set_manager.generate_kuhn_poker_info_sets()
        
        # Training statistics
        self.iteration_count = 0
        self.total_nodes_touched = 0
        
        # Performance tracking
        self.exploitability_history = []
        self.strategy_history = []
    
    def train(self, iterations: int, log_interval: int = 1000, 
              eval_interval: int = 5000, verbose: bool = True) -> StrategyProfile:
        """
        Train MCCFR for specified number of iterations.
        
        Args:
            iterations: Number of MCCFR iterations to run
            log_interval: How often to log progress
            eval_interval: How often to evaluate exploitability (0 to disable)
            verbose: Whether to print progress information
            
        Returns:
            Final strategy profile
        """
        if verbose:
            print(f"Starting MCCFR training for {iterations} iterations")
            print(f"Info sets: P0={len(self.info_set_manager.get_player_info_sets(0))}, "
                  f"P1={len(self.info_set_manager.get_player_info_sets(1))}")
        
        # Progress bar
        pbar = tqdm(range(iterations), desc="MCCFR Training") if verbose else range(iterations)
        
        for iteration in pbar:
            # Alternate which player is the traversing player
            traversing_player = iteration % 2
            
            # Run one MCCFR iteration
            self._mccfr_iteration(traversing_player)
            
            self.iteration_count += 1
            self.regret_table.increment_iteration()
            
            # Logging and evaluation
            if verbose and (iteration + 1) % log_interval == 0:
                stats = self.regret_table.get_statistics()
                pbar.set_postfix({
                    'Nodes': self.total_nodes_touched,
                    'Regret_std': f"{stats['regret_stats'].get('std', 0):.3f}"
                })
            
            # Periodic exploitability evaluation
            if eval_interval > 0 and (iteration + 1) % eval_interval == 0:
                if verbose:
                    tqdm.write(f"Iteration {iteration + 1}: Evaluating exploitability...")
                
                strategy_profile = self.get_strategy_profile()
                # Note: Exploitability evaluation will be implemented separately
                # self.exploitability_history.append(exploitability)
        
        if verbose:
            print(f"\nTraining completed! Total nodes touched: {self.total_nodes_touched}")
        
        return self.get_strategy_profile()
    
    def _mccfr_iteration(self, traversing_player: int) -> None:
        """
        Run one iteration of external sampling MCCFR.
        
        Args:
            traversing_player: Player for whom we're updating regrets (0 or 1)
        """
        # Sample a random card deal
        self.env.reset(seed=None)  # Random deal each iteration
        
        # Start traversal from root with reach probabilities = 1
        self._traverse(self.env, traversing_player, reach_prob_p0=1.0, reach_prob_p1=1.0)
    
    def _traverse(self, env: KuhnPokerEnv, traversing_player: int, 
                  reach_prob_p0: float, reach_prob_p1: float) -> float:
        """
        Recursive traversal function for external sampling MCCFR.
        
        Args:
            env: Current game state
            traversing_player: Player being trained (0 or 1)
            reach_prob_p0: Reach probability for player 0
            reach_prob_p1: Reach probability for player 1
            
        Returns:
            Utility for the traversing player
        """
        self.total_nodes_touched += 1
        
        # Terminal node - return utility
        if env.is_terminal_state:
            return env.get_payoff(traversing_player)
        
        # Get current player and information set
        current_player = env.current_player
        info_set = info_set_from_game_state(env, current_player)
        info_set_key = info_set.to_string()
        legal_actions = info_set.get_legal_actions()
        
        # Get current strategy using regret matching
        strategy = self.regret_table.get_strategy(info_set_key, legal_actions)
        
        if current_player == traversing_player:
            # Traversing player: consider all actions
            return self._handle_traversing_player(
                env, info_set, info_set_key, legal_actions, strategy, 
                traversing_player, reach_prob_p0, reach_prob_p1
            )
        else:
            # Non-traversing player: sample action according to strategy
            return self._handle_sampling_player(
                env, strategy, legal_actions, traversing_player, 
                reach_prob_p0, reach_prob_p1, current_player
            )
    
    def _handle_traversing_player(self, env: KuhnPokerEnv, info_set: KuhnInfoSet, 
                                  info_set_key: str, legal_actions: List[int], 
                                  strategy: Dict[int, float], traversing_player: int,
                                  reach_prob_p0: float, reach_prob_p1: float) -> float:
        """Handle node where traversing player acts (update regrets)."""
        current_player = env.current_player
        
        # Calculate reach probability for the traversing player
        reach_prob_traversing = reach_prob_p0 if traversing_player == 0 else reach_prob_p1
        reach_prob_opponent = reach_prob_p1 if traversing_player == 0 else reach_prob_p0
        
        # Compute utilities for each action
        action_utilities = {}
        node_utility = 0.0
        
        for action in legal_actions:
            # Clone environment and take action
            child_env = env.clone()
            child_env.step(action)
            
            # Calculate new reach probabilities
            new_reach_p0 = reach_prob_p0
            new_reach_p1 = reach_prob_p1
            
            if current_player == 0:
                new_reach_p0 *= strategy[action]
            else:
                new_reach_p1 *= strategy[action]
            
            # Recursively traverse child
            action_utility = self._traverse(
                child_env, traversing_player, new_reach_p0, new_reach_p1
            )
            
            action_utilities[action] = action_utility
            node_utility += strategy[action] * action_utility
        
        # Update regrets
        for action in legal_actions:
            regret = action_utilities[action] - node_utility
            weighted_regret = reach_prob_opponent * regret
            self.regret_table.add_regret(info_set_key, action, weighted_regret)
        
        # Update strategy sum (for average strategy computation)
        self.regret_table.update_strategy_sum(
            info_set_key, strategy, reach_prob_traversing
        )
        
        return node_utility
    
    def _handle_sampling_player(self, env: KuhnPokerEnv, strategy: Dict[int, float], 
                               legal_actions: List[int], traversing_player: int,
                               reach_prob_p0: float, reach_prob_p1: float, 
                               current_player: int) -> float:
        """Handle node where non-traversing player acts (sample action)."""
        
        # Sample action according to strategy
        probabilities = [strategy[action] for action in legal_actions]
        sampled_action = self.rng.choice(legal_actions, p=probabilities)
        
        # Take the sampled action
        env.step(sampled_action)
        
        # Update reach probabilities (only the acting player's probability changes)
        new_reach_p0 = reach_prob_p0
        new_reach_p1 = reach_prob_p1
        
        if current_player == 0:
            new_reach_p0 *= strategy[sampled_action]
        else:
            new_reach_p1 *= strategy[sampled_action]
        
        # Continue traversal
        return self._traverse(env, traversing_player, new_reach_p0, new_reach_p1)
    
    def get_strategy_profile(self) -> StrategyProfile:
        """Get the current average strategy profile."""
        all_info_sets = self.info_set_manager.get_all_info_sets()
        return StrategyProfile(self.regret_table, all_info_sets)
    
    def print_strategies(self, strategy_type: str = 'average') -> None:
        """Print current strategies for all information sets."""
        all_info_sets = self.info_set_manager.get_all_info_sets()
        self.regret_table.print_strategies(all_info_sets, strategy_type)
    
    def get_statistics(self) -> Dict[str, any]:
        """Get training statistics."""
        base_stats = self.regret_table.get_statistics()
        base_stats.update({
            'mccfr_iterations': self.iteration_count,
            'total_nodes_touched': self.total_nodes_touched,
            'avg_nodes_per_iteration': self.total_nodes_touched / max(self.iteration_count, 1)
        })
        return base_stats
    
    def save_strategy(self, filepath: str) -> None:
        """Save the current strategy to file."""
        # Implementation for saving strategy would go here
        # For now, just save the average strategies as text
        strategy_profile = self.get_strategy_profile()
        
        with open(filepath, 'w') as f:
            f.write(f"MCCFR Strategy Profile\n")
            f.write(f"Iterations: {self.iteration_count}\n")
            f.write(f"Total nodes touched: {self.total_nodes_touched}\n\n")
            
            all_info_sets = self.info_set_manager.get_all_info_sets()
            for info_set in sorted(all_info_sets, key=lambda x: x.to_string()):
                info_set_key = info_set.to_string()
                strategy = strategy_profile.get_strategy(info_set_key)
                
                f.write(f"{info_set_key}:\n")
                for action, prob in strategy.items():
                    action_name = info_set.get_action_meaning(action)
                    f.write(f"  {action_name}: {prob:.6f}\n")
                f.write("\n")
    
    def reset(self) -> None:
        """Reset the trainer to initial state."""
        self.regret_table.clear()
        self.iteration_count = 0
        self.total_nodes_touched = 0
        self.exploitability_history.clear()
        self.strategy_history.clear()


def run_quick_test():
    """Run a quick test of the MCCFR trainer."""
    print("Quick MCCFR Test")
    print("=" * 30)
    
    trainer = MCCFRTrainer(seed=42)
    
    # Train for a small number of iterations
    strategy_profile = trainer.train(iterations=1000, log_interval=500, verbose=True)
    
    # Print final strategies
    trainer.print_strategies('average')
    
    # Print statistics
    stats = trainer.get_statistics()
    print(f"\nTraining Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return trainer, strategy_profile


if __name__ == "__main__":
    run_quick_test()