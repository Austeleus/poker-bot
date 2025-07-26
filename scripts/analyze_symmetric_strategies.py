#!/usr/bin/env python3
"""
Symmetric Strategy Analysis for MCCFR Poker Bot.

This script analyzes the symmetrical training results, showing:
- Player 0 strategies when SB and when BB  
- Player 1 strategies when SB and when BB
- Convergence analysis between players for same positions
- Exploitability calculation in mbb/hand
- Strategy comparison and differences

Usage:
    python scripts/analyze_symmetric_strategies.py checkpoints/model_name
"""

import sys
import os
import argparse
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.holdem_mccfr_trainer import HoldemMCCFRTrainer
from core.card_utils import get_preflop_hand_type, get_preflop_hand_strength, card_to_string
from core.holdem_info_set import HoldemInfoSet, Street, BettingRound, HoldemAction
from evaluation.exploitability_calculator import ExploitabilityCalculator


class SymmetricStrategyAnalyzer:
    """Analyzer for symmetric MCCFR strategies with exploitability calculation."""
    
    def __init__(self, model_path: str):
        """Initialize with trained model."""
        self.model_path = model_path
        
        # Initialize trainer
        self.trainer = HoldemMCCFRTrainer(
            small_blind=1,
            big_blind=2, 
            initial_stack=200
        )
        
        try:
            self.trainer.load_strategy(model_path)
            print(f"✅ Loaded strategy from: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load strategy: {e}")
            sys.exit(1)
            
        # Action mapping
        self.action_names = {
            0: "Fold",
            1: "Call", 
            2: "Raise1/4",
            3: "Raise1/2", 
            4: "RaisePot",
            5: "All-in"
        }
        
        # Position names
        self.position_names = {0: "SB", 1: "BB"}
        
        # Generate test hands
        self.test_hands = self._generate_test_hands()
        
    def _generate_test_hands(self) -> List[Tuple[Tuple[int, int], str]]:
        """Generate representative hands for analysis."""
        hands = []
        
        # Key representative hands 
        key_hands = [
            (48, 49),   # AA
            (44, 45),   # KK 
            (40, 41),   # QQ
            (36, 37),   # JJ
            (32, 33),   # TT
            (48, 44),   # AKs
            (48, 47),   # AKo
            (48, 40),   # AQs
            (44, 40),   # KQs
            (32, 28),   # T9s
            (28, 24),   # 98s
            (0, 1),     # 22
            (0, 17),    # 62o
        ]
        
        for cards in key_hands:
            hand_type = get_preflop_hand_type(cards)
            hands.append((cards, hand_type))
            
        return hands
    
    def get_strategy_for_position(self, player: int, cards: Tuple[int, int], 
                                 position: int) -> np.ndarray:
        """
        Get strategy for a specific player, cards, and position.
        
        Args:
            player: Player index (0 or 1)
            cards: Hole cards
            position: 0=SB, 1=BB
            
        Returns:
            Strategy array
        """
        # Create info set with specific position
        # Determine stack sizes based on position
        if position == 0:  # SB position
            stack_sizes = (199, 198) if player == 0 else (198, 199)
        else:  # BB position  
            stack_sizes = (198, 199) if player == 0 else (199, 198)
            
        info_set = HoldemInfoSet(
            player=player,
            hole_cards=cards,
            community_cards=(),
            street=Street.PREFLOP,
            betting_history=[],
            position=position,
            stack_sizes=stack_sizes,
            pot_size=3,  # SB + BB
            current_bet=2 if position == 0 else 0,  # SB faces BB, BB can check
            small_blind=1,
            big_blind=2
        )
        
        return self.trainer.strategy_profile.get_average_strategy(player, info_set)
    
    def analyze_positional_strategies(self):
        """Analyze strategies for each player in each position."""
        print(f"\n🎯 Positional Strategy Analysis")
        print("=" * 80)
        
        for position in [0, 1]:  # SB, BB
            pos_name = self.position_names[position]
            print(f"\n📍 {pos_name} Position Strategies")
            print("-" * 50)
            print("Hand     | Player 0 Action  | P0 Prob | Player 1 Action  | P1 Prob | Difference")
            print("-" * 75)
            
            total_difference = 0.0
            convergent_hands = 0
            
            for cards, hand_type in self.test_hands:
                try:
                    # Get strategies for both players in this position
                    strategy_p0 = self.get_strategy_for_position(0, cards, position)
                    strategy_p1 = self.get_strategy_for_position(1, cards, position)
                    
                    # Best actions
                    best_p0 = np.argmax(strategy_p0)
                    best_p1 = np.argmax(strategy_p1)
                    
                    action_p0 = self.action_names[best_p0]
                    action_p1 = self.action_names[best_p1]
                    
                    prob_p0 = strategy_p0[best_p0]
                    prob_p1 = strategy_p1[best_p1]
                    
                    # L1 distance between strategies
                    difference = np.sum(np.abs(strategy_p0 - strategy_p1))
                    total_difference += difference
                    
                    # Check convergence (strategies are similar)
                    if difference < 0.2:  # Threshold for convergence
                        convergent_hands += 1
                        conv_marker = "✓"
                    else:
                        conv_marker = "✗"
                    
                    print(f"{hand_type:8} | {action_p0:15} | {prob_p0:.3f}  | {action_p1:15} | {prob_p1:.3f}  | {difference:.3f} {conv_marker}")
                    
                except Exception as e:
                    print(f"{hand_type:8} | Error: {str(e)[:50]}")
            
            avg_difference = total_difference / len(self.test_hands)
            convergence_rate = convergent_hands / len(self.test_hands)
            
            print(f"\n{pos_name} Summary:")
            print(f"  Average strategy difference: {avg_difference:.3f}")
            print(f"  Convergence rate: {convergence_rate:.1%}")
    
    def compare_positions_within_player(self, player: int):
        """Compare SB vs BB strategies for a single player."""
        print(f"\n👤 Player {player} Position Comparison")
        print("-" * 50)
        print("Hand     | SB Action       | SB Prob | BB Action       | BB Prob | Diff")
        print("-" * 70)
        
        for cards, hand_type in self.test_hands:
            try:
                strategy_sb = self.get_strategy_for_position(player, cards, 0)  # SB
                strategy_bb = self.get_strategy_for_position(player, cards, 1)  # BB
                
                best_sb = np.argmax(strategy_sb)
                best_bb = np.argmax(strategy_bb)
                
                action_sb = self.action_names[best_sb]
                action_bb = self.action_names[best_bb]
                
                prob_sb = strategy_sb[best_sb]
                prob_bb = strategy_bb[best_bb]
                
                difference = np.sum(np.abs(strategy_sb - strategy_bb))
                
                print(f"{hand_type:8} | {action_sb:14} | {prob_sb:.3f}  | {action_bb:14} | {prob_bb:.3f}  | {difference:.3f}")
                
            except Exception as e:
                print(f"{hand_type:8} | Error: {str(e)[:50]}")
    
    def calculate_exploitability(self, num_samples: int = 500) -> Dict[str, float]:
        """
        Calculate exploitability in mbb/hand using best response analysis.
        
        Args:
            num_samples: Number of hands to sample for evaluation
            
        Returns:
            Dictionary with exploitability metrics
        """
        print(f"\n📊 Exploitability Analysis")
        print("=" * 50)
        
        # Initialize exploitability calculator
        calculator = ExploitabilityCalculator(self.trainer, num_samples=num_samples)
        
        # Calculate exploitability
        exploitability_metrics = calculator.calculate_exploitability()
        
        # Display results
        print(f"Player 0 Exploitability: {exploitability_metrics['player_0_exploitability_mbb']:.1f} mbb/hand")
        print(f"Player 1 Exploitability: {exploitability_metrics['player_1_exploitability_mbb']:.1f} mbb/hand")
        print(f"Total Exploitability: {exploitability_metrics['total_exploitability_mbb']:.1f} mbb/hand")
        print(f"Average Exploitability: {exploitability_metrics['average_exploitability_mbb']:.1f} mbb/hand")
        print(f"Samples used: {exploitability_metrics['samples_used']:,}")
        
        # Interpret results
        avg_exploit = exploitability_metrics['average_exploitability_mbb']
        if avg_exploit < 10:
            status = "🟢 Excellent (< 10 mbb/hand)"
        elif avg_exploit < 50:
            status = "🟡 Good (< 50 mbb/hand)"
        elif avg_exploit < 100:
            status = "🟠 Fair (< 100 mbb/hand)"
        else:
            status = "🔴 Poor (> 100 mbb/hand)"
            
        print(f"Overall Status: {status}")
        
        return exploitability_metrics
    
    def generate_convergence_report(self):
        """Generate comprehensive convergence report."""
        print(f"\n📈 Strategy Convergence Report")
        print("=" * 60)
        
        # Analyze convergence for each position
        for position in [0, 1]:
            pos_name = self.position_names[position]
            
            differences = []
            for cards, hand_type in self.test_hands:
                try:
                    strategy_p0 = self.get_strategy_for_position(0, cards, position)
                    strategy_p1 = self.get_strategy_for_position(1, cards, position)
                    difference = np.sum(np.abs(strategy_p0 - strategy_p1))
                    differences.append(difference)
                except:
                    continue
            
            if differences:
                avg_diff = np.mean(differences)
                max_diff = np.max(differences)
                convergent_count = sum(1 for d in differences if d < 0.2)
                convergence_rate = convergent_count / len(differences)
                
                print(f"\n{pos_name} Position Convergence:")
                print(f"  Average difference: {avg_diff:.3f}")
                print(f"  Maximum difference: {max_diff:.3f}")
                print(f"  Convergence rate: {convergence_rate:.1%} (diff < 0.2)")
                
                if convergence_rate > 0.8:
                    print(f"  Status: ✅ Well converged")
                elif convergence_rate > 0.6:
                    print(f"  Status: ⚠️  Partially converged")
                else:
                    print(f"  Status: ❌ Poor convergence")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze Symmetric MCCFR Strategies')
    parser.add_argument('model_path', help='Path to trained model')
    parser.add_argument('--export', type=str, help='Export analysis to JSON file')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SymmetricStrategyAnalyzer(args.model_path)
    
    # Show model info
    stats = analyzer.trainer.strategy_profile.get_total_stats()
    print(f"\n📊 Model Statistics:")
    print(f"  Total info sets: {stats['total_info_sets']:,}")
    print(f"  Total regret: {stats['total_regret']:,.1f}")
    print(f"  Memory usage: {stats['total_memory_mb']:.2f} MB")
    
    # Main analyses
    analyzer.analyze_positional_strategies()
    
    print(f"\n" + "="*80)
    analyzer.compare_positions_within_player(0)
    analyzer.compare_positions_within_player(1)
    
    print(f"\n" + "="*80)
    exploitability = analyzer.calculate_exploitability()
    
    print(f"\n" + "="*80)
    analyzer.generate_convergence_report()
    
    # Export if requested
    if args.export:
        export_data = {
            'model_path': args.model_path,
            'model_stats': stats,
            'exploitability': exploitability,
            'timestamp': str(np.datetime64('now'))
        }
        
        with open(args.export, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"\n💾 Analysis exported to: {args.export}")
    
    print(f"\n✅ Symmetric strategy analysis complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("🔍 Symmetric MCCFR Strategy Analyzer")
        print("=" * 40)
        print("Usage:")
        print("  python scripts/analyze_symmetric_strategies.py <model_path>")
        print("  python scripts/analyze_symmetric_strategies.py checkpoints/model --export analysis.json")
        sys.exit(1)
    
    main()