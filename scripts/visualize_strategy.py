#!/usr/bin/env python3
"""
Comprehensive strategy visualization for trained MCCFR models.

This script loads a trained poker bot strategy and provides multiple visualization options:
- Hand charts showing optimal actions
- Strategy tables with probabilities
- Range analysis for different situations
- Comparison between players
- Export to various formats

Usage:
    python scripts/visualize_strategy.py trained_models/my_model
    python scripts/visualize_strategy.py --help
"""

import sys
import os
import argparse
import numpy as np
import json
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.holdem_mccfr_trainer import HoldemMCCFRTrainer
from core.card_utils import get_preflop_hand_type, get_preflop_hand_strength


class StrategyVisualizer:
    """Comprehensive strategy visualization and analysis."""
    
    def __init__(self, model_path: str):
        """Initialize with a trained model."""
        self.model_path = model_path
        
        # Initialize trainer with same parameters as training
        self.trainer = HoldemMCCFRTrainer(
            small_blind=1,
            big_blind=2, 
            initial_stack=200,
            preflop_only=True
        )
        
        try:
            # Handle different path formats
            if model_path.endswith('_player_0.pkl') or model_path.endswith('_player_1.pkl'):
                # Remove player suffix for loading
                base_path = model_path.replace('_player_0.pkl', '').replace('_player_1.pkl', '')
                self.trainer.load_strategy(base_path)
            else:
                self.trainer.load_strategy(model_path)
            print(f"✅ Loaded strategy from: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load strategy: {e}")
            print(f"Make sure the file exists and try the base path without _player_X.pkl suffix")
            sys.exit(1)
        
        # Action mapping for display
        self.action_names = {
            0: "Fold",
            1: "Call", 
            2: "Raise1/4",
            3: "Raise1/2",
            4: "RaisePot", 
            5: "All-in"
        }
        
        self.action_colors = {
            0: "🔴",  # Fold - Red
            1: "🟡",  # Call - Yellow
            2: "🟠",  # Raise 1/4 - Orange
            3: "🟠",  # Raise 1/2 - Orange
            4: "🔴",  # Raise Pot - Red (aggressive)
            5: "🔴"   # All-in - Red (very aggressive)
        }
        
        # Generate all possible preflop hands for analysis
        self.all_hands = self._generate_hand_matrix()
    
    def _generate_hand_matrix(self) -> List[Tuple[Tuple[int, int], str]]:
        """Generate all 169 unique preflop hands."""
        hands = []
        
        # All possible rank combinations
        ranks = list(range(13))  # 0=2, 1=3, ..., 12=A
        
        for rank1 in ranks:
            for rank2 in ranks:
                if rank1 >= rank2:  # Avoid duplicates
                    # Create card representations
                    if rank1 == rank2:
                        # Pair - use same suit for both
                        cards = (rank1 * 4, rank2 * 4)
                        hand_type = get_preflop_hand_type(cards)
                    else:
                        # Non-pair - create suited and offsuit versions
                        # Suited version
                        cards_suited = (rank1 * 4, rank2 * 4)
                        hand_type_suited = get_preflop_hand_type(cards_suited)
                        hands.append((cards_suited, hand_type_suited))
                        
                        # Offsuit version
                        cards_offsuit = (rank1 * 4, rank2 * 4 + 1)
                        hand_type_offsuit = get_preflop_hand_type(cards_offsuit)
                        cards = cards_offsuit
                        hand_type = hand_type_offsuit
                    
                    if rank1 == rank2 or not any(h[1] == hand_type for h in hands):
                        hands.append((cards, hand_type))
        
        return hands
    
    def show_basic_strategy_table(self, player: int = 0):
        """Display basic strategy table for key hands."""
        print(f"\n🃏 Basic Strategy Table (Player {player})")
        print("=" * 80)
        print("Hand     | Best Action  | Prob  | Fold  | Call  | R1/4  | R1/2  | RPot  | Allin")
        print("-" * 80)
        
        # Key representative hands (using proper card encoding)
        key_hands = [
            (48, 49),   # AA (A♠, A♥)
            (44, 45),   # KK (K♠, K♥) 
            (40, 41),   # QQ (Q♠, Q♥)
            (36, 37),   # JJ (J♠, J♥)
            (32, 33),   # TT (T♠, T♥)
            (28, 29),   # 99 (9♠, 9♥)
            (48, 44),   # AKs (A♠, K♠)
            (48, 47),   # AKo (A♠, K♥)
            (48, 40),   # AQs (A♠, Q♠)
            (48, 43),   # AQo (A♠, Q♥)
            (44, 40),   # KQs (K♠, Q♠)
            (44, 43),   # KQo (K♠, Q♥)
            (32, 28),   # T9s (T♠, 9♠)
            (28, 24),   # 97s (9♠, 7♠)
            (16, 12),   # 54s (5♠, 4♠)
            (0, 1),     # 22 (2♠, 2♥)
            (0, 17),    # 62o (2♠, 6♥)
            (4, 21),    # 36o (3♠, 6♥)
        ]
        
        for cards in key_hands:
            try:
                hand_type = get_preflop_hand_type(cards)
                strategy = self.trainer.get_strategy(player, cards)
                
                # Check if strategy is valid
                if len(strategy) != 6:
                    print(f"{hand_type:8} | Error: Invalid strategy length {len(strategy)}")
                    continue
                
                # Find best action
                best_action_idx = np.argmax(strategy)
                best_action = self.action_names[best_action_idx]
                best_prob = strategy[best_action_idx]
                
                # Format probabilities
                prob_strs = [f"{p:.2f}" for p in strategy]
                
                print(f"{hand_type:8} | {best_action:11} | {best_prob:.2f} | " + 
                      " | ".join(f"{p:4}" for p in prob_strs))
                
            except Exception as e:
                hand_type = get_preflop_hand_type(cards) if 'cards' in locals() else "Unknown"
                print(f"{hand_type:8} | Error: {str(e)[:50]}")
    
    def show_hand_chart(self, player: int = 0, action_threshold: float = 0.4):
        """Display poker hand chart with dominant actions."""
        print(f"\n🎯 Hand Chart - Dominant Actions (Player {player})")
        print(f"Threshold: {action_threshold:.0%} (actions with ≥{action_threshold:.0%} probability)")
        print("=" * 60)
        
        # Create 13x13 grid for hand chart
        chart = {}
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        
        for i, rank1 in enumerate(ranks):
            for j, rank2 in enumerate(ranks):
                # Determine hand type
                if i == j:  # Pair
                    hand_str = f"{rank1}{rank1}"
                elif i < j:  # Suited (upper triangle)
                    hand_str = f"{rank1}{rank2}s"
                else:  # Offsuit (lower triangle)
                    hand_str = f"{rank2}{rank1}o"
                
                # Convert to card integers and get strategy
                try:
                    # Create representative cards for this hand type
                    rank1_int = 12 - i  # Convert back to 0-12 scale
                    rank2_int = 12 - j
                    
                    if i == j:  # Pair
                        cards = (rank1_int * 4, rank2_int * 4)
                    elif i < j:  # Suited
                        cards = (rank1_int * 4, rank2_int * 4)
                    else:  # Offsuit
                        cards = (rank1_int * 4, rank2_int * 4 + 1)
                    
                    strategy = self.trainer.get_strategy(player, cards)
                    
                    # Find dominant action
                    max_prob = np.max(strategy)
                    if max_prob >= action_threshold:
                        dominant_action = np.argmax(strategy)
                        chart[(i, j)] = {
                            'hand': hand_str,
                            'action': dominant_action,
                            'prob': max_prob,
                            'color': self.action_colors[dominant_action]
                        }
                    else:
                        chart[(i, j)] = {
                            'hand': hand_str,
                            'action': -1,  # Mixed strategy
                            'prob': max_prob,
                            'color': "⚪"  # Mixed
                        }
                        
                except Exception:
                    chart[(i, j)] = {
                        'hand': hand_str,
                        'action': -2,  # Error
                        'prob': 0.0,
                        'color': "⚫"
                    }
        
        # Print the chart
        print("   ", end="")
        for rank in ranks:
            print(f" {rank:2}", end="")
        print()
        
        for i, rank1 in enumerate(ranks):
            print(f"{rank1:2} ", end="")
            for j, rank2 in enumerate(ranks):
                cell = chart[(i, j)]
                print(f" {cell['color']:2}", end="")
            print(f"  {rank1}")
        
        # Legend
        print("\n📋 Legend:")
        print("🔴 Fold/Aggressive (Fold, Raise Pot, All-in)")
        print("🟡 Call")
        print("🟠 Moderate Raise (1/4 pot, 1/2 pot)")
        print("⚪ Mixed Strategy (no dominant action)")
        print("⚫ Error/Unknown")
    
    def show_detailed_hand_analysis(self, hands: List[str], player: int = 0):
        """Show detailed analysis for specific hands."""
        print(f"\n🔬 Detailed Hand Analysis (Player {player})")
        print("=" * 80)
        
        for hand_str in hands:
            try:
                # Convert hand string to cards
                cards = self._hand_string_to_cards(hand_str)
                if cards is None:
                    print(f"❌ Invalid hand: {hand_str}")
                    continue
                
                strategy = self.trainer.get_strategy(player, cards)
                hand_strength = get_preflop_hand_strength(cards)
                
                print(f"\n🃏 {hand_str} (Strength: {hand_strength:.3f})")
                print("-" * 40)
                
                for action_idx, prob in enumerate(strategy):
                    if prob > 0.01:  # Show actions with >1% probability
                        action_name = self.action_names[action_idx]
                        bar = "█" * int(prob * 20)  # Simple bar chart
                        print(f"  {action_name:10}: {prob:.3f} |{bar:<20}|")
                
                # Calculate key metrics
                aggression = np.sum(strategy[2:])  # Sum of raising actions
                passivity = strategy[0] + strategy[1]  # Fold + Call
                
                print(f"  Aggression:  {aggression:.3f}")
                print(f"  Passivity:   {passivity:.3f}")
                
            except Exception as e:
                print(f"❌ Error analyzing {hand_str}: {e}")
    
    def compare_players(self, hands: List[str]):
        """Compare strategies between Player 0 and Player 1."""
        print(f"\n⚖️  Player Comparison")
        print("=" * 80)
        print("Hand     | Player 0 Best Action | Player 1 Best Action | Difference")
        print("-" * 80)
        
        for hand_str in hands:
            try:
                cards = self._hand_string_to_cards(hand_str)
                if cards is None:
                    continue
                
                strategy_p0 = self.trainer.get_strategy(0, cards)
                strategy_p1 = self.trainer.get_strategy(1, cards)
                
                # Best actions
                best_p0 = np.argmax(strategy_p0)
                best_p1 = np.argmax(strategy_p1)
                
                action_p0 = self.action_names[best_p0]
                action_p1 = self.action_names[best_p1]
                
                # Strategy difference (L1 distance)
                difference = np.sum(np.abs(strategy_p0 - strategy_p1))
                
                same_action = "✓" if best_p0 == best_p1 else "✗"
                
                print(f"{hand_str:8} | {action_p0:18} | {action_p1:18} | {difference:.3f} {same_action}")
                
            except Exception as e:
                print(f"{hand_str:8} | Error: {str(e)[:60]}")
    
    def export_strategy_json(self, output_file: str, player: int = 0):
        """Export complete strategy to JSON format."""
        print(f"\n💾 Exporting strategy to JSON: {output_file}")
        
        strategy_data = {
            'model_path': self.model_path,
            'player': player,
            'hands': {},
            'metadata': {
                'total_hands': 0,
                'export_time': str(np.datetime64('now')),
                'action_names': self.action_names
            }
        }
        
        # Export strategies for all hands
        for cards, hand_type in self.all_hands:
            try:
                strategy = self.trainer.get_strategy(player, cards)
                hand_strength = get_preflop_hand_strength(cards)
                
                # Find best action
                best_action_idx = np.argmax(strategy)
                
                strategy_data['hands'][hand_type] = {
                    'strategy': strategy.tolist(),
                    'best_action': int(best_action_idx),
                    'best_action_name': self.action_names[best_action_idx],
                    'best_action_prob': float(strategy[best_action_idx]),
                    'hand_strength': float(hand_strength),
                    'cards': list(cards)
                }
                
                strategy_data['metadata']['total_hands'] += 1
                
            except Exception as e:
                print(f"Warning: Failed to export {hand_type}: {e}")
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(strategy_data, f, indent=2)
        
        print(f"✅ Exported {strategy_data['metadata']['total_hands']} hands to {output_file}")
    
    def show_range_analysis(self, player: int = 0):
        """Analyze playing ranges by action type."""
        print(f"\n📊 Range Analysis (Player {player})")
        print("=" * 60)
        
        # Categorize hands by dominant action
        action_ranges = {action: [] for action in self.action_names.keys()}
        hand_strengths = {action: [] for action in self.action_names.keys()}
        
        for cards, hand_type in self.all_hands:
            try:
                strategy = self.trainer.get_strategy(player, cards)
                hand_strength = get_preflop_hand_strength(cards)
                
                # Find dominant action
                dominant_action = np.argmax(strategy)
                dominant_prob = strategy[dominant_action]
                
                # Only include if action is truly dominant (>40%)
                if dominant_prob >= 0.4:
                    action_ranges[dominant_action].append(hand_type)
                    hand_strengths[dominant_action].append(hand_strength)
                
            except Exception:
                continue
        
        # Display ranges
        for action_idx, action_name in self.action_names.items():
            hands = action_ranges[action_idx]
            strengths = hand_strengths[action_idx]
            
            if hands:
                avg_strength = np.mean(strengths)
                min_strength = np.min(strengths)
                max_strength = np.max(strengths)
                
                print(f"\n{action_name} Range ({len(hands)} hands):")
                print(f"  Avg Strength: {avg_strength:.3f}")
                print(f"  Range: {min_strength:.3f} - {max_strength:.3f}")
                print(f"  Hands: {', '.join(sorted(hands)[:15])}")
                if len(hands) > 15:
                    print(f"         ... and {len(hands) - 15} more")
            else:
                print(f"\n{action_name} Range: No dominant hands")
    
    def _hand_string_to_cards(self, hand_str: str) -> Optional[Tuple[int, int]]:
        """Convert hand string (e.g., 'AA', 'AKs', 'AKo') to card integers."""
        if len(hand_str) < 2:
            return None
        
        # Rank mapping
        rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
                   'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        
        rank1_char = hand_str[0]
        rank2_char = hand_str[1]
        
        if rank1_char not in rank_map or rank2_char not in rank_map:
            return None
        
        rank1 = rank_map[rank1_char]
        rank2 = rank_map[rank2_char]
        
        # Handle suited/offsuit
        if len(hand_str) == 2:  # Pair (e.g., 'AA')
            return (rank1 * 4, rank2 * 4 + 1)  # Different suits for pair
        elif len(hand_str) == 3:
            if hand_str[2].lower() == 's':  # Suited
                return (rank1 * 4, rank2 * 4)
            elif hand_str[2].lower() == 'o':  # Offsuit
                return (rank1 * 4, rank2 * 4 + 1)
        
        return None


def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(description='Visualize MCCFR Strategy')
    parser.add_argument('model_path', help='Path to trained model')
    parser.add_argument('--player', type=int, default=0, choices=[0, 1],
                       help='Player to analyze (default: 0)')
    parser.add_argument('--hands', nargs='+', 
                       default=['AA', 'KK', 'AKs', 'AKo', '22', '62o'],
                       help='Specific hands to analyze')
    parser.add_argument('--export', type=str,
                       help='Export strategy to JSON file')
    parser.add_argument('--chart-threshold', type=float, default=0.4,
                       help='Threshold for hand chart dominant actions')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    viz = StrategyVisualizer(args.model_path)
    
    # Show model info
    stats = viz.trainer.strategy_profile.get_total_stats()
    print(f"\n📊 Model Statistics:")
    print(f"  Total info sets: {stats['total_info_sets']:,}")
    print(f"  Total regret: {stats['total_regret']:,.1f}")
    print(f"  Memory usage: {stats['total_memory_mb']:.2f} MB")
    
    # Main visualizations
    viz.show_basic_strategy_table(args.player)
    viz.show_hand_chart(args.player, args.chart_threshold)
    viz.show_detailed_hand_analysis(args.hands, args.player)
    viz.show_range_analysis(args.player)
    
    # Player comparison (if analyzing player 0)
    if args.player == 0:
        viz.compare_players(args.hands)
    
    # Export if requested
    if args.export:
        viz.export_strategy_json(args.export, args.player)
    
    print(f"\n✅ Strategy visualization complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("🎨 MCCFR Strategy Visualizer")
        print("=" * 40)
        print("Usage:")
        print("  python scripts/visualize_strategy.py <model_path>")
        print("  python scripts/visualize_strategy.py trained_models/my_model")
        print("  python scripts/visualize_strategy.py trained_models/my_model --export strategy.json")
        print("  python scripts/visualize_strategy.py trained_models/my_model --hands AA KK AKs --player 1")
        sys.exit(1)
    
    main()