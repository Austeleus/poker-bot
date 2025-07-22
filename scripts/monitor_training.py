#!/usr/bin/env python3
"""
Monitor MCCFR training progress and analyze learned strategies.

Usage:
    python scripts/monitor_training.py trained_models/my_model
"""

import sys
import os
import numpy as np

# Add project root to path  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.holdem_mccfr_trainer import HoldemMCCFRTrainer
from core.card_utils import get_preflop_hand_type


def analyze_model(model_path):
    """Analyze a trained MCCFR model."""
    
    print(f"🔍 Analyzing Model: {model_path}")
    print("=" * 60)
    
    # Load the trained model
    trainer = HoldemMCCFRTrainer(preflop_only=True)
    
    try:
        trainer.load_strategy(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Get model statistics
    stats = trainer.strategy_profile.get_total_stats()
    print(f"\n📊 Model Statistics:")
    print(f"  Players: {stats['num_players']}")
    print(f"  Total info sets: {stats['total_info_sets']:,}")
    print(f"  Total regret: {stats['total_regret']:,.1f}")
    print(f"  Memory usage: {stats['total_memory_mb']:.2f} MB")
    
    # Analyze key hands
    key_hands = [
        (48, 49),  # AA
        (44, 47),  # KK
        (48, 44),  # AKs
        (48, 45),  # AKo
        (32, 33),  # 99
        (0, 1),    # 22
        (32, 28),  # T9s
        (0, 17),   # 62o
    ]
    
    print(f"\n🃏 Learned Strategies (Player 0):")
    print("Hand | Action      | Prob  | Full Strategy")
    print("-" * 55)
    
    for cards in key_hands:
        strategy = trainer.get_strategy(0, cards)
        hand_type = get_preflop_hand_type(cards)
        
        # Find best action
        best_action_idx = np.argmax(strategy)
        best_prob = strategy[best_action_idx]
        
        action_names = ["Fold", "Call", "Raise1/4", "Raise1/2", "RaisePot", "All-in"]
        best_action = action_names[best_action_idx]
        
        # Format strategy for display
        strategy_str = " ".join([f"{s:.2f}" for s in strategy])
        
        print(f"{hand_type:4} | {best_action:10} | {best_prob:.1%} | {strategy_str}")
    
    # Calculate strategy characteristics
    print(f"\n📈 Strategy Analysis:")
    
    aggressive_hands = 0
    total_hands = len(key_hands)
    
    for cards in key_hands:
        strategy = trainer.get_strategy(0, cards)
        aggression = np.sum(strategy[2:])  # Sum of raising actions
        
        if aggression > 0.5:  # More than 50% raising
            aggressive_hands += 1
    
    print(f"  Aggressive hands: {aggressive_hands}/{total_hands} ({aggressive_hands/total_hands:.1%})")
    
    # Strategy entropy (convergence measure)
    entropies = []
    for cards in key_hands:
        strategy = trainer.get_strategy(0, cards)
        entropy = -np.sum(strategy * np.log(strategy + 1e-10))
        entropies.append(entropy)
    
    avg_entropy = np.mean(entropies)
    print(f"  Average entropy: {avg_entropy:.3f}")
    print(f"  Convergence: {'High' if avg_entropy < 1.0 else 'Moderate' if avg_entropy < 1.5 else 'Low'}")


def monitor_checkpoints():
    """Monitor training progress from checkpoints."""
    
    checkpoint_dir = "checkpoints"
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Find all checkpoints
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("mccfr_checkpoint_") and filename.endswith(".pkl"):
            try:
                iteration = int(filename.split("_")[2].replace(".pkl", ""))
                checkpoints.append((iteration, filename))
            except:
                continue
    
    if not checkpoints:
        print("❌ No checkpoints found")
        return
    
    checkpoints.sort()  # Sort by iteration
    
    print(f"📈 Training Progress ({len(checkpoints)} checkpoints found):")
    print("Iteration | Info Sets | Total Regret | Convergence")
    print("-" * 50)
    
    for iteration, filename in checkpoints:
        filepath = os.path.join(checkpoint_dir, filename)
        
        # Load checkpoint  
        trainer = HoldemMCCFRTrainer(preflop_only=True)
        try:
            trainer.load_strategy(filepath.replace("_player_0.pkl", "").replace("_player_1.pkl", ""))
            stats = trainer.strategy_profile.get_total_stats()
            
            # Calculate convergence metric
            test_cards = (48, 49)  # AA
            strategy = trainer.get_strategy(0, test_cards)
            entropy = -np.sum(strategy * np.log(strategy + 1e-10))
            
            convergence = "High" if entropy < 1.0 else "Med" if entropy < 1.5 else "Low"
            
            print(f"{iteration:8} | {stats['total_info_sets']:8} | {stats['total_regret']:11.1f} | {convergence}")
            
        except Exception as e:
            print(f"{iteration:8} | Error loading checkpoint: {e}")


def main():
    """Main monitoring function."""
    
    if len(sys.argv) < 2:
        print("🔍 MCCFR Training Monitor")
        print("=" * 30)
        print("Usage:")
        print("  python scripts/monitor_training.py <model_path>    # Analyze specific model")
        print("  python scripts/monitor_training.py --checkpoints   # Monitor checkpoint progress")
        print()
        print("Examples:")
        print("  python scripts/monitor_training.py trained_models/my_model")
        print("  python scripts/monitor_training.py --checkpoints")
        return
    
    if sys.argv[1] == "--checkpoints":
        monitor_checkpoints()
    else:
        model_path = sys.argv[1]
        analyze_model(model_path)


if __name__ == "__main__":
    main()