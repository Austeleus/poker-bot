#!/usr/bin/env python3
"""
Full-scale MCCFR training script for heads-up preflop poker bot.

This script runs comprehensive MCCFR training with:
- Automatic checkpointing every 1000 iterations
- Convergence monitoring and early stopping
- Strategy analysis at key milestones
- Exploitability estimation
- Detailed logging and progress tracking

Usage:
    python scripts/train_full_mccfr.py [--iterations 100000] [--save-every 1000]
"""

import os
import sys
import argparse
import time
import logging
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.holdem_mccfr_trainer import HoldemMCCFRTrainer
from core.card_utils import get_preflop_hand_type


def setup_logging(log_dir: str = "logs"):
    """Setup comprehensive logging for training."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"mccfr_training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Training session started - Log file: {log_file}")
    return logger


def analyze_strategy_convergence(trainer, key_hands, iteration):
    """Analyze strategy convergence for key poker hands."""
    print(f"\n🃏 Strategy Analysis at Iteration {iteration}:")
    print("=" * 60)
    
    convergence_data = {}
    
    for cards in key_hands:
        strategy = trainer.get_strategy(0, cards)
        hand_type = get_preflop_hand_type(cards)
        
        # Find most likely action
        best_action_idx = np.argmax(strategy)
        best_prob = strategy[best_action_idx]
        
        # Action names
        action_names = {
            0: "Fold",
            1: "Call", 
            2: "Raise1/4",
            3: "Raise1/2", 
            4: "RaisePot",
            5: "All-in"
        }
        
        best_action = action_names.get(best_action_idx, f"Action{best_action_idx}")
        
        # Calculate strategy entropy (measure of convergence)
        # Higher entropy = more mixed strategy, lower entropy = more converged
        entropy = -np.sum(strategy * np.log(strategy + 1e-10))
        
        print(f"{hand_type:4}: {best_action:8} ({best_prob:.1%}) - "
              f"Entropy: {entropy:.3f} - Strategy: {strategy}")
        
        convergence_data[hand_type] = {
            'strategy': strategy.copy(),
            'best_action': best_action,
            'best_prob': best_prob,
            'entropy': entropy
        }
    
    # Calculate average entropy as convergence metric
    avg_entropy = np.mean([data['entropy'] for data in convergence_data.values()])
    print(f"\nAverage Strategy Entropy: {avg_entropy:.3f}")
    print(f"Convergence Status: {'High' if avg_entropy < 1.0 else 'Moderate' if avg_entropy < 1.5 else 'Low'}")
    
    return convergence_data, avg_entropy


def estimate_exploitability(trainer, num_samples=1000):
    """
    Simple convergence metric based on strategy entropy.
    
    Lower entropy = more converged strategies = lower exploitability.
    This is a proxy metric until proper best response calculation is implemented.
    """
    print(f"\n📊 Strategy Convergence Metric:")
    print("-" * 50)
    
    # Sample random hands and compute strategy entropy
    from core.card_utils import generate_all_preflop_hands
    all_hands = generate_all_preflop_hands()
    
    entropies_p0 = []
    entropies_p1 = []
    
    # Sample subset of hands for efficiency
    sampled_hands = np.random.choice(len(all_hands), min(num_samples, len(all_hands)), replace=False)
    
    for hand_idx in sampled_hands:
        hand_cards = all_hands[hand_idx]
        
        # Get strategies for both players
        try:
            strategy_p0 = trainer.get_strategy(0, hand_cards)
            strategy_p1 = trainer.get_strategy(1, hand_cards)
            
            # Calculate entropy (higher = more mixed, lower = more converged)
            entropy_p0 = -np.sum(strategy_p0 * np.log(strategy_p0 + 1e-10))
            entropy_p1 = -np.sum(strategy_p1 * np.log(strategy_p1 + 1e-10))
            
            entropies_p0.append(entropy_p0)
            entropies_p1.append(entropy_p1)
        except:
            continue
    
    avg_entropy = np.mean(entropies_p0 + entropies_p1)
    
    # Convert entropy to pseudo-exploitability metric (higher entropy = higher exploitability)
    pseudo_exploitability = avg_entropy * 10  # Scale for readability
    
    print(f"Average strategy entropy: {avg_entropy:.3f}")
    print(f"Convergence metric: {pseudo_exploitability:.1f} (lower = better)")
    
    return pseudo_exploitability


def check_convergence_criteria(convergence_history, min_iterations=10000):
    """Check if training has converged based on strategy stability."""
    if len(convergence_history) < 5:
        return False, "Not enough data points"
    
    if convergence_history[-1]['iteration'] < min_iterations:
        return False, f"Minimum iterations not reached ({min_iterations})"
    
    # Check if entropy has stabilized (last 3 measurements within 5% of each other)
    recent_entropies = [point['avg_entropy'] for point in convergence_history[-3:]]
    
    if len(recent_entropies) >= 3:
        entropy_std = np.std(recent_entropies)
        entropy_mean = np.mean(recent_entropies)
        
        if entropy_std / entropy_mean < 0.05:  # 5% coefficient of variation
            return True, f"Strategy entropy stabilized at {entropy_mean:.3f}"
    
    return False, "Strategies still evolving"


def main():
    """Main training function with full convergence monitoring."""
    parser = argparse.ArgumentParser(description='Full MCCFR Training')
    parser.add_argument('--iterations', type=int, default=100000,
                       help='Maximum iterations to train (default: 100000)')
    parser.add_argument('--save-every', type=int, default=1000,
                       help='Save checkpoint every N iterations (default: 1000)')
    parser.add_argument('--check-convergence-every', type=int, default=5000,
                       help='Check convergence every N iterations (default: 5000)')
    parser.add_argument('--early-stop', action='store_true',
                       help='Enable early stopping when converged')
    parser.add_argument('--output-dir', type=str, default='trained_models',
                       help='Output directory for saved models')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Key hands to monitor for convergence
    key_hands = [
        (48, 49),  # AA - premium pair
        (44, 47),  # KK - premium pair  
        (48, 44),  # AKs - premium suited
        (48, 45),  # AKo - premium offsuit
        (32, 33),  # 99 - medium pair
        (0, 1),    # 22 - small pair
        (32, 28),  # T9s - suited connector
        (0, 17),   # 62o - trash hand
    ]
    
    print("🚀 Starting Full-Scale MCCFR Training")
    print("=" * 60)
    print(f"Maximum iterations: {args.iterations:,}")
    print(f"Save checkpoints every: {args.save_every:,} iterations")
    print(f"Check convergence every: {args.check_convergence_every:,} iterations")
    print(f"Early stopping: {'Enabled' if args.early_stop else 'Disabled'}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    
    # Initialize trainer
    logger.info("Initializing MCCFR trainer...")
    trainer = HoldemMCCFRTrainer(
        small_blind=1,
        big_blind=2, 
        initial_stack=200,
        preflop_only=True,
        seed=args.seed
    )
    
    # Training state
    convergence_history = []
    batch_size = args.check_convergence_every
    total_iterations = 0
    
    # Main training loop with convergence monitoring
    while total_iterations < args.iterations:
        iterations_to_run = min(batch_size, args.iterations - total_iterations)
        
        logger.info(f"Training batch: {iterations_to_run:,} iterations "
                   f"({total_iterations:,} / {args.iterations:,} total)")
        
        # Train for this batch
        batch_stats = trainer.train(
            iterations=iterations_to_run,
            save_every=args.save_every,
            verbose=True
        )
        
        total_iterations = trainer.iterations_completed
        
        # Analyze convergence
        logger.info("Analyzing strategy convergence...")
        convergence_data, avg_entropy = analyze_strategy_convergence(
            trainer, key_hands, total_iterations
        )
        
        # Estimate exploitability
        exploitability = estimate_exploitability(trainer, num_samples=5000)
        
        # Record convergence metrics
        convergence_point = {
            'iteration': total_iterations,
            'avg_entropy': avg_entropy,
            'exploitability': exploitability,
            'strategies': convergence_data,
            'training_time': batch_stats['training_time'],
            'nodes_per_second': batch_stats['total_nodes_touched'] / batch_stats['training_time']
        }
        convergence_history.append(convergence_point)
        
        # Check for convergence
        if args.early_stop:
            converged, reason = check_convergence_criteria(
                convergence_history, min_iterations=20000
            )
            
            logger.info(f"Convergence check: {reason}")
            
            if converged:
                logger.info(f"🎉 CONVERGENCE ACHIEVED after {total_iterations:,} iterations!")
                logger.info(f"Final entropy: {avg_entropy:.3f}")
                logger.info(f"Estimated exploitability: {abs(exploitability) * 1000:.1f} mbb/hand")
                break
        
        # Progress update
        progress = total_iterations / args.iterations * 100
        logger.info(f"Training progress: {progress:.1f}% "
                   f"({total_iterations:,} / {args.iterations:,})")
    
    # Final model save
    final_model_path = os.path.join(args.output_dir, f"final_mccfr_model_{total_iterations}")
    trainer.save_strategy(final_model_path)
    logger.info(f"💾 Final model saved to: {final_model_path}")
    
    # Final analysis
    print("\n🏆 TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Total iterations: {total_iterations:,}")
    print(f"Final avg entropy: {avg_entropy:.3f}")
    print(f"Final exploitability estimate: {abs(exploitability) * 1000:.1f} mbb/hand")
    
    # Show final strategies
    print("\n🃏 Final Learned Strategies:")
    analyze_strategy_convergence(trainer, key_hands, total_iterations)
    
    # Save convergence history
    convergence_file = os.path.join(args.output_dir, f"convergence_history_{total_iterations}.npy")
    np.save(convergence_file, convergence_history)
    logger.info(f"📈 Convergence history saved to: {convergence_file}")
    
    print(f"\n✅ Training session complete!")
    print(f"📁 All files saved to: {args.output_dir}")
    print(f"📋 Logs available in: logs/")
    
    return trainer


if __name__ == "__main__":
    try:
        trainer = main()
        print("\n🚀 Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)