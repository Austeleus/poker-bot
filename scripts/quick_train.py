#!/usr/bin/env python3
"""
Quick MCCFR training script for immediate results.

Usage:
    python scripts/quick_train.py          # Train for 10k iterations  
    python scripts/quick_train.py 50000    # Train for 50k iterations
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.holdem_mccfr_trainer import train_preflop_bot


def main():
    """Quick training with sensible defaults."""
    
    # Parse command line argument
    if len(sys.argv) > 1:
        try:
            iterations = int(sys.argv[1])
        except ValueError:
            print("Usage: python quick_train.py [iterations]")
            sys.exit(1)
    else:
        iterations = 10000  # Default
    
    print(f"🚀 Quick MCCFR Training: {iterations:,} iterations")
    print("=" * 50)
    
    # Ensure directories exist
    os.makedirs("trained_models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    # Train with automatic checkpointing every 1000 iterations
    model_path = f"trained_models/quick_model_{iterations}"
    
    trainer = train_preflop_bot(
        iterations=iterations,
        save_path=model_path,
        verbose=True
    )
    
    print(f"\n✅ Quick training completed!")
    print(f"📁 Model saved to: {model_path}")
    
    return trainer


if __name__ == "__main__":
    main()