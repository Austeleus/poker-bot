#!/usr/bin/env python3
"""
Simple MCCFR test focused on core functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.mccfr_trainer import MCCFRTrainer
from agents.baseline_agent import MCCFRAgent, RandomAgent, play_tournament, print_tournament_results


def test_mccfr_training(iterations=50000):
    """Test MCCFR training with specified iterations."""
    print("=" * 50)
    print("TESTING MCCFR TRAINING")
    print("=" * 50)
    
    trainer = MCCFRTrainer(seed=42)
    
    # Train for specified iterations
    print(f"Training MCCFR for {iterations:,} iterations...")
    strategy_profile = trainer.train(iterations=iterations, log_interval=max(1000, iterations//10), verbose=True)
    
    # Print final strategies
    print(f"\nFinal strategies after {iterations:,} iterations:")
    print_detailed_strategies(strategy_profile, trainer.info_set_manager)
    
    # Check if strategies look reasonable
    print("\nStrategy Analysis:")
    
    # Jack with no history should check more often than bet
    jack_initial = strategy_profile.get_strategy("1/")
    if jack_initial:
        check_prob = jack_initial.get(0, 0.0)
        bet_prob = jack_initial.get(1, 0.0)
        print(f"Jack initial: Check={check_prob:.3f}, Bet={bet_prob:.3f}")
        if check_prob > bet_prob:
            print("✓ Jack plays conservatively")
        else:
            print("⚠ Jack might be too aggressive")
    
    # Jack facing bet should almost always fold
    jack_facing_bet = strategy_profile.get_strategy("1/B")
    if jack_facing_bet:
        fold_prob = jack_facing_bet.get(1, 0.0)
        print(f"Jack vs bet: Fold probability = {fold_prob:.3f}")
        if fold_prob > 0.9:
            print("✓ Jack folds to bets as expected")
        else:
            print("⚠ Jack not folding enough")
    
    # King should be aggressive
    king_initial = strategy_profile.get_strategy("3/")
    if king_initial:
        bet_prob = king_initial.get(1, 0.0)
        print(f"King initial: Bet probability = {bet_prob:.3f}")
        if bet_prob > 0.5:
            print("✓ King plays aggressively")
        else:
            print("⚠ King not aggressive enough")
    
    # King facing bet should usually call
    king_facing_bet = strategy_profile.get_strategy("3/B")
    if king_facing_bet:
        call_prob = king_facing_bet.get(0, 0.0)
        print(f"King vs bet: Call probability = {call_prob:.3f}")
        if call_prob > 0.8:
            print("✓ King calls bets as expected")
        else:
            print("⚠ King not calling enough")
    
    return trainer, strategy_profile


def test_mccfr_vs_baselines(strategy_profile):
    """Test MCCFR strategy against baseline agents."""
    print("\n" + "=" * 50)
    print("TESTING MCCFR VS BASELINES")
    print("=" * 50)
    
    # Create agents
    mccfr_agent = MCCFRAgent(0, strategy_profile, "MCCFR")
    random_agent = RandomAgent(0, "Random")
    
    # Simple head-to-head test
    print("Playing 100 games: MCCFR vs Random")
    
    mccfr_total = 0
    random_total = 0
    
    for game in range(100):
        # Alternate positions
        if game % 2 == 0:
            p0 = MCCFRAgent(0, strategy_profile, "MCCFR")
            p1 = RandomAgent(1, "Random")
        else:
            p0 = RandomAgent(0, "Random")
            p1 = MCCFRAgent(1, strategy_profile, "MCCFR")
        
        # Play game
        from agents.baseline_agent import play_game
        payoff, _ = play_game(p0, p1, seed=game)
        
        if game % 2 == 0:
            mccfr_total += payoff
            random_total -= payoff
        else:
            mccfr_total -= payoff
            random_total += payoff
    
    mccfr_avg = mccfr_total / 100
    random_avg = random_total / 100
    
    print(f"Results after 100 games:")
    print(f"MCCFR average payoff: {mccfr_avg:.4f}")
    print(f"Random average payoff: {random_avg:.4f}")
    
    if mccfr_avg > 0.1:
        print("✓ MCCFR significantly outperforms random")
    elif mccfr_avg > 0.0:
        print("✓ MCCFR outperforms random")
    else:
        print("⚠ MCCFR not clearly superior to random")
    
    return mccfr_avg


def print_detailed_strategies(strategy_profile, info_set_manager):
    """Print detailed strategies for both players."""
    
    print("\n" + "=" * 70)
    print("PLAYER 0 STRATEGIES")
    print("=" * 70)
    
    player_0_info_sets = info_set_manager.get_player_info_sets(0)
    for info_set in sorted(player_0_info_sets, key=lambda x: x.to_string()):
        info_set_key = info_set.to_string()
        strategy = strategy_profile.get_strategy(info_set_key)
        
        # Parse info set for readable description
        card_str, history = info_set_key.split('/')
        card_name = {"1": "Jack", "2": "Queen", "3": "King"}[card_str]
        
        if history == "":
            situation = "Initial decision"
        elif history == "C":
            situation = "After opponent checks"
        elif history == "B":
            situation = "Facing opponent's bet"
        elif history == "CB":
            situation = "After check-bet sequence (opponent bet after we checked)"
        else:
            situation = f"History: {history}"
        
        print(f"\n{info_set_key} - {card_name}, {situation}:")
        for action, prob in strategy.items():
            action_name = info_set.get_action_meaning(action)
            print(f"  {action_name}: {prob:.4f} ({prob*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("PLAYER 1 STRATEGIES")
    print("=" * 70)
    
    player_1_info_sets = info_set_manager.get_player_info_sets(1)
    for info_set in sorted(player_1_info_sets, key=lambda x: x.to_string()):
        info_set_key = info_set.to_string()
        strategy = strategy_profile.get_strategy(info_set_key)
        
        # Parse info set for readable description
        card_str, history = info_set_key.split('/')
        card_name = {"1": "Jack", "2": "Queen", "3": "King"}[card_str]
        
        if history == "C":
            situation = "After opponent checks"
        elif history == "B":
            situation = "After opponent bets"
        else:
            situation = f"History: {history}"
        
        print(f"\n{info_set_key} - {card_name}, {situation}:")
        for action, prob in strategy.items():
            action_name = info_set.get_action_meaning(action)
            print(f"  {action_name}: {prob:.4f} ({prob*100:.1f}%)")


def main(iterations=50000):
    """Run simple MCCFR tests."""
    print("SIMPLE MCCFR TEST")
    print("=" * 60)
    print(f"Running with {iterations:,} iterations")
    
    # Test training
    trainer, strategy_profile = test_mccfr_training(iterations)
    
    # Test against baselines
    mccfr_performance = test_mccfr_vs_baselines(strategy_profile)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    stats = trainer.get_statistics()
    print(f"Total iterations: {stats['mccfr_iterations']}")
    print(f"Nodes touched: {stats['total_nodes_touched']}")
    print(f"Average performance vs random: {mccfr_performance:.4f}")
    
    if mccfr_performance > 0.1:
        print("\n🎉 SUCCESS: MCCFR implementation working correctly!")
    elif mccfr_performance > 0.0:
        print("\n✅ GOOD: MCCFR beats random players")
    else:
        print("\n⚠️ NEEDS WORK: MCCFR not clearly better than random")
    
    return trainer, strategy_profile


if __name__ == "__main__":
    import sys
    
    # Default iterations
    iterations = 50000
    
    # Check for command line argument
    if len(sys.argv) > 1:
        try:
            iterations = int(sys.argv[1])
            print(f"Using {iterations:,} iterations from command line")
        except ValueError:
            print(f"Invalid iterations argument '{sys.argv[1]}', using default {iterations:,}")
    
    main(iterations)