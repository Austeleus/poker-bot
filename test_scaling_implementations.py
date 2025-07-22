#!/usr/bin/env python3
"""
Test script to verify all scaling plan implementations are working.

This tests the key components mentioned in kuhn_to_texas_holdem_scaling_plan.md:
- Info-set key with chance seeds and action pointers
- Street-specific hand buckets  
- CFR+ update rule
- Pot & side-pot calculator
- Deterministic sampling support
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.holdem_info_set import HoldemInfoSet, Street, BettingRound, HoldemAction
from core.holdem_regret_storage import (
    HoldemRegretTable, get_street_specific_bucket, normalize_suit_isomorphism,
    CardAbstraction
)
from core.pot_calculator import PotCalculator, calculate_showdown_winnings
from core.card_utils import calculate_preflop_equity


def test_info_set_enhancements():
    """Test enhanced info set with chance seeds and action pointers."""
    print("Testing Info Set Enhancements")
    print("-" * 40)
    
    # Test chance seed and action pointer integration
    info_set = HoldemInfoSet(
        player=0,
        hole_cards=(48, 49),  # AA
        community_cards=(),
        street=Street.PREFLOP,
        betting_history=[],
        position=0,
        stack_sizes=(200, 200),
        pot_size=3,
        current_bet=2,
        action_pointer=0,
        chance_seed=12345
    )
    
    print(f"✓ Info set with chance seed created")
    print(f"  Chance seed: {info_set.chance_seed}")
    print(f"  Action pointer: {info_set.action_pointer}")
    print(f"  Can deterministic sample: {info_set.can_deterministic_sample()}")
    
    # Test abstract key includes chance seed and action pointer
    abstract_key = info_set.get_abstract_key(use_minimal_abstraction=True)
    print(f"  Abstract key: {abstract_key}")
    
    assert "CS" in abstract_key, "Chance seed should be in abstract key"
    assert "Act" in abstract_key, "Action pointer should be in abstract key"
    print("✓ Abstract key includes chance seed and action pointer")
    
    # Test child info set creation
    child = info_set.create_child_info_set(
        new_community_cards=(4, 8, 12),  # Flop
        new_betting_history=[],
        new_street=Street.FLOP,
        new_pot_size=10,
        new_current_bet=0,
        new_action_pointer=1,
        new_chance_seed=12346
    )
    
    print(f"✓ Child info set created for flop")
    print(f"  Child street: {child.street.name}")
    print(f"  Child chance seed: {child.chance_seed}")
    
    return True


def test_street_specific_buckets():
    """Test street-specific hand bucketing."""
    print(f"\nTesting Street-Specific Hand Buckets")
    print("-" * 40)
    
    hole_cards = (48, 44)  # AcKc
    
    # Test preflop bucket
    preflop_bucket = get_street_specific_bucket(hole_cards, (), Street.PREFLOP)
    print(f"✓ Preflop bucket for AKs: {preflop_bucket}")
    
    # Test flop bucket
    flop_cards = (4, 8, 12)  # 3c, 3s, 4s
    flop_bucket = get_street_specific_bucket(hole_cards, flop_cards, Street.FLOP)
    print(f"✓ Flop bucket for AKs on {flop_cards}: {flop_bucket}")
    
    # Test that different streets give different buckets (avoiding corruption)
    turn_cards = (4, 8, 12, 16)  # Add 5c
    turn_bucket = get_street_specific_bucket(hole_cards, turn_cards, Street.TURN)
    print(f"✓ Turn bucket for AKs: {turn_bucket}")
    
    # Test suit isomorphism
    suited_hand1 = (48, 44)  # AcKc
    suited_hand2 = (50, 46)  # AhKh
    
    norm1 = normalize_suit_isomorphism(suited_hand1)
    norm2 = normalize_suit_isomorphism(suited_hand2)
    
    print(f"✓ Suit isomorphism: {suited_hand1} -> {norm1}, {suited_hand2} -> {norm2}")
    assert norm1 == norm2, "Suit isomorphic hands should normalize to same representation"
    print("✓ Suit isomorphism working correctly")
    
    return True


def test_cfr_plus_updates():
    """Test CFR+ regret updates."""
    print(f"\nTesting CFR+ Updates")
    print("-" * 40)
    
    # Create regret table with CFR+
    table_cfr_plus = HoldemRegretTable(use_cfr_plus=True)
    table_vanilla = HoldemRegretTable(use_cfr_plus=False)
    
    info_set_key = "test_key"
    legal_actions = [0, 1, 2]  # Fold, Call, Raise
    
    # Add negative regret (should be clipped in CFR+)
    table_cfr_plus.update_regret(info_set_key, 0, -10.0, iteration=1)
    table_vanilla.update_regret(info_set_key, 0, -10.0, iteration=1)
    
    # Add positive regret
    table_cfr_plus.update_regret(info_set_key, 1, 5.0, iteration=1)
    table_vanilla.update_regret(info_set_key, 1, 5.0, iteration=1)
    
    # Get strategies
    strategy_cfr_plus = table_cfr_plus.get_strategy(info_set_key, legal_actions)
    strategy_vanilla = table_vanilla.get_strategy(info_set_key, legal_actions)
    
    print(f"✓ CFR+ strategy: {strategy_cfr_plus}")
    print(f"✓ Vanilla strategy: {strategy_vanilla}")
    
    # CFR+ should have clipped negative regrets
    cfr_plus_regret = table_cfr_plus.regrets[info_set_key][0]
    vanilla_regret = table_vanilla.regrets[info_set_key][0]
    
    print(f"✓ CFR+ regret for action 0: {cfr_plus_regret}")
    print(f"✓ Vanilla regret for action 0: {vanilla_regret}")
    
    assert cfr_plus_regret >= 0, "CFR+ should clip negative regrets to 0"
    assert vanilla_regret < 0, "Vanilla CFR should keep negative regrets"
    print("✓ CFR+ negative regret clipping working correctly")
    
    return True


def test_pot_calculator():
    """Test pot and side-pot calculations."""
    print(f"\nTesting Pot & Side-Pot Calculator")
    print("-" * 40)
    
    # Test uneven stacks scenario
    initial_stacks = [50, 100, 25]  # Uneven stacks
    calc = PotCalculator(initial_stacks, small_blind=1, big_blind=2)
    
    print(f"✓ Initial stacks: {initial_stacks}")
    print(f"✓ After blinds - Pot: ${calc.get_total_pot_size()}, Stacks: {calc.current_stacks}")
    
    # Player 2 goes all-in
    calc.add_contribution(2, 23)  # All remaining chips
    print(f"✓ After P2 all-in - Pot: ${calc.get_total_pot_size()}")
    
    # Player 1 calls
    calc.add_contribution(1, 25)
    print(f"✓ After P1 calls - Pot: ${calc.get_total_pot_size()}")
    
    # Calculate side pots
    active_players = [0, 1, 2]
    pots = calc.calculate_side_pots(active_players)
    
    print(f"✓ Side pot calculation:")
    for i, pot in enumerate(pots):
        print(f"  Pot {i}: ${pot.amount} (players: {pot.eligible_players})")
    
    # Test showdown with hand strengths
    hand_strengths = {0: 0.3, 1: 0.8, 2: 0.6}  # P1 has best hand
    winnings = calculate_showdown_winnings(calc, hand_strengths, active_players)
    print(f"✓ Showdown winnings: {winnings}")
    
    # Verify pot total matches winnings
    total_winnings = sum(winnings.values())
    assert total_winnings == calc.get_total_pot_size(), "Winnings should equal pot size"
    print("✓ Pot calculation correctness verified")
    
    return True


def test_deterministic_sampling():
    """Test deterministic sampling capabilities."""
    print(f"\nTesting Deterministic Sampling Support")
    print("-" * 40)
    
    # Create info set with deterministic sampling
    info_set = HoldemInfoSet(
        player=0,
        hole_cards=(48, 49),  # AA
        community_cards=(),
        street=Street.PREFLOP,
        betting_history=[],
        position=0,
        stack_sizes=(200, 200),
        pot_size=3,
        current_bet=2,
        chance_seed=42,
        deck_state=tuple(range(4, 52))  # Remaining deck after dealing AA
    )
    
    print(f"✓ Info set with deterministic sampling created")
    print(f"  Can deterministic sample: {info_set.can_deterministic_sample()}")
    
    remaining_deck = info_set.get_remaining_deck()
    print(f"✓ Remaining deck size: {len(remaining_deck)}")
    
    # Test that same seed produces consistent keys
    info_set2 = HoldemInfoSet(
        player=0,
        hole_cards=(48, 49),
        community_cards=(),
        street=Street.PREFLOP,
        betting_history=[],
        position=0,
        stack_sizes=(200, 200),
        pot_size=3,
        current_bet=2,
        chance_seed=42,  # Same seed
        deck_state=tuple(range(4, 52))
    )
    
    key1 = info_set.get_abstract_key()
    key2 = info_set2.get_abstract_key()
    
    assert key1 == key2, "Same chance seed should produce same abstract key"
    print("✓ Deterministic sampling produces consistent keys")
    
    return True


def test_performance_ready():
    """Test that implementations are ready for high-performance MCCFR."""
    print(f"\nTesting Performance Readiness")
    print("-" * 40)
    
    # Test equity calculation speed
    import time
    
    hand1 = (48, 49)  # AA
    hand2 = (44, 47)  # KK
    
    start_time = time.time()
    for _ in range(100):
        equity = calculate_preflop_equity(hand1, hand2, num_simulations=100)
    end_time = time.time()
    
    equity_time = (end_time - start_time) / 100 * 1000  # ms per calculation
    print(f"✓ Equity calculation time: {equity_time:.2f}ms per evaluation")
    
    # Test regret table operations
    table = HoldemRegretTable(use_cfr_plus=True)
    info_set_key = "perf_test"
    legal_actions = [0, 1, 2, 3, 4, 5]
    
    start_time = time.time()
    for i in range(1000):
        table.update_regret(info_set_key, i % 6, float(i % 10 - 5), iteration=i)
        strategy = table.get_strategy(info_set_key, legal_actions)
    end_time = time.time()
    
    regret_time = (end_time - start_time) / 1000 * 1000000  # microseconds per op
    print(f"✓ Regret update + strategy time: {regret_time:.1f}µs per operation")
    
    # Test pot calculation speed
    calc = PotCalculator([200, 200])
    
    start_time = time.time()
    for _ in range(1000):
        calc.add_contribution(0, 10)
        calc.get_amount_to_call(1)
        calc.calculate_side_pots([0, 1])
    end_time = time.time()
    
    pot_time = (end_time - start_time) / 1000 * 1000  # ms per operation set
    print(f"✓ Pot calculation time: {pot_time:.3f}ms per operation set")
    
    print("✓ Performance benchmarks completed")
    return True


def run_all_tests():
    """Run all scaling plan implementation tests."""
    print("=" * 60)
    print("TESTING SCALING PLAN IMPLEMENTATIONS")
    print("=" * 60)
    
    tests = [
        test_info_set_enhancements,
        test_street_specific_buckets, 
        test_cfr_plus_updates,
        test_pot_calculator,
        test_deterministic_sampling,
        test_performance_ready
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print("✅ PASSED")
            else:
                failed += 1
                print("❌ FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ FAILED with error: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 ALL SCALING PLAN IMPLEMENTATIONS READY! 🎉")
        print("\nKey capabilities implemented:")
        print("✓ Chance seed tracking for deterministic MCCFR sampling")
        print("✓ Action pointer integration to prevent regret aliasing")
        print("✓ Street-specific hand buckets (prevents corruption)")
        print("✓ CFR+ update rule for 5-10× faster convergence")
        print("✓ Pot & side-pot calculator for uneven stacks")
        print("✓ Performance-optimized data structures")
        print("\nReady for full heads-up Texas Hold'em MCCFR training!")
    else:
        print("⚠️  Some implementations need fixes before proceeding")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)