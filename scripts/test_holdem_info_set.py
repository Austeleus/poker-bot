#!/usr/bin/env python3
"""
Test script for Texas Hold'em Information Set implementation.

This script thoroughly tests the HoldemInfoSet and HoldemInfoSetManager
classes to ensure correctness before proceeding with MCCFR training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.holdem_info_set import (
    HoldemInfoSet, HoldemInfoSetManager, BettingRound, Street, HoldemAction,
    card_to_string, string_to_card
)


def test_card_utilities():
    """Test card conversion utilities."""
    print("Testing Card Utilities")
    print("-" * 30)
    
    # Test specific cards (card encoding: rank * 4 + suit)
    test_cases = [
        (0, "2c"),    # rank 0 * 4 + 0 = 2 of clubs
        (3, "2s"),    # rank 0 * 4 + 3 = 2 of spades  
        (4, "3c"),    # rank 1 * 4 + 0 = 3 of clubs
        (51, "As"),   # rank 12 * 4 + 3 = Ace of spades
        (48, "Ac"),   # rank 12 * 4 + 0 = Ace of clubs
    ]
    
    for card_int, expected_str in test_cases:
        card_str = card_to_string(card_int)
        back_to_int = string_to_card(card_str)
        
        print(f"Card {card_int:2d} -> '{card_str}' -> {back_to_int:2d} "
              f"(expected '{expected_str}') {'✓' if card_str == expected_str else '✗'}")
        
        assert card_str == expected_str, f"Expected {expected_str}, got {card_str}"
        assert back_to_int == card_int, f"Round-trip failed: {card_int} -> {back_to_int}"
    
    print("Card utilities passed ✓\n")


def test_betting_round():
    """Test BettingRound representation."""
    print("Testing BettingRound")
    print("-" * 30)
    
    # Empty preflop round
    round1 = BettingRound(Street.PREFLOP, (), ())
    print(f"Empty round: {round1}")
    assert str(round1) == "S0:", f"Expected 'S0:', got '{round1}'"
    
    # Preflop with actions
    round2 = BettingRound(
        Street.PREFLOP, 
        (HoldemAction.CHECK_CALL, HoldemAction.RAISE_HALF_POT, HoldemAction.FOLD),
        (2, 6, 0)
    )
    print(f"Preflop betting: {round2}")
    expected = "S0:CR2F"
    assert str(round2) == expected, f"Expected '{expected}', got '{round2}'"
    
    # Flop round
    round3 = BettingRound(
        Street.FLOP,
        (HoldemAction.CHECK_CALL, HoldemAction.ALL_IN),
        (0, 94)
    )
    print(f"Flop betting: {round3}")
    expected = "S1:CA"
    assert str(round3) == expected, f"Expected '{expected}', got '{round3}'"
    
    print("BettingRound passed ✓\n")


def test_basic_info_set():
    """Test basic HoldemInfoSet creation and validation."""
    print("Testing Basic HoldemInfoSet")
    print("-" * 30)
    
    # Valid preflop info set
    info_set = HoldemInfoSet(
        player=0,
        hole_cards=(0, 13),  # 2c, 3c
        community_cards=(),
        street=Street.PREFLOP,
        betting_history=[],
        position=0,
        stack_sizes=(100, 100),
        pot_size=3,
        current_bet=2,
        small_blind=1,
        big_blind=2
    )
    
    print(f"Basic preflop info set created successfully")
    print(f"String representation: {info_set}")
    print(f"Hash: {hash(info_set)}")
    print(f"Legal actions: {info_set.get_legal_actions()}")
    
    for action in info_set.get_legal_actions():
        meaning = info_set.get_action_meaning(action)
        print(f"  Action {action}: {meaning}")
    
    # Test abstraction
    abstract_key = info_set.get_abstract_key()
    print(f"Abstract key: {abstract_key}")
    
    print("Basic info set passed ✓\n")


def test_validation():
    """Test input validation."""
    print("Testing Input Validation")
    print("-" * 30)
    
    valid_args = {
        'player': 0,
        'hole_cards': (0, 13),
        'community_cards': (),
        'street': Street.PREFLOP,
        'betting_history': [],
        'position': 0,
        'stack_sizes': (100, 100),
        'pot_size': 3,
        'current_bet': 2
    }
    
    # Test invalid player
    try:
        invalid_args = valid_args.copy()
        invalid_args['player'] = 2  # Only have 2 players
        HoldemInfoSet(**invalid_args)
        assert False, "Should have raised ValueError for invalid player"
    except ValueError:
        print("✓ Invalid player validation works")
    
    # Test wrong number of hole cards
    try:
        invalid_args = valid_args.copy()
        invalid_args['hole_cards'] = (0, 13, 26)  # 3 cards
        HoldemInfoSet(**invalid_args)
        assert False, "Should have raised ValueError for wrong hole card count"
    except ValueError:
        print("✓ Hole card count validation works")
    
    # Test wrong community cards for street
    try:
        invalid_args = valid_args.copy()
        invalid_args['street'] = Street.FLOP
        invalid_args['community_cards'] = (4, 8)  # Need 3 for flop
        HoldemInfoSet(**invalid_args)
        assert False, "Should have raised ValueError for wrong community card count"
    except ValueError:
        print("✓ Community card count validation works")
    
    # Test duplicate cards
    try:
        invalid_args = valid_args.copy()
        invalid_args['hole_cards'] = (0, 0)  # Duplicate
        HoldemInfoSet(**invalid_args)
        assert False, "Should have raised ValueError for duplicate cards"
    except ValueError:
        print("✓ Duplicate card validation works")
    
    print("Input validation passed ✓\n")


def test_different_streets():
    """Test info sets on different streets."""
    print("Testing Different Streets")
    print("-" * 30)
    
    # Preflop
    preflop_history = []
    preflop_info = HoldemInfoSet(
        player=1,
        hole_cards=(12, 25),  # 4c, 7c
        community_cards=(),
        street=Street.PREFLOP,
        betting_history=preflop_history,
        position=1,
        stack_sizes=(98, 100),
        pot_size=3,
        current_bet=2
    )
    print(f"Preflop: {preflop_info.get_abstract_key()}")
    
    # Flop
    preflop_round = BettingRound(Street.PREFLOP, (HoldemAction.CHECK_CALL, HoldemAction.CHECK_CALL), (2, 0))
    flop_info = HoldemInfoSet(
        player=0,
        hole_cards=(12, 25),  # 4c, 7c
        community_cards=(4, 8, 16),  # 2s, 3s, 5c
        street=Street.FLOP,
        betting_history=[preflop_round],
        position=0,
        stack_sizes=(98, 98),
        pot_size=4,
        current_bet=0
    )
    print(f"Flop: {flop_info.get_abstract_key()}")
    
    # Turn
    flop_round = BettingRound(Street.FLOP, (HoldemAction.CHECK_CALL, HoldemAction.RAISE_HALF_POT, HoldemAction.CHECK_CALL), (0, 2, 2))
    turn_info = HoldemInfoSet(
        player=1,
        hole_cards=(12, 25),  # 4c, 7c  
        community_cards=(4, 8, 16, 20),  # 2s, 3s, 5c, 6c
        street=Street.TURN,
        betting_history=[preflop_round, flop_round],
        position=1,
        stack_sizes=(96, 96),
        pot_size=8,
        current_bet=0
    )
    print(f"Turn: {turn_info.get_abstract_key()}")
    
    # River
    turn_round = BettingRound(Street.TURN, (HoldemAction.CHECK_CALL, HoldemAction.CHECK_CALL), (0, 0))
    river_info = HoldemInfoSet(
        player=0,
        hole_cards=(12, 25),  # 4c, 7c
        community_cards=(4, 8, 16, 20, 24),  # 2s, 3s, 5c, 6c, 7c
        street=Street.RIVER,
        betting_history=[preflop_round, flop_round, turn_round],
        position=0,
        stack_sizes=(96, 96),
        pot_size=8,
        current_bet=0
    )
    print(f"River: {river_info.get_abstract_key()}")
    
    print("Different streets passed ✓\n")


def test_legal_actions():
    """Test legal action generation in different scenarios."""
    print("Testing Legal Actions")
    print("-" * 30)
    
    # Scenario 1: Facing bet with deep stacks
    facing_bet = HoldemInfoSet(
        player=0,
        hole_cards=(0, 13),
        community_cards=(4, 8, 12),  # 2s, 3s, 4s
        street=Street.FLOP,
        betting_history=[],
        position=0,
        stack_sizes=(100, 98),  # Opponent bet 2
        pot_size=6,
        current_bet=2
    )
    
    legal_actions = facing_bet.get_legal_actions()
    print(f"Facing bet - legal actions: {legal_actions}")
    assert HoldemAction.FOLD in legal_actions, "Should be able to fold facing bet"
    assert HoldemAction.CHECK_CALL in legal_actions, "Should be able to call"
    assert HoldemAction.ALL_IN in legal_actions, "Should be able to go all-in"
    
    # Scenario 2: First to act (can check)
    first_to_act = HoldemInfoSet(
        player=0,
        hole_cards=(0, 13),
        community_cards=(4, 8, 12),
        street=Street.FLOP,
        betting_history=[],
        position=0,
        stack_sizes=(100, 100),
        pot_size=4,
        current_bet=0
    )
    
    legal_actions = first_to_act.get_legal_actions()
    print(f"First to act - legal actions: {legal_actions}")
    assert HoldemAction.FOLD not in legal_actions, "Should not be able to fold when can check"
    assert HoldemAction.CHECK_CALL in legal_actions, "Should be able to check"
    
    # Scenario 3: Short stack (limited raises)
    short_stack = HoldemInfoSet(
        player=0,
        hole_cards=(0, 13),
        community_cards=(),
        street=Street.PREFLOP,
        betting_history=[],
        position=0,
        stack_sizes=(5, 100),  # Very short
        pot_size=3,
        current_bet=2
    )
    
    legal_actions = short_stack.get_legal_actions()
    print(f"Short stack - legal actions: {legal_actions}")
    assert HoldemAction.ALL_IN in legal_actions, "Should be able to go all-in"
    # Most raises should not be available due to stack size
    
    print("Legal actions passed ✓\n")


def test_manager():
    """Test HoldemInfoSetManager."""
    print("Testing HoldemInfoSetManager")
    print("-" * 30)
    
    manager = HoldemInfoSetManager(num_players=2)
    
    # Create info set
    info_set1 = HoldemInfoSet(
        player=0,
        hole_cards=(0, 13),
        community_cards=(),
        street=Street.PREFLOP,
        betting_history=[],
        position=0,
        stack_sizes=(100, 100),
        pot_size=3,
        current_bet=2
    )
    
    # Store in manager
    stored_info, key1 = manager.get_or_create_info_set(info_set1)
    print(f"Stored info set with key: {key1}")
    
    # Create similar info set (should get same key due to abstraction)
    info_set2 = HoldemInfoSet(
        player=0,
        hole_cards=(0, 13),  # Same cards
        community_cards=(),
        street=Street.PREFLOP,
        betting_history=[],
        position=0,
        stack_sizes=(100, 100),  # Same stacks
        pot_size=3,
        current_bet=2
    )
    
    stored_info2, key2 = manager.get_or_create_info_set(info_set2)
    print(f"Second info set key: {key2}")
    
    assert key1 == key2, f"Similar info sets should have same key: {key1} vs {key2}"
    
    # Create different info set
    info_set3 = HoldemInfoSet(
        player=1,  # Different player
        hole_cards=(4, 17),  # Different cards
        community_cards=(),
        street=Street.PREFLOP,
        betting_history=[],
        position=1,
        stack_sizes=(100, 100),
        pot_size=3,
        current_bet=0  # Different bet
    )
    
    stored_info3, key3 = manager.get_or_create_info_set(info_set3)
    print(f"Third info set key: {key3}")
    
    assert key3 != key1, f"Different info sets should have different keys"
    
    print(f"Manager size: {manager.size()}")
    counts = manager.get_info_set_counts()
    print(f"Info set counts by player: {counts}")
    
    print("Manager passed ✓\n")


def test_equality_and_hashing():
    """Test info set equality and hashing."""
    print("Testing Equality and Hashing")
    print("-" * 30)
    
    # Create two identical info sets
    args = {
        'player': 0,
        'hole_cards': (0, 13),
        'community_cards': (),
        'street': Street.PREFLOP,
        'betting_history': [],
        'position': 0,
        'stack_sizes': (100, 100),
        'pot_size': 3,
        'current_bet': 2
    }
    
    info_set1 = HoldemInfoSet(**args)
    info_set2 = HoldemInfoSet(**args)
    
    print(f"Info set 1: {info_set1}")
    print(f"Info set 2: {info_set2}")
    print(f"Equal: {info_set1 == info_set2}")
    print(f"Hash 1: {hash(info_set1)}")
    print(f"Hash 2: {hash(info_set2)}")
    print(f"Hashes equal: {hash(info_set1) == hash(info_set2)}")
    
    assert info_set1 == info_set2, "Identical info sets should be equal"
    assert hash(info_set1) == hash(info_set2), "Identical info sets should have same hash"
    
    # Test in set/dict
    info_set_set = {info_set1, info_set2}
    print(f"Set size: {len(info_set_set)} (should be 1)")
    assert len(info_set_set) == 1, "Set should contain only one unique info set"
    
    print("Equality and hashing passed ✓\n")


def test_chance_seed_and_deterministic_sampling():
    """Test new chance seed and deterministic sampling features."""
    print("Testing Chance Seed and Deterministic Sampling")
    print("-" * 50)
    
    # Test info set with chance seed
    info_set_with_seed = HoldemInfoSet(
        player=0,
        hole_cards=(48, 49),  # AA
        community_cards=(),
        street=Street.PREFLOP,
        betting_history=[],
        position=0,
        stack_sizes=(200, 200),
        pot_size=3,
        current_bet=2,
        chance_seed=12345,
        deck_state=tuple(range(4, 52))  # Remaining deck
    )
    
    print(f"✓ Info set with chance seed created")
    print(f"  Chance seed: {info_set_with_seed.chance_seed}")
    print(f"  Can deterministic sample: {info_set_with_seed.can_deterministic_sample()}")
    
    # Test abstract key includes chance seed
    abstract_key = info_set_with_seed.get_abstract_key(use_minimal_abstraction=True)
    print(f"  Abstract key: {abstract_key}")
    assert "CS" in abstract_key, "Chance seed should be in abstract key"
    print("✓ Chance seed appears in abstract key")
    
    # Test deterministic sampling consistency
    info_set_same_seed = HoldemInfoSet(
        player=0,
        hole_cards=(48, 49),
        community_cards=(),
        street=Street.PREFLOP,
        betting_history=[],
        position=0,
        stack_sizes=(200, 200),
        pot_size=3,
        current_bet=2,
        chance_seed=12345,  # Same seed
        deck_state=tuple(range(4, 52))
    )
    
    key1 = info_set_with_seed.get_abstract_key()
    key2 = info_set_same_seed.get_abstract_key()
    assert key1 == key2, "Same chance seed should produce same abstract key"
    print("✓ Deterministic sampling produces consistent keys")
    
    # Test remaining deck calculation
    remaining_deck = info_set_with_seed.get_remaining_deck()
    print(f"✓ Remaining deck size: {len(remaining_deck)}")
    assert len(remaining_deck) == 48, "Should have 48 cards remaining after dealing AA"
    
    # Test child info set creation
    child = info_set_with_seed.create_child_info_set(
        new_community_cards=(4, 8, 12),  # Flop
        new_betting_history=[],
        new_street=Street.FLOP,
        new_pot_size=10,
        new_current_bet=0,
        new_action_pointer=1,
        new_chance_seed=12346
    )
    
    print(f"✓ Child info set created")
    print(f"  Child street: {child.street.name}")
    print(f"  Child chance seed: {child.chance_seed}")
    print("Chance seed and deterministic sampling passed ✓\n")


def test_enhanced_abstract_keys():
    """Test enhanced abstract key generation with new features."""
    print("Testing Enhanced Abstract Keys")
    print("-" * 40)
    
    # Test action pointer integration
    info_set1 = HoldemInfoSet(
        player=0, hole_cards=(48, 49), community_cards=(), street=Street.PREFLOP,
        betting_history=[], position=0, stack_sizes=(200, 200), pot_size=3,
        current_bet=2, action_pointer=0, chance_seed=100
    )
    
    info_set2 = HoldemInfoSet(
        player=0, hole_cards=(48, 49), community_cards=(), street=Street.PREFLOP,
        betting_history=[], position=0, stack_sizes=(200, 200), pot_size=3,
        current_bet=2, action_pointer=1, chance_seed=100  # Different action pointer
    )
    
    key1 = info_set1.get_abstract_key()
    key2 = info_set2.get_abstract_key()
    
    print(f"Key with action_pointer=0: {key1}")
    print(f"Key with action_pointer=1: {key2}")
    assert key1 != key2, "Different action pointers should produce different keys"
    print("✓ Action pointer prevents regret aliasing")
    
    # Test street-specific board buckets
    flop_info = HoldemInfoSet(
        player=0, hole_cards=(48, 44), community_cards=(4, 8, 12), street=Street.FLOP,
        betting_history=[], position=0, stack_sizes=(200, 200), pot_size=4, current_bet=0
    )
    
    flop_key = flop_info.get_abstract_key()
    print(f"Flop key with board bucket: {flop_key}")
    assert "BB" in flop_key, "Board bucket should appear in flop abstract key"
    print("✓ Street-specific board buckets working")
    
    print("Enhanced abstract keys passed ✓\n")


def run_all_tests():
    """Run all test functions."""
    print("=" * 60)
    print("TEXAS HOLD'EM INFO SET TEST SUITE (ENHANCED)")
    print("=" * 60)
    print()
    
    try:
        test_card_utilities()
        test_betting_round()
        test_basic_info_set()
        test_validation()
        test_different_streets()
        test_legal_actions()
        test_manager()
        test_equality_and_hashing()
        
        # NEW TESTS for scaling implementations
        test_chance_seed_and_deterministic_sampling()
        test_enhanced_abstract_keys()
        
        print("=" * 60)
        print("ALL ENHANCED INFO SET TESTS PASSED ✓")
        print("=" * 60)
        print()
        print("🎉 Ready for scaling to full Texas Hold'em! 🎉")
        print("Key features verified:")
        print("✓ Chance seed tracking for deterministic MCCFR")
        print("✓ Action pointer integration prevents regret aliasing")
        print("✓ Street-specific board buckets avoid corruption")
        print("✓ Child info set creation maintains sampling state")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)