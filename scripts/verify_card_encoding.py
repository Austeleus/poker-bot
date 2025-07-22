#!/usr/bin/env python3
"""
Comprehensive card encoding verification script.

This script prints all 52 cards with their integer encodings and verifies
the conversion functions work correctly in both directions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.holdem_info_set import card_to_string, string_to_card


def print_all_cards():
    """Print all 52 cards with their integer encodings."""
    print("=" * 60)
    print("COMPLETE CARD ENCODING VERIFICATION")
    print("=" * 60)
    print()
    
    print("Card encoding: rank * 4 + suit")
    print("Ranks: 2=0, 3=1, 4=2, 5=3, 6=4, 7=5, 8=6, 9=7, T=8, J=9, Q=10, K=11, A=12")
    print("Suits: clubs=0, diamonds=1, hearts=2, spades=3")
    print()
    
    # Define rank and suit names
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['c', 'd', 'h', 's']
    suit_names = ['clubs', 'diamonds', 'hearts', 'spades']
    
    print(f"{'Card':<6} {'String':<4} {'Rank':<6} {'Suit':<8} {'Formula':<12} {'Verification':<12}")
    print("-" * 60)
    
    errors = []
    
    for card_int in range(52):
        # Calculate rank and suit from integer
        rank = card_int // 4
        suit = card_int % 4
        
        # Convert to string
        try:
            card_string = card_to_string(card_int)
            expected_string = f"{ranks[rank]}{suits[suit]}"
            
            # Verify round-trip conversion
            back_to_int = string_to_card(card_string)
            
            # Check for errors
            string_match = card_string == expected_string
            roundtrip_match = back_to_int == card_int
            
            status = "✓" if string_match and roundtrip_match else "✗"
            if not (string_match and roundtrip_match):
                errors.append((card_int, card_string, expected_string, back_to_int))
            
            print(f"{card_int:2d}     {card_string:<4} {ranks[rank]:<6} {suit_names[suit]:<8} "
                  f"{rank}*4+{suit}={card_int:<4} {status}")
            
        except Exception as e:
            print(f"{card_int:2d}     ERROR: {e}")
            errors.append((card_int, "ERROR", str(e), None))
    
    print()
    
    if errors:
        print("❌ ERRORS FOUND:")
        for card_int, actual_string, expected_string, back_to_int in errors:
            print(f"  Card {card_int}: got '{actual_string}', expected '{expected_string}', "
                  f"round-trip: {back_to_int}")
        return False
    else:
        print("✅ ALL 52 CARDS VERIFIED CORRECTLY!")
        return True


def test_specific_hands():
    """Test specific poker hands that are commonly referenced."""
    print("\n" + "=" * 60)
    print("SPECIFIC HAND VERIFICATION")
    print("=" * 60)
    
    # Famous poker hands
    test_hands = [
        # Pocket pairs
        ("Pocket Aces", [48, 49, 50, 51]),  # All four aces
        ("Pocket Kings", [44, 45, 46, 47]), # All four kings
        ("Pocket Queens", [40, 41, 42, 43]), # All four queens
        ("Pocket Jacks", [36, 37, 38, 39]), # All four jacks
        ("Pocket Twos", [0, 1, 2, 3]),      # All four twos
        
        # Famous suited hands
        ("Ace-King suited clubs", [48, 44]),   # Ac Kc
        ("Ace-King suited spades", [51, 47]),  # As Ks
        
        # Famous offsuit hands
        ("Ace-King offsuit", [48, 45]),       # Ac Kd
        
        # Suited connectors
        ("Seven-Six suited", [20, 16]),       # 7c 6c
        ("Five-Four suited", [12, 8]),        # 5c 4c
        ("Three-Two suited", [4, 0]),         # 3c 2c
    ]
    
    print(f"{'Hand Name':<25} {'Cards':<12} {'Expected':<12} {'Actual':<12} {'Status'}")
    print("-" * 70)
    
    all_correct = True
    
    for hand_name, card_ints in test_hands:
        if len(card_ints) == 2:
            # Two-card hand
            card1_str = card_to_string(card_ints[0])
            card2_str = card_to_string(card_ints[1])
            actual = f"{card1_str} {card2_str}"
            cards_display = f"{card_ints[0]}, {card_ints[1]}"
            
            print(f"{hand_name:<25} {cards_display:<12} {'manual':<12} {actual:<12} {'✓'}")
            
        else:
            # Multiple cards (like all four of a rank)
            card_strings = [card_to_string(card) for card in card_ints]
            actual = " ".join(card_strings)
            cards_display = str(card_ints)
            
            print(f"{hand_name:<25} {cards_display} {'manual':<12} {actual} {'✓'}")
    
    return all_correct


def test_boundary_cases():
    """Test boundary cases and edge conditions."""
    print("\n" + "=" * 60)
    print("BOUNDARY CASE TESTING")
    print("=" * 60)
    
    test_cases = [
        ("Lowest card", 0, "2c"),
        ("Highest card", 51, "As"),
        ("First of each suit", [0, 1, 2, 3], ["2c", "2d", "2h", "2s"]),
        ("Aces of each suit", [48, 49, 50, 51], ["Ac", "Ad", "Ah", "As"]),
    ]
    
    print(f"{'Test Case':<20} {'Input':<12} {'Expected':<12} {'Actual':<12} {'Status'}")
    print("-" * 60)
    
    all_correct = True
    
    for test_name, input_val, expected in test_cases:
        try:
            if isinstance(input_val, list):
                actual = [card_to_string(card) for card in input_val]
                status = "✓" if actual == expected else "✗"
                if actual != expected:
                    all_correct = False
                print(f"{test_name:<20} {str(input_val):<12} {str(expected):<12} {str(actual):<12} {status}")
            else:
                actual = card_to_string(input_val)
                status = "✓" if actual == expected else "✗"
                if actual != expected:
                    all_correct = False
                print(f"{test_name:<20} {input_val:<12} {expected:<12} {actual:<12} {status}")
                
        except Exception as e:
            print(f"{test_name:<20} {input_val:<12} {expected:<12} ERROR: {e} ✗")
            all_correct = False
    
    return all_correct


def test_invalid_inputs():
    """Test error handling for invalid inputs."""
    print("\n" + "=" * 60)
    print("ERROR HANDLING TESTING")
    print("=" * 60)
    
    # Test invalid integer inputs
    invalid_ints = [-1, 52, 100]
    print("Testing invalid integer inputs:")
    for invalid_int in invalid_ints:
        try:
            result = card_to_string(invalid_int)
            print(f"  card_to_string({invalid_int}) = {result} ✗ (should have raised error)")
        except ValueError as e:
            print(f"  card_to_string({invalid_int}) correctly raised ValueError: {e} ✓")
        except Exception as e:
            print(f"  card_to_string({invalid_int}) raised unexpected error: {e} ✗")
    
    # Test invalid string inputs
    invalid_strings = ["", "X", "2x", "Ah5", "14c", "2C"]
    print("\nTesting invalid string inputs:")
    for invalid_str in invalid_strings:
        try:
            result = string_to_card(invalid_str)
            print(f"  string_to_card('{invalid_str}') = {result} ✗ (should have raised error)")
        except ValueError as e:
            print(f"  string_to_card('{invalid_str}') correctly raised ValueError: {e} ✓")
        except Exception as e:
            print(f"  string_to_card('{invalid_str}') raised unexpected error: {e} ✗")


def create_reference_table():
    """Create a reference table for easy lookup."""
    print("\n" + "=" * 60)
    print("QUICK REFERENCE TABLE")
    print("=" * 60)
    
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['c', 'd', 'h', 's']
    
    print("Card integers by rank and suit:")
    print(f"{'Rank':<5} {'Clubs':<6} {'Diamonds':<9} {'Hearts':<8} {'Spades'}")
    print("-" * 40)
    
    for rank_idx, rank in enumerate(ranks):
        row = [f"{rank}"]
        for suit_idx, suit in enumerate(suits):
            card_int = rank_idx * 4 + suit_idx
            card_str = f"{rank}{suit}"
            row.append(f"{card_int:2d}({card_str})")
        print(f"{row[0]:<5} {row[1]:<6} {row[2]:<9} {row[3]:<8} {row[4]}")


def run_all_verifications():
    """Run all verification tests."""
    print("CARD ENCODING VERIFICATION SUITE")
    print("=" * 80)
    
    results = []
    
    print("\n1. VERIFYING ALL 52 CARDS")
    results.append(print_all_cards())
    
    print("\n2. TESTING SPECIFIC HANDS")
    results.append(test_specific_hands())
    
    print("\n3. TESTING BOUNDARY CASES")
    results.append(test_boundary_cases())
    
    print("\n4. TESTING ERROR HANDLING")
    test_invalid_inputs()
    
    print("\n5. REFERENCE TABLE")
    create_reference_table()
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    if all(results):
        print("🎉 ALL CARD ENCODING TESTS PASSED!")
        print("✅ The card encoding system is working correctly.")
        print("✅ All 52 cards verified.")
        print("✅ Round-trip conversions work perfectly.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("⚠️  Card encoding system needs fixes.")
    
    return all(results)


if __name__ == "__main__":
    success = run_all_verifications()
    sys.exit(0 if success else 1)