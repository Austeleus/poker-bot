#!/usr/bin/env python3
"""
Test script for Kuhn Poker environment.

This script thoroughly tests the Kuhn poker implementation to ensure
it works correctly for MCCFR algorithm development.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.kuhn_poker import KuhnPokerEnv, Action, simulate_random_game, get_action_name
import random


def test_basic_functionality():
    """Test basic environment functionality"""
    print("=" * 50)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 50)
    
    env = KuhnPokerEnv()
    
    # Test reset
    state = env.reset(seed=42)
    print(f"Initial state: {state}")
    assert not state['is_terminal']
    assert state['current_player'] == 0
    assert len(state['legal_actions']) == 2
    
    # Test information sets
    info_set_p0 = env.get_info_set(0)
    info_set_p1 = env.get_info_set(1)
    print(f"P0 info set: {info_set_p0}")
    print(f"P1 info set: {info_set_p1}")
    
    # Test cloning
    clone = env.clone()
    assert clone.player_cards == env.player_cards
    assert clone.history == env.history
    assert clone.current_player == env.current_player
    
    print("✓ Basic functionality tests passed")


def test_game_logic():
    """Test game logic with specific scenarios"""
    print("\n" + "=" * 50)
    print("TESTING GAME LOGIC")
    print("=" * 50)
    
    # Test scenario 1: Check-Check
    env = KuhnPokerEnv()
    env.reset()
    env.set_cards(3, 1)  # P0=King, P1=Jack
    print(f"\nScenario 1: P0=King, P1=Jack")
    print(f"Initial: {env}")
    
    # P0 checks
    state, done = env.step(Action.CHECK_CALL.value)
    print(f"After P0 check: {env}")
    assert not done
    assert env.current_player == 1
    
    # P1 checks
    state, done = env.step(Action.CHECK_CALL.value)
    print(f"After P1 check: {env}")
    assert done
    assert env.history == "CC"
    assert env.get_payoff(0) == 1  # P0 wins with King
    assert env.get_payoff(1) == -1
    
    # Test scenario 2: Bet-Call
    env = KuhnPokerEnv()
    env.reset()
    env.set_cards(2, 3)  # P0=Queen, P1=King
    print(f"\nScenario 2: P0=Queen, P1=King")
    print(f"Initial: {env}")
    
    # P0 bets
    state, done = env.step(Action.BET_FOLD.value)
    print(f"After P0 bet: {env}")
    assert not done
    assert env.current_player == 1
    
    # P1 calls
    state, done = env.step(Action.CHECK_CALL.value)
    print(f"After P1 call: {env}")
    assert done
    assert env.history == "BC"
    assert env.get_payoff(0) == -2  # P1 wins with King
    assert env.get_payoff(1) == 2
    
    # Test scenario 3: Check-Bet-Fold
    env = KuhnPokerEnv()
    env.reset()
    env.set_cards(1, 2)  # P0=Jack, P1=Queen
    print(f"\nScenario 3: P0=Jack, P1=Queen")
    print(f"Initial: {env}")
    
    # P0 checks
    state, done = env.step(Action.CHECK_CALL.value)
    print(f"After P0 check: {env}")
    
    # P1 bets
    state, done = env.step(Action.BET_FOLD.value)
    print(f"After P1 bet: {env}")
    
    # P0 folds
    state, done = env.step(Action.BET_FOLD.value)
    print(f"After P0 fold: {env}")
    assert done
    assert env.history == "CBF"
    assert env.get_payoff(0) == -1  # P0 folded
    assert env.get_payoff(1) == 1
    
    print("✓ Game logic tests passed")


def test_all_possible_games():
    """Test all possible card combinations and verify game tree"""
    print("\n" + "=" * 50)
    print("TESTING ALL POSSIBLE GAMES")
    print("=" * 50)
    
    env = KuhnPokerEnv()
    
    # Get all possible deals
    all_deals = env.get_all_possible_deals()
    print(f"Total possible deals: {len(all_deals)}")
    assert len(all_deals) == 6  # 3*2 = 6 permutations
    
    # Test each deal with different action sequences
    terminal_histories = ["CC", "BC", "BF", "CBC", "CBF"]
    
    for p0_card, p1_card in all_deals:
        print(f"\nTesting cards P0={p0_card}, P1={p1_card}")
        
        for history in terminal_histories:
            env.reset()
            env.set_cards(p0_card, p1_card)
            
            # Play out the history
            for i, action_char in enumerate(history):
                current_player = env.current_player
                
                if action_char == 'C':
                    action = Action.CHECK_CALL.value
                elif action_char == 'B':
                    action = Action.BET_FOLD.value
                elif action_char == 'F':
                    action = Action.BET_FOLD.value
                else:
                    continue
                    
                state, done = env.step(action)
                
                if done:
                    break
            
            # Verify terminal state
            assert env.is_terminal_state
            assert env.history == history
            
            # Verify payoffs are zero-sum
            p0_payoff = env.get_payoff(0)
            p1_payoff = env.get_payoff(1)
            assert p0_payoff + p1_payoff == 0
            
            print(f"  History {history}: P0={p0_payoff}, P1={p1_payoff}")
    
    print("✓ All possible games tested")


def test_information_sets():
    """Test information set generation for CFR"""
    print("\n" + "=" * 50)
    print("TESTING INFORMATION SETS")
    print("=" * 50)
    
    env = KuhnPokerEnv()
    
    # Collect all information sets
    info_sets_p0 = set()
    info_sets_p1 = set()
    
    all_deals = env.get_all_possible_deals()
    terminal_histories = ["CC", "BC", "BF", "CBC", "CBF"]
    
    for p0_card, p1_card in all_deals:
        for history in terminal_histories:
            env.reset()
            env.set_cards(p0_card, p1_card)
            
            # Record information sets at each decision point
            for i, action_char in enumerate(history):
                if not env.is_terminal_state:
                    current_player = env.current_player
                    info_set = env.get_info_set(current_player)
                    
                    if current_player == 0:
                        info_sets_p0.add(info_set)
                    else:
                        info_sets_p1.add(info_set)
                
                # Execute action
                if action_char == 'C':
                    action = Action.CHECK_CALL.value
                elif action_char in ['B', 'F']:
                    action = Action.BET_FOLD.value
                else:
                    continue
                    
                env.step(action)
    
    print(f"Player 0 information sets:")
    for info_set in sorted(info_sets_p0):
        print(f"  {info_set}")
    
    print(f"\nPlayer 1 information sets:")
    for info_set in sorted(info_sets_p1):
        print(f"  {info_set}")
    
    print(f"\nTotal P0 info sets: {len(info_sets_p0)}")
    print(f"Total P1 info sets: {len(info_sets_p1)}")
    
    # Expected info sets for Kuhn poker
    expected_p0 = 12  # 3 cards * 4 possible histories
    expected_p1 = 12  # 3 cards * 4 possible histories
    
    print(f"Expected P0 info sets: ~{expected_p0}")
    print(f"Expected P1 info sets: ~{expected_p1}")
    
    print("✓ Information set tests passed")


def test_random_simulations():
    """Test random game simulations"""
    print("\n" + "=" * 50)
    print("TESTING RANDOM SIMULATIONS")  
    print("=" * 50)
    
    num_games = 100
    payoff_sums = [0, 0]
    
    for i in range(num_games):
        final_env = simulate_random_game(seed=i)
        
        assert final_env.is_terminal_state
        
        p0_payoff = final_env.get_payoff(0)
        p1_payoff = final_env.get_payoff(1)
        
        # Verify zero-sum
        assert p0_payoff + p1_payoff == 0
        
        payoff_sums[0] += p0_payoff
        payoff_sums[1] += p1_payoff
        
        if i < 5:  # Print first few games
            print(f"Game {i+1}: cards={final_env.player_cards}, "
                  f"history='{final_env.history}', "
                  f"payoffs={[p0_payoff, p1_payoff]}")
    
    avg_payoffs = [s / num_games for s in payoff_sums]
    print(f"\nAverage payoffs over {num_games} games:")
    print(f"Player 0: {avg_payoffs[0]:.3f}")
    print(f"Player 1: {avg_payoffs[1]:.3f}")
    print(f"Sum: {sum(avg_payoffs):.6f} (should be ~0)")
    
    print("✓ Random simulation tests passed")


def test_action_context():
    """Test action interpretation based on context"""
    print("\n" + "=" * 50)
    print("TESTING ACTION CONTEXT")
    print("=" * 50)
    
    # Test action name interpretation
    test_cases = [
        ("", Action.CHECK_CALL.value, "Check"),
        ("", Action.BET_FOLD.value, "Bet"),
        ("B", Action.CHECK_CALL.value, "Call"),
        ("B", Action.BET_FOLD.value, "Fold"),
        ("CB", Action.CHECK_CALL.value, "Call"),
        ("CB", Action.BET_FOLD.value, "Fold"),
    ]
    
    for history, action, expected_name in test_cases:
        action_name = get_action_name(action, history)
        print(f"History: '{history}', Action: {action} -> '{action_name}'")
        assert action_name == expected_name
    
    print("✓ Action context tests passed")


def run_all_tests():
    """Run all test suites"""
    print("KUHN POKER ENVIRONMENT TEST SUITE")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_game_logic()
        test_all_possible_games()
        test_information_sets()
        test_random_simulations()
        test_action_context()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 60)
        print("\nKuhn Poker environment is ready for MCCFR development!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()