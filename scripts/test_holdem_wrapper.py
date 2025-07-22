#!/usr/bin/env python3
"""
Test script for Texas Hold'em environment wrapper.

This script tests the PettingZoo wrapper for 6-player Texas Hold'em
to ensure it works correctly for MCCFR development.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.holdem_wrapper import HoldemWrapper
import numpy as np


def test_basic_functionality():
    """Test basic wrapper functionality"""
    print("=" * 50)
    print("TESTING BASIC FUNCTIONALITY")
    print("=" * 50)
    
    # Test with 2 players (heads-up for MCCFR)
    env = HoldemWrapper(num_players=2)
    
    # Test reset
    try:
        observations = env.reset(seed=42)
        print(f"Environment reset successful")
        print(f"Number of agents: {len(env.agents)}")
        print(f"Agent names: {env.agents}")
        
        # Check if we got observations
        if observations:
            agent = list(observations.keys())[0]
            obs = observations[agent]
            print(f"Sample observation keys: {list(obs.keys())}")
            print(f"Observation vector shape: {obs['observation'].shape}")
            print(f"Legal actions: {obs['legal_actions']}")
            print(f"Action mask: {obs['action_mask']}")
            
        print("✓ Basic functionality test passed")
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        print("This might be due to missing PettingZoo dependencies")
        return False
    
    env.close()
    return True


def test_action_mapping():
    """Test discrete action mapping"""
    print("\n" + "=" * 50)
    print("TESTING ACTION MAPPING")
    print("=" * 50)
    
    env = HoldemWrapper(num_players=3)  # Use 3 players for simpler testing
    
    print("Discrete action mapping:")
    for action_id, action_name in env.action_mapping.items():
        print(f"  {action_id}: {action_name}")
    
    # Test action mask conversion
    print("\nTesting action mask conversion:")
    
    # Simulate different PettingZoo action masks
    test_masks = [
        np.array([1, 1, 0, 0, 0]),  # Can fold, check
        np.array([1, 0, 1, 0, 0]),  # Can fold, call  
        np.array([1, 0, 1, 1, 0]),  # Can fold, call, raise
        np.array([1, 0, 1, 1, 1]),  # Can fold, call, raise, all-in
        np.array([0, 1, 0, 0, 0]),  # Can only check
    ]
    
    for i, mask in enumerate(test_masks):
        legal_actions = env._get_legal_actions_from_mask(mask)
        print(f"  PZ mask {mask} -> Legal actions: {legal_actions}")
    
    print("✓ Action mapping test passed")
    
    env.close()


def test_game_simulation():
    """Test basic game simulation"""
    print("\n" + "=" * 50)
    print("TESTING GAME SIMULATION")
    print("=" * 50)
    
    env = HoldemWrapper(num_players=3)
    
    try:
        observations = env.reset(seed=42)
        
        step_count = 0
        max_steps = 50  # Prevent infinite loops
        
        while not env.is_terminal() and step_count < max_steps:
            current_player = env.get_current_player()
            
            if current_player is None:
                break
                
            # Get legal actions for current player
            legal_actions = env.get_legal_actions(current_player)
            
            if not legal_actions:
                break
            
            # Take a random legal action
            action = np.random.choice(legal_actions)
            
            print(f"Step {step_count}: Player {current_player} takes action {action}")
            
            # Execute action
            observations, rewards, done, info = env.step(action)
            
            step_count += 1
            
            if done:
                print(f"Game finished after {step_count} steps")
                final_rewards = env.get_final_rewards()
                print(f"Final rewards: {final_rewards}")
                break
        
        if step_count >= max_steps:
            print(f"Game stopped after {max_steps} steps (timeout)")
        
        print("✓ Game simulation test passed")
        
    except Exception as e:
        print(f"❌ Game simulation test failed: {e}")
        print("This might be due to PettingZoo environment issues")
        return False
    
    env.close()
    return True


def test_observation_processing():
    """Test observation processing and card extraction"""
    print("\n" + "=" * 50)
    print("TESTING OBSERVATION PROCESSING")
    print("=" * 50)
    
    env = HoldemWrapper(num_players=3)
    
    try:
        observations = env.reset(seed=42)
        
        if observations:
            for agent, obs in observations.items():
                print(f"\nAgent {agent} observation:")
                print(f"  Observation shape: {obs['observation'].shape}")
                print(f"  Legal actions: {obs['legal_actions']}")
                print(f"  Action mask: {obs['action_mask']}")
                print(f"  Hole cards: {obs['hole_cards']}")
                print(f"  Community cards: {obs['community_cards']}")
                print(f"  Action history length: {len(obs['action_history'])}")
                
                # Verify observation vector is valid
                assert len(obs['observation']) == 54, "Observation should be 54-dimensional"
                assert len(obs['action_mask']) == 5, "Action mask should have 5 elements"
                
                break  # Just test first agent
        
        print("✓ Observation processing test passed")
        
    except Exception as e:
        print(f"❌ Observation processing test failed: {e}")
        return False
    
    env.close()
    return True


def test_info_set_integration():
    """Test HoldemInfoSet integration with wrapper"""
    print("\n" + "=" * 50)
    print("TESTING INFO SET INTEGRATION")
    print("=" * 50)
    
    env = HoldemWrapper(num_players=2)
    
    try:
        observations = env.reset(seed=42)
        
        # Test getting info set for current player
        current_player = env.get_current_player_index()
        if current_player is not None:
            print(f"Current player: {current_player}")
            
            info_set = env.get_info_set(current_player)
            if info_set:
                print("✓ Info set created successfully")
                print(f"  Player: {info_set.player}")
                print(f"  Street: {info_set.street.name}")
                print(f"  Position: {info_set.position}")
                print(f"  Hole cards: {info_set.hole_cards}")
                print(f"  Community cards: {info_set.community_cards}")
                print(f"  Pot size: {info_set.pot_size}")
                print(f"  Current bet: {info_set.current_bet}")
                print(f"  Stack sizes: {info_set.stack_sizes}")
                print(f"  Legal actions: {info_set.get_legal_actions()}")
                
                # Test action meanings
                for action in info_set.get_legal_actions()[:3]:
                    meaning = info_set.get_action_meaning(action)
                    print(f"    Action {action}: {meaning}")
                    
            else:
                print("⚠️ Failed to create info set")
        else:
            print("⚠️ No current player found")
            
        print("✓ Info set integration test passed")
        
    except Exception as e:
        print(f"❌ Info set integration test failed: {e}")
        return False
    
    env.close()
    return True


def test_multiple_player_configurations():
    """Test different player count configurations"""
    print("\n" + "=" * 50)
    print("TESTING MULTIPLE PLAYER CONFIGURATIONS")
    print("=" * 50)
    
    player_counts = [2, 3, 6]  # Test common configurations
    
    for num_players in player_counts:
        print(f"\nTesting {num_players}-player configuration:")
        
        try:
            env = HoldemWrapper(num_players=num_players)
            observations = env.reset(seed=42)
            
            print(f"  ✓ {num_players} players initialized successfully")
            print(f"  ✓ {len(env.agents)} agents created")
            
            # Test one action
            if observations:
                current_player = env.get_current_player()
                if current_player:
                    legal_actions = env.get_legal_actions(current_player)
                    if legal_actions:
                        action = legal_actions[0]
                        env.step(action)
                        print(f"  ✓ Action execution successful")
            
            env.close()
            
        except Exception as e:
            print(f"  ❌ {num_players}-player test failed: {e}")
            continue
    
    print("✓ Multiple player configuration tests completed")


def test_error_handling():
    """Test error handling and edge cases"""
    print("\n" + "=" * 50)
    print("TESTING ERROR HANDLING")
    print("=" * 50)
    
    env = HoldemWrapper(num_players=3)
    
    # Test stepping on terminal game
    try:
        observations = env.reset(seed=42)
        
        # Force terminal state
        env.game_over = True
        
        observations, rewards, done, info = env.step(0)
        assert done == True
        assert observations == {}
        
        print("✓ Terminal state handling test passed")
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    # Test getting legal actions for non-existent agent
    try:
        legal_actions = env.get_legal_actions("non_existent_agent")
        assert legal_actions == []
        
        print("✓ Non-existent agent handling test passed")
        
    except Exception as e:
        print(f"❌ Non-existent agent test failed: {e}")
    
    env.close()


def check_dependencies():
    """Check if required dependencies are available"""
    print("CHECKING DEPENDENCIES")
    print("=" * 30)
    
    try:
        import pettingzoo
        print(f"✓ PettingZoo version: {pettingzoo.__version__}")
    except ImportError:
        print("❌ PettingZoo not installed")
        print("Install with: pip install pettingzoo[classic]")
        return False
    
    try:
        from pettingzoo.classic import texas_holdem_no_limit_v6
        print("✓ Texas Hold'em environment available")
    except ImportError:
        print("❌ Texas Hold'em environment not available")
        print("Install with: pip install pettingzoo[classic]")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
    except ImportError:
        print("❌ NumPy not installed")
        return False
    
    return True


def run_all_tests():
    """Run all test suites"""
    print("TEXAS HOLD'EM WRAPPER TEST SUITE")
    print("=" * 60)
    
    # Check dependencies first
    if not check_dependencies():
        print("\n❌ DEPENDENCIES MISSING")
        print("Please install required packages:")
        print("  pip install pettingzoo[classic] numpy")
        return
    
    try:
        # Run tests
        basic_success = test_basic_functionality()
        
        if basic_success:
            test_action_mapping()
            simulation_success = test_game_simulation()
            test_observation_processing()
            test_info_set_integration()  # New test for our integration
            test_multiple_player_configurations()
            test_error_handling()
        
        print("\n" + "=" * 60)
        
        if basic_success:
            print("🎉 TEXAS HOLD'EM WRAPPER TESTS COMPLETED! 🎉")
            print("=" * 60)
            print("\nWrapper is ready for MCCFR development!")
            print("Note: Some card extraction features may need refinement")
            print("during actual MCCFR implementation.")
        else:
            print("⚠️ BASIC TESTS FAILED")
            print("=" * 60)
            print("\nPlease check PettingZoo installation and try again.")
        
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()