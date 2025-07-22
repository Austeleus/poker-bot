"""
Baseline agent implementation for MCCFR strategies.

This module provides agent wrappers that can play poker using
strategies learned from MCCFR training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Optional, Tuple
import numpy as np
from abc import ABC, abstractmethod

from envs.kuhn_poker import KuhnPokerEnv, Action
from core.info_set import info_set_from_game_state
from core.regret_storage import StrategyProfile


class PokerAgent(ABC):
    """Abstract base class for poker agents."""
    
    def __init__(self, player_id: int, name: str = "PokerAgent"):
        self.player_id = player_id
        self.name = name
        self.rng = np.random.default_rng()
    
    @abstractmethod
    def get_action(self, env: KuhnPokerEnv) -> int:
        """Get action for current game state."""
        pass
    
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducible behavior."""
        self.rng = np.random.default_rng(seed)


class MCCFRAgent(PokerAgent):
    """
    Agent that plays using a strategy learned from MCCFR training.
    
    This agent looks up the current information set and samples
    actions according to the learned strategy.
    """
    
    def __init__(self, player_id: int, strategy_profile: StrategyProfile, 
                 name: str = "MCCFR Agent"):
        super().__init__(player_id, name)
        self.strategy_profile = strategy_profile
        
        # Statistics tracking
        self.actions_taken = 0
        self.info_sets_encountered = set()
        self.action_history = []
    
    def get_action(self, env: KuhnPokerEnv) -> int:
        """
        Get action based on MCCFR-learned strategy.
        
        Args:
            env: Current game environment
            
        Returns:
            Action to take (0 or 1 for Kuhn poker)
        """
        if env.current_player != self.player_id:
            raise ValueError(f"Agent {self.player_id} asked to act, but current player is {env.current_player}")
        
        if env.is_terminal_state:
            raise ValueError("Cannot get action from terminal state")
        
        # Get current information set
        info_set = info_set_from_game_state(env, self.player_id)
        info_set_key = info_set.to_string()
        legal_actions = env.get_legal_actions()
        
        # Track statistics
        self.info_sets_encountered.add(info_set_key)
        
        # Get strategy for this information set
        strategy = self.strategy_profile.get_strategy(info_set_key)
        
        if not strategy:
            # Fall back to uniform random if no strategy available
            action = self.rng.choice(legal_actions)
        else:
            # Sample action according to strategy
            action = self.strategy_profile.sample_action(
                info_set_key, legal_actions, self.rng
            )
        
        # Track action
        self.actions_taken += 1
        self.action_history.append((info_set_key, action))
        
        return action
    
    def get_action_probabilities(self, env: KuhnPokerEnv) -> Dict[int, float]:
        """Get action probabilities for current state."""
        info_set = info_set_from_game_state(env, self.player_id)
        info_set_key = info_set.to_string()
        
        strategy = self.strategy_profile.get_strategy(info_set_key)
        
        if not strategy:
            # Uniform probabilities if no strategy
            legal_actions = env.get_legal_actions()
            uniform_prob = 1.0 / len(legal_actions)
            return {action: uniform_prob for action in legal_actions}
        
        return strategy.copy()
    
    def reset_statistics(self) -> None:
        """Reset agent statistics."""
        self.actions_taken = 0
        self.info_sets_encountered.clear()
        self.action_history.clear()
    
    def get_statistics(self) -> Dict[str, any]:
        """Get agent statistics."""
        return {
            'actions_taken': self.actions_taken,
            'unique_info_sets_encountered': len(self.info_sets_encountered),
            'info_sets_encountered': list(self.info_sets_encountered),
            'action_history': self.action_history.copy()
        }


class RandomAgent(PokerAgent):
    """Agent that plays randomly (useful for testing and baselines)."""
    
    def __init__(self, player_id: int, name: str = "Random Agent"):
        super().__init__(player_id, name)
    
    def get_action(self, env: KuhnPokerEnv) -> int:
        """Get random legal action."""
        legal_actions = env.get_legal_actions()
        return self.rng.choice(legal_actions)


class UniformAgent(PokerAgent):
    """Agent that always plays uniform random strategy."""
    
    def __init__(self, player_id: int, name: str = "Uniform Agent"):
        super().__init__(player_id, name)
    
    def get_action(self, env: KuhnPokerEnv) -> int:
        """Get uniformly random action."""
        legal_actions = env.get_legal_actions()
        return self.rng.choice(legal_actions)


class AlwaysFoldAgent(PokerAgent):
    """Agent that always folds when possible (very exploitable baseline)."""
    
    def __init__(self, player_id: int, name: str = "Always Fold Agent"):
        super().__init__(player_id, name)
    
    def get_action(self, env: KuhnPokerEnv) -> int:
        """Always fold if possible, otherwise check/call."""
        legal_actions = env.get_legal_actions()
        
        # If facing a bet, fold (action 1 = BET_FOLD becomes fold)
        info_set = info_set_from_game_state(env, self.player_id)
        if info_set.history.endswith('B'):
            return Action.BET_FOLD.value  # Fold
        else:
            return Action.CHECK_CALL.value  # Check


class AlwaysBetAgent(PokerAgent):
    """Agent that always bets/calls when possible (aggressive baseline)."""
    
    def __init__(self, player_id: int, name: str = "Always Bet Agent"):
        super().__init__(player_id, name)
    
    def get_action(self, env: KuhnPokerEnv) -> int:
        """Always bet if possible, call if facing bet."""
        info_set = info_set_from_game_state(env, self.player_id)
        
        if info_set.history.endswith('B'):
            return Action.CHECK_CALL.value  # Call when facing bet
        else:
            return Action.BET_FOLD.value  # Bet when possible


def _create_agent_for_position(template_agent: PokerAgent, player_id: int) -> PokerAgent:
    """Create a new agent instance with the specified player ID."""
    if isinstance(template_agent, MCCFRAgent):
        return MCCFRAgent(player_id, template_agent.strategy_profile, template_agent.name)
    elif isinstance(template_agent, RandomAgent):
        return RandomAgent(player_id, template_agent.name)
    elif isinstance(template_agent, UniformAgent):
        return UniformAgent(player_id, template_agent.name)
    elif isinstance(template_agent, AlwaysFoldAgent):
        return AlwaysFoldAgent(player_id, template_agent.name)
    elif isinstance(template_agent, AlwaysBetAgent):
        return AlwaysBetAgent(player_id, template_agent.name)
    else:
        # Generic copy - just change player_id
        if hasattr(template_agent, 'strategy_profile'):
            new_agent = type(template_agent)(player_id, template_agent.strategy_profile, template_agent.name)
        else:
            new_agent = type(template_agent)(player_id, template_agent.name)
        return new_agent


def play_game(agent1: PokerAgent, agent2: PokerAgent, seed: Optional[int] = None) -> Tuple[int, List[str]]:
    """
    Play a single game between two agents.
    
    Args:
        agent1: Player 0 agent
        agent2: Player 1 agent
        seed: Random seed for game
        
    Returns:
        Tuple of (winner_payoff, game_log)
    """
    env = KuhnPokerEnv()
    env.reset(seed=seed)
    
    agents = [agent1, agent2]
    game_log = []
    
    game_log.append(f"Game start: P0 card={env.player_cards[0]}, P1 card={env.player_cards[1]}")
    
    while not env.is_terminal_state:
        current_player = env.current_player
        current_agent = agents[current_player]
        
        # Get action from agent
        action = current_agent.get_action(env)
        
        # Get action meaning for logging
        info_set = info_set_from_game_state(env, current_player)
        action_name = info_set.get_action_meaning(action)
        
        game_log.append(f"Player {current_player} ({current_agent.name}): {action_name}")
        
        # Take action
        env.step(action)
    
    # Game finished
    p0_payoff = env.get_payoff(0)
    p1_payoff = env.get_payoff(1)
    
    game_log.append(f"Game end: P0 payoff={p0_payoff}, P1 payoff={p1_payoff}")
    game_log.append(f"Final history: {env.history}")
    
    return p0_payoff, game_log


def play_tournament(agents: List[PokerAgent], num_games: int = 100, 
                   seed: Optional[int] = None, verbose: bool = False) -> Dict[str, any]:
    """
    Play a round-robin tournament between agents.
    
    Args:
        agents: List of agents to compete
        num_games: Number of games per matchup
        seed: Random seed
        verbose: Whether to print detailed results
        
    Returns:
        Tournament results dictionary
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_agents = len(agents)
    results = {}
    
    # Initialize results
    for i, agent in enumerate(agents):
        results[agent.name] = {
            'total_payoff': 0.0,
            'games_played': 0,
            'wins': 0,
            'losses': 0,
            'draws': 0,
            'matchups': {}
        }
    
    # Play all matchups
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            agent1_template = agents[i]
            agent2_template = agents[j]
            
            matchup_results = {'agent1_payoff': 0.0, 'agent2_payoff': 0.0}
            
            if verbose:
                print(f"\nMatchup: {agent1_template.name} vs {agent2_template.name}")
            
            for game in range(num_games):
                # Create agents with correct player IDs for this game
                if game % 2 == 0:
                    # agent1 as P0, agent2 as P1
                    p0_agent = _create_agent_for_position(agent1_template, 0)
                    p1_agent = _create_agent_for_position(agent2_template, 1)
                    p0_payoff, game_log = play_game(p0_agent, p1_agent, seed=game if seed else None)
                    agent1_payoff = p0_payoff
                    agent2_payoff = -p0_payoff
                else:
                    # agent2 as P0, agent1 as P1
                    p0_agent = _create_agent_for_position(agent2_template, 0)
                    p1_agent = _create_agent_for_position(agent1_template, 1)
                    p0_payoff, game_log = play_game(p0_agent, p1_agent, seed=game if seed else None)
                    agent1_payoff = -p0_payoff
                    agent2_payoff = p0_payoff
                
                # Update results
                matchup_results['agent1_payoff'] += agent1_payoff
                matchup_results['agent2_payoff'] += agent2_payoff
                
                # Update individual agent stats
                results[agent1_template.name]['total_payoff'] += agent1_payoff
                results[agent1_template.name]['games_played'] += 1
                results[agent2_template.name]['total_payoff'] += agent2_payoff
                results[agent2_template.name]['games_played'] += 1
                
                # Update win/loss/draw counts
                if agent1_payoff > 0:
                    results[agent1_template.name]['wins'] += 1
                    results[agent2_template.name]['losses'] += 1
                elif agent1_payoff < 0:
                    results[agent1_template.name]['losses'] += 1
                    results[agent2_template.name]['wins'] += 1
                else:
                    results[agent1_template.name]['draws'] += 1
                    results[agent2_template.name]['draws'] += 1
            
            # Store matchup results
            results[agent1_template.name]['matchups'][agent2_template.name] = matchup_results['agent1_payoff'] / num_games
            results[agent2_template.name]['matchups'][agent1_template.name] = matchup_results['agent2_payoff'] / num_games
            
            if verbose:
                avg1 = matchup_results['agent1_payoff'] / num_games
                avg2 = matchup_results['agent2_payoff'] / num_games
                print(f"  {agent1_template.name}: {avg1:.4f} avg payoff")
                print(f"  {agent2_template.name}: {avg2:.4f} avg payoff")
    
    # Calculate final averages
    for agent_name, agent_results in results.items():
        games_played = agent_results['games_played']
        if games_played > 0:
            agent_results['avg_payoff'] = agent_results['total_payoff'] / games_played
        else:
            agent_results['avg_payoff'] = 0.0
    
    return results


def print_tournament_results(results: Dict[str, any]) -> None:
    """Print formatted tournament results."""
    print("\nTournament Results")
    print("=" * 50)
    
    # Sort agents by average payoff
    sorted_agents = sorted(results.items(), key=lambda x: x[1]['avg_payoff'], reverse=True)
    
    for rank, (agent_name, agent_results) in enumerate(sorted_agents, 1):
        print(f"{rank}. {agent_name}")
        print(f"   Average payoff: {agent_results['avg_payoff']:.4f}")
        print(f"   Total payoff: {agent_results['total_payoff']:.2f}")
        print(f"   Record: {agent_results['wins']}-{agent_results['losses']}-{agent_results['draws']}")
        print(f"   Games played: {agent_results['games_played']}")
        print()


def run_agent_test():
    """Test the agent implementations."""
    print("Testing Poker Agents")
    print("=" * 30)
    
    # Create test agents
    random_agent1 = RandomAgent(0, "Random 1")
    random_agent2 = RandomAgent(1, "Random 2")
    fold_agent = AlwaysFoldAgent(0, "Folder")
    bet_agent = AlwaysBetAgent(1, "Bettor")
    
    # Play a few test games
    print("Test game 1: Random vs Random")
    payoff, log = play_game(random_agent1, random_agent2, seed=42)
    for line in log:
        print(f"  {line}")
    
    print(f"\nTest game 2: Folder vs Bettor")
    payoff, log = play_game(fold_agent, bet_agent, seed=123)
    for line in log:
        print(f"  {line}")
    
    # Mini tournament
    print(f"\nMini tournament:")
    agents = [random_agent1, fold_agent, bet_agent]
    tournament_results = play_tournament(agents, num_games=20, seed=42, verbose=False)
    print_tournament_results(tournament_results)
    
    return agents, tournament_results


if __name__ == "__main__":
    run_agent_test()