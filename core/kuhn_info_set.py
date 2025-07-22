"""
Information Set representation for poker games.

This module provides functionality to create and manage information sets
for CFR algorithms, starting with Kuhn poker and extensible to Texas Hold'em.
"""

from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import hashlib


class InfoSet(ABC):
    """Abstract base class for information sets in poker games."""
    
    def __init__(self, player: int):
        self.player = player
        self._hash = None
    
    @abstractmethod
    def to_string(self) -> str:
        """Convert information set to string representation."""
        pass
    
    @abstractmethod
    def get_legal_actions(self) -> List[int]:
        """Get legal actions available from this information set."""
        pass
    
    def __str__(self) -> str:
        return self.to_string()
    
    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self.to_string())
        return self._hash
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, InfoSet):
            return False
        return self.to_string() == other.to_string()


class KuhnInfoSet(InfoSet):
    """
    Information set for Kuhn poker.
    
    Format: "{card}/{history}"
    Examples: "1/", "2/C", "3/CB"
    
    In Kuhn poker:
    - Cards: 1=Jack, 2=Queen, 3=King
    - History: C=check/call, B=bet, F=fold
    """
    
    def __init__(self, player: int, card: int, history: str):
        super().__init__(player)
        self.card = card
        self.history = history
        
        # Validate inputs
        if card not in [1, 2, 3]:
            raise ValueError(f"Invalid card {card}. Must be 1, 2, or 3")
        if player not in [0, 1]:
            raise ValueError(f"Invalid player {player}. Must be 0 or 1")
        if not all(c in 'CBF' for c in history):
            raise ValueError(f"Invalid history '{history}'. Must contain only C, B, F")
    
    def to_string(self) -> str:
        """Convert to string format: card/history"""
        return f"{self.card}/{self.history}"
    
    def get_legal_actions(self) -> List[int]:
        """
        Get legal actions for Kuhn poker.
        
        In Kuhn poker, both actions are always legal:
        - Action 0: CHECK_CALL (check if no bet, call if facing bet)
        - Action 1: BET_FOLD (bet if no bet, fold if facing bet)
        """
        return [0, 1]
    
    def get_action_meaning(self, action: int) -> str:
        """Get the meaning of an action in current context."""
        if action == 0:  # CHECK_CALL
            if self.history.endswith('B'):
                return "Call"
            else:
                return "Check"
        elif action == 1:  # BET_FOLD
            if self.history.endswith('B'):
                return "Fold"
            else:
                return "Bet"
        else:
            raise ValueError(f"Invalid action {action}")
    
    def is_decision_point(self) -> bool:
        """Check if this information set represents a decision point."""
        # In Kuhn poker, all non-terminal info sets are decision points
        terminal_histories = ["CC", "BC", "BF", "CBC", "CBF"]
        return self.history not in terminal_histories
    
    @staticmethod
    def from_string(info_set_str: str, player: int) -> 'KuhnInfoSet':
        """Create KuhnInfoSet from string representation."""
        if '/' not in info_set_str:
            raise ValueError(f"Invalid info set string '{info_set_str}'. Must contain '/'")
        
        card_str, history = info_set_str.split('/', 1)
        try:
            card = int(card_str)
        except ValueError:
            raise ValueError(f"Invalid card '{card_str}'. Must be integer")
        
        return KuhnInfoSet(player, card, history)


class InfoSetManager:
    """
    Manages information sets for CFR algorithms.
    
    Provides functionality to:
    - Create and store information sets
    - Track which information sets belong to which players
    - Generate all possible information sets for a game
    """
    
    def __init__(self):
        self.info_sets: Dict[str, InfoSet] = {}
        self.player_info_sets: Dict[int, List[str]] = {0: [], 1: []}
    
    def get_or_create_info_set(self, info_set_key: str, player: int, **kwargs) -> InfoSet:
        """Get existing info set or create new one."""
        if info_set_key not in self.info_sets:
            # Create new info set based on game type
            if 'card' in kwargs and 'history' in kwargs:
                # Kuhn poker info set
                info_set = KuhnInfoSet(player, kwargs['card'], kwargs['history'])
            else:
                raise ValueError("Insufficient information to create info set")
            
            self.info_sets[info_set_key] = info_set
            if info_set_key not in self.player_info_sets[player]:
                self.player_info_sets[player].append(info_set_key)
        
        return self.info_sets[info_set_key]
    
    def get_info_set(self, info_set_key: str) -> Optional[InfoSet]:
        """Get information set by key."""
        return self.info_sets.get(info_set_key)
    
    def get_player_info_sets(self, player: int) -> List[InfoSet]:
        """Get all information sets for a specific player."""
        return [self.info_sets[key] for key in self.player_info_sets.get(player, [])]
    
    def get_all_info_sets(self) -> List[InfoSet]:
        """Get all information sets."""
        return list(self.info_sets.values())
    
    def generate_kuhn_poker_info_sets(self) -> None:
        """Generate all possible information sets for Kuhn poker."""
        cards = [1, 2, 3]  # Jack, Queen, King
        
        # All possible action histories that lead to decision points
        decision_histories = [
            "",     # Initial decision for P0
            "C",    # P1's decision after P0 checks
            "B",    # P1's decision after P0 bets
            "CB"    # P0's decision after P0 checks, P1 bets
        ]
        
        for card in cards:
            for history in decision_histories:
                # Determine which player acts next based on history length
                if len(history) % 2 == 0:
                    player = 0  # P0 acts on even history lengths
                else:
                    player = 1  # P1 acts on odd history lengths
                
                # Special case: "CB" is P0's decision (responding to P1's bet)
                if history == "CB":
                    player = 0
                
                info_set_key = f"{card}/{history}"
                self.get_or_create_info_set(info_set_key, player, card=card, history=history)
    
    def clear(self) -> None:
        """Clear all stored information sets."""
        self.info_sets.clear()
        self.player_info_sets = {0: [], 1: []}
    
    def size(self) -> int:
        """Get total number of information sets."""
        return len(self.info_sets)
    
    def get_info_set_counts(self) -> Dict[int, int]:
        """Get count of information sets per player."""
        return {player: len(info_sets) for player, info_sets in self.player_info_sets.items()}


def create_kuhn_info_set(player: int, card: int, history: str) -> KuhnInfoSet:
    """Convenience function to create a Kuhn poker information set."""
    return KuhnInfoSet(player, card, history)


def info_set_from_game_state(env, player: int) -> KuhnInfoSet:
    """Create information set from current game state."""
    if hasattr(env, 'player_cards') and hasattr(env, 'history'):
        # Kuhn poker environment
        card = env.player_cards[player]
        history = env.history
        return KuhnInfoSet(player, card, history)
    else:
        raise ValueError("Unsupported environment type")


if __name__ == "__main__":
    # Test the information set implementation
    print("Testing Kuhn Poker Information Sets")
    print("=" * 40)
    
    # Test basic creation
    info_set = KuhnInfoSet(0, 1, "")
    print(f"Basic info set: {info_set}")
    print(f"Legal actions: {info_set.get_legal_actions()}")
    print(f"Action 0 meaning: {info_set.get_action_meaning(0)}")
    print(f"Action 1 meaning: {info_set.get_action_meaning(1)}")
    
    # Test with betting history
    info_set2 = KuhnInfoSet(1, 3, "B")
    print(f"\nInfo set facing bet: {info_set2}")
    print(f"Action 0 meaning: {info_set2.get_action_meaning(0)}")
    print(f"Action 1 meaning: {info_set2.get_action_meaning(1)}")
    
    # Test manager
    print(f"\nTesting InfoSetManager:")
    manager = InfoSetManager()
    manager.generate_kuhn_poker_info_sets()
    
    print(f"Total info sets: {manager.size()}")
    counts = manager.get_info_set_counts()
    print(f"Player 0 info sets: {counts[0]}")
    print(f"Player 1 info sets: {counts[1]}")
    
    print(f"\nPlayer 0 info sets:")
    for info_set in manager.get_player_info_sets(0):
        print(f"  {info_set}")
    
    print(f"\nPlayer 1 info sets:")
    for info_set in manager.get_player_info_sets(1):
        print(f"  {info_set}")