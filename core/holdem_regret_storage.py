"""
Regret storage system for Texas Hold'em MCCFR.

This module provides regret tables and strategy computation specifically
designed for Texas Hold'em poker with card and action abstractions.
"""

from typing import Dict, List, Tuple, Optional, Set, DefaultDict
from collections import defaultdict
import numpy as np
from dataclasses import dataclass
import pickle
import os

from core.holdem_info_set import HoldemInfoSet, HoldemAction, Street
from core.card_utils import evaluate_hand_strength, get_preflop_hand_strength


@dataclass
class CardAbstraction:
    """Card abstraction for reducing hole card combinations."""
    
    # Preflop hand strength categories for HEADS-UP play
    # Different from full-ring because you only face one opponent
    HAND_BUCKETS = {
        'premium': 0,     # AA, KK, QQ, JJ, AKs, AKo
        'strong': 1,      # TT-66, AQs, AJs, ATs, KQs, AQo, AJo
        'medium': 2,      # 55-22, A9s-A2s, KJs, KTs, QJs, ATo, KQo
        'weak': 3,        # Suited connectors, one-gappers, KJo, QJo, etc.
        'trash': 4        # Dominated hands, unsuited junk
    }
    
    @classmethod
    def get_preflop_bucket(cls, hole_cards: Tuple[int, int]) -> int:
        """
        Get preflop hand strength bucket for HEADS-UP play.
        
        Note: These classifications are optimized for heads-up poker where
        you only face one opponent. Hand strengths differ significantly
        from full-ring (9-player) poker.
        
        Args:
            hole_cards: Two cards as 0-51 integers
            
        Returns:
            Bucket index (0-4): 0=premium, 1=strong, 2=medium, 3=weak, 4=trash
        """
        if len(hole_cards) != 2:
            return cls.HAND_BUCKETS['trash']
            
        card1, card2 = hole_cards
        
        # Convert to ranks and suits
        rank1, suit1 = card1 // 4, card1 % 4
        rank2, suit2 = card2 // 4, card2 % 4
        
        # Ensure rank1 >= rank2
        if rank1 < rank2:
            rank1, rank2 = rank2, rank1
            
        # Check for pairs (heads-up adjusted)
        if rank1 == rank2:
            if rank1 >= 12:  # AA
                return cls.HAND_BUCKETS['premium']
            elif rank1 >= 9:   # KK, QQ, JJ (premium in heads-up)
                return cls.HAND_BUCKETS['premium']
            elif rank1 >= 4:   # TT-66 (strong in heads-up)
                return cls.HAND_BUCKETS['strong']
            else:              # 55-22 (medium in heads-up, not weak!)
                return cls.HAND_BUCKETS['medium']
        
        # Check for suited/offsuit combinations
        is_suited = suit1 == suit2
        
        # Ace combinations
        if rank1 == 12:  # Ace high
            if rank2 == 11:  # AK
                return cls.HAND_BUCKETS['premium']
            elif rank2 == 10:  # AQ
                return cls.HAND_BUCKETS['strong'] if is_suited else cls.HAND_BUCKETS['medium']
            elif rank2 == 9:   # AJ
                return cls.HAND_BUCKETS['strong'] if is_suited else cls.HAND_BUCKETS['medium']
            elif rank2 >= 8:   # AT
                return cls.HAND_BUCKETS['medium'] if is_suited else cls.HAND_BUCKETS['weak']
            elif is_suited:    # A9s-A2s
                return cls.HAND_BUCKETS['weak']
            else:
                return cls.HAND_BUCKETS['trash']
        
        # King combinations
        if rank1 == 11:  # King high
            if rank2 == 10:  # KQ
                return cls.HAND_BUCKETS['strong'] if is_suited else cls.HAND_BUCKETS['medium']
            elif rank2 == 9:   # KJ
                return cls.HAND_BUCKETS['medium'] if is_suited else cls.HAND_BUCKETS['weak']
            elif is_suited and rank2 >= 8:  # KTs+
                return cls.HAND_BUCKETS['medium']
            # Special case for KK (already handled above in pairs section)
            else:
                return cls.HAND_BUCKETS['trash']
        
        # Queen combinations  
        if rank1 == 10:  # Queen high
            if rank2 == 9:   # QJ
                return cls.HAND_BUCKETS['medium'] if is_suited else cls.HAND_BUCKETS['weak']
            elif is_suited and rank2 >= 8:  # QTs+
                return cls.HAND_BUCKETS['medium']
            else:
                return cls.HAND_BUCKETS['trash']
        
        # Suited connectors and gaps
        if is_suited:
            gap = rank1 - rank2
            if gap <= 1 and rank2 >= 4:  # Suited connectors 65s+
                return cls.HAND_BUCKETS['medium']
            elif gap <= 2 and rank2 >= 6:  # Small gaps 86s+
                return cls.HAND_BUCKETS['weak']
        
        return cls.HAND_BUCKETS['trash']
    
    @classmethod
    def get_flop_bucket(cls, hole_cards: Tuple[int, int], community_cards: Tuple[int, int, int]) -> int:
        """
        Get flop hand strength bucket.
        
        Args:
            hole_cards: Two hole cards
            community_cards: Three flop cards
            
        Returns:
            Bucket index (0-9 for more granular flop classification)
        """
        all_cards = list(hole_cards) + list(community_cards)
        hand_strength = evaluate_hand_strength(all_cards)
        
        # Convert hand strength to bucket (0-9)
        if hand_strength.rank.value >= 6:  # Full house or better
            return 0
        elif hand_strength.rank.value >= 4:  # Straight or flush
            return 1  
        elif hand_strength.rank.value >= 3:  # Three of a kind
            return 2
        elif hand_strength.rank.value >= 2:  # Two pair
            return 3
        elif hand_strength.rank.value >= 1:  # Pair
            # Classify pair strength
            if hand_strength.primary_value >= 10:  # JJ+
                return 4
            elif hand_strength.primary_value >= 7:  # 88-TT
                return 5
            else:  # Low pair
                return 6
        else:  # High card
            # Use preflop strength as proxy
            preflop_strength = get_preflop_hand_strength(hole_cards)
            if preflop_strength >= 0.8:
                return 7  # Strong high card
            elif preflop_strength >= 0.5:
                return 8  # Medium high card
            else:
                return 9  # Weak high card
    
    @classmethod
    def get_turn_bucket(cls, hole_cards: Tuple[int, int], community_cards: Tuple[int, int, int, int]) -> int:
        """Get turn hand strength bucket (similar to flop but with 4th card)."""
        # For now, use same logic as flop - can be refined later
        return cls.get_flop_bucket(hole_cards, community_cards[:3])
    
    @classmethod
    def get_river_bucket(cls, hole_cards: Tuple[int, int], community_cards: Tuple[int, int, int, int, int]) -> int:
        """Get river hand strength bucket (final hand ranking)."""
        # For now, use same logic as flop - can be refined later
        return cls.get_flop_bucket(hole_cards, community_cards[:3])


class HoldemRegretTable:
    """
    Regret table for Texas Hold'em information sets.
    
    Stores regrets and cumulative strategies with efficient lookup
    using abstracted information set keys. Supports CFR+ for faster convergence.
    """
    
    def __init__(self, num_actions: int = 6, use_cfr_plus: bool = True):
        """
        Initialize regret table.
        
        Args:
            num_actions: Number of actions (6 for Hold'em)
            use_cfr_plus: Use CFR+ update rule for faster convergence
        """
        self.num_actions = num_actions
        self.use_cfr_plus = use_cfr_plus
        
        # Regret storage: info_set_key -> action -> regret
        self.regrets: DefaultDict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(num_actions, dtype=np.float64)
        )
        
        # Cumulative strategy storage: info_set_key -> action -> cumulative_prob
        self.cumulative_strategies: DefaultDict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(num_actions, dtype=np.float64)
        )
        
        # CFR+ specific storage
        if self.use_cfr_plus:
            # Store positive regret sums for CFR+
            self.positive_regret_sums: DefaultDict[str, np.ndarray] = defaultdict(
                lambda: np.zeros(num_actions, dtype=np.float64)
            )
        
        # Current strategy cache for faster access
        self._strategy_cache: Dict[str, np.ndarray] = {}
        self._cache_dirty: Set[str] = set()
        
        # Statistics
        self.total_iterations = 0
        self.info_set_count = 0
        
    def get_strategy(self, info_set_key: str, legal_actions: List[int]) -> np.ndarray:
        """
        Get current strategy using regret matching.
        
        Args:
            info_set_key: Abstract key for information set
            legal_actions: List of legal action indices
            
        Returns:
            Strategy array (probabilities for each action)
        """
        if info_set_key not in self._strategy_cache or info_set_key in self._cache_dirty:
            self._compute_strategy(info_set_key, legal_actions)
            self._cache_dirty.discard(info_set_key)
            
        strategy = self._strategy_cache[info_set_key].copy()
        
        # Ensure only legal actions have positive probability
        for action in range(self.num_actions):
            if action not in legal_actions:
                strategy[action] = 0.0
                
        # Renormalize
        strategy_sum = np.sum(strategy)
        if strategy_sum > 0:
            strategy /= strategy_sum
        else:
            # Uniform over legal actions if all regrets are non-positive
            legal_prob = 1.0 / len(legal_actions)
            strategy.fill(0.0)
            for action in legal_actions:
                strategy[action] = legal_prob
                
        return strategy
    
    def _compute_strategy(self, info_set_key: str, legal_actions: List[int]) -> None:
        """Compute strategy from regrets using regret matching."""
        regrets = self.regrets[info_set_key]
        strategy = np.zeros(self.num_actions, dtype=np.float64)
        
        # Positive regrets only
        positive_regrets = np.maximum(regrets, 0.0)
        regret_sum = np.sum(positive_regrets)
        
        if regret_sum > 0:
            strategy = positive_regrets / regret_sum
        else:
            # Uniform over legal actions
            if legal_actions:
                legal_prob = 1.0 / len(legal_actions)
                for action in legal_actions:
                    strategy[action] = legal_prob
                    
        self._strategy_cache[info_set_key] = strategy
        
    def update_regret(self, info_set_key: str, action: int, regret: float, iteration: int = 0) -> None:
        """
        Update regret for a specific action using CFR+ if enabled.
        
        Args:
            info_set_key: Abstract key for information set
            action: Action index
            regret: Regret value to add
            iteration: Current iteration (for CFR+ discount factor)
        """
        if self.use_cfr_plus:
            # CFR+ update: regret[t+1] = max(0, regret[t] + R[t+1])
            # But also use linear weighting for averaging
            current_regret = self.regrets[info_set_key][action]
            new_regret = max(0, current_regret + regret)
            self.regrets[info_set_key][action] = new_regret
            
            # Track positive regret sums for strategy averaging
            if new_regret > 0:
                self.positive_regret_sums[info_set_key][action] += new_regret
        else:
            # Standard CFR update
            self.regrets[info_set_key][action] += regret
            
        self._cache_dirty.add(info_set_key)
        
        # Track info set count
        if info_set_key not in self.cumulative_strategies:
            self.info_set_count += 1
            
    def update_strategy(self, info_set_key: str, strategy: np.ndarray, weight: float = 1.0) -> None:
        """
        Update cumulative strategy for average policy computation.
        
        Args:
            info_set_key: Abstract key for information set
            strategy: Current strategy probabilities
            weight: Weight for this strategy update
        """
        self.cumulative_strategies[info_set_key] += strategy * weight
        
    def get_average_strategy(self, info_set_key: str, legal_actions: List[int]) -> np.ndarray:
        """
        Get average strategy over all iterations.
        
        Args:
            info_set_key: Abstract key for information set
            legal_actions: List of legal action indices
            
        Returns:
            Average strategy array
        """
        cumulative = self.cumulative_strategies[info_set_key]
        cumulative_sum = np.sum(cumulative)
        
        if cumulative_sum > 0:
            strategy = cumulative / cumulative_sum
        else:
            # Uniform over legal actions if no cumulative strategy
            strategy = np.zeros(self.num_actions, dtype=np.float64)
            if legal_actions:
                legal_prob = 1.0 / len(legal_actions)
                for action in legal_actions:
                    strategy[action] = legal_prob
                    
        # Ensure only legal actions have positive probability
        for action in range(self.num_actions):
            if action not in legal_actions:
                strategy[action] = 0.0
                
        # Renormalize
        strategy_sum = np.sum(strategy)
        if strategy_sum > 0:
            strategy /= strategy_sum
            
        return strategy
    
    def get_regret_sum(self, info_set_key: str) -> float:
        """Get total absolute regret for an information set."""
        return np.sum(np.abs(self.regrets[info_set_key]))
    
    def get_total_regret(self) -> float:
        """Get total regret across all information sets."""
        total = 0.0
        for regret_array in self.regrets.values():
            total += np.sum(np.abs(regret_array))
        return total
    
    def clear_cache(self) -> None:
        """Clear strategy cache to force recomputation."""
        self._strategy_cache.clear()
        self._cache_dirty.clear()
        
    def save(self, filepath: str) -> None:
        """Save regret table to file."""
        data = {
            'regrets': dict(self.regrets),
            'cumulative_strategies': dict(self.cumulative_strategies),
            'total_iterations': self.total_iterations,
            'info_set_count': self.info_set_count,
            'num_actions': self.num_actions
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
            
    def load(self, filepath: str) -> None:
        """Load regret table from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.regrets = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float64))
        self.cumulative_strategies = defaultdict(lambda: np.zeros(self.num_actions, dtype=np.float64))
        
        for key, regret_array in data['regrets'].items():
            self.regrets[key] = regret_array
            
        for key, strategy_array in data['cumulative_strategies'].items():
            self.cumulative_strategies[key] = strategy_array
            
        self.total_iterations = data.get('total_iterations', 0)
        self.info_set_count = data.get('info_set_count', len(self.regrets))
        self.num_actions = data.get('num_actions', 6)
        
        self.clear_cache()
        
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the regret table."""
        return {
            'total_info_sets': len(self.regrets),
            'total_regret': self.get_total_regret(),
            'average_regret_per_infoset': self.get_total_regret() / max(len(self.regrets), 1),
            'total_iterations': self.total_iterations,
            'memory_usage_mb': self._estimate_memory_usage()
        }
        
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        regret_size = len(self.regrets) * self.num_actions * 8  # 8 bytes per float64
        strategy_size = len(self.cumulative_strategies) * self.num_actions * 8
        cache_size = len(self._strategy_cache) * self.num_actions * 8
        total_bytes = regret_size + strategy_size + cache_size
        return total_bytes / (1024 * 1024)  # Convert to MB


class HoldemStrategyProfile:
    """
    Strategy profile for Texas Hold'em players.
    
    Manages strategies for all players and provides utilities
    for strategy evaluation and comparison.
    """
    
    def __init__(self, num_players: int = 2):
        """Initialize strategy profile for multiple players."""
        self.num_players = num_players
        self.regret_tables: List[HoldemRegretTable] = []
        
        for i in range(num_players):
            self.regret_tables.append(HoldemRegretTable())
            
    def get_strategy(self, player: int, info_set: HoldemInfoSet) -> np.ndarray:
        """Get strategy for a player at an information set."""
        if not (0 <= player < self.num_players):
            raise ValueError(f"Invalid player {player}")
            
        # Use minimal abstraction (suit isomorphism only)
        abstract_key = info_set.get_abstract_key(use_minimal_abstraction=True)
        legal_actions = info_set.get_legal_actions()
        
        return self.regret_tables[player].get_strategy(abstract_key, legal_actions)
    
    def get_average_strategy(self, player: int, info_set: HoldemInfoSet) -> np.ndarray:
        """Get average strategy for a player at an information set."""
        if not (0 <= player < self.num_players):
            raise ValueError(f"Invalid player {player}")
            
        abstract_key = info_set.get_abstract_key(use_minimal_abstraction=True)
        legal_actions = info_set.get_legal_actions()
        
        return self.regret_tables[player].get_average_strategy(abstract_key, legal_actions)
    
    def update_regret(self, player: int, info_set: HoldemInfoSet, action: int, regret: float) -> None:
        """Update regret for a player's action."""
        abstract_key = info_set.get_abstract_key(use_minimal_abstraction=True)
        self.regret_tables[player].update_regret(abstract_key, action, regret)
        
    def update_strategy(self, player: int, info_set: HoldemInfoSet, strategy: np.ndarray, weight: float = 1.0) -> None:
        """Update cumulative strategy for a player."""
        abstract_key = info_set.get_abstract_key(use_minimal_abstraction=True)
        self.regret_tables[player].update_strategy(abstract_key, strategy, weight)
        
    def save(self, filepath_prefix: str) -> None:
        """Save all player strategies to files."""
        for player in range(self.num_players):
            filepath = f"{filepath_prefix}_player_{player}.pkl"
            self.regret_tables[player].save(filepath)
            
    def load(self, filepath_prefix: str) -> None:
        """Load all player strategies from files."""
        for player in range(self.num_players):
            filepath = f"{filepath_prefix}_player_{player}.pkl"
            if os.path.exists(filepath):
                self.regret_tables[player].load(filepath)
                
    def get_total_stats(self) -> Dict[str, any]:
        """Get combined statistics for all players."""
        total_info_sets = sum(len(table.regrets) for table in self.regret_tables)
        total_regret = sum(table.get_total_regret() for table in self.regret_tables)
        total_memory = sum(table._estimate_memory_usage() for table in self.regret_tables)
        
        return {
            'num_players': self.num_players,
            'total_info_sets': total_info_sets,
            'total_regret': total_regret,
            'total_memory_mb': total_memory,
            'player_stats': [table.get_stats() for table in self.regret_tables]
        }


def create_minimal_abstract_key(info_set: HoldemInfoSet) -> str:
    """
    Create abstract key with minimal abstraction (suit isomorphism only).
    
    This avoids arbitrary hand strength classifications and lets MCCFR
    learn the actual strategic value of different hands.
    
    Args:
        info_set: Texas Hold'em information set
        
    Returns:
        Abstract key string with minimal abstraction
    """
    # Apply only suit isomorphism (provably equivalent hands)
    normalized_cards = normalize_suit_isomorphism(info_set.hole_cards)
    
    # Use normalized cards instead of arbitrary strength buckets
    parts = [
        f"P{info_set.player}",
        f"H{normalized_cards[0]},{normalized_cards[1]}",  # Actual cards (suit-normalized)
        f"St{info_set.street.value}",
        f"Pos{info_set.position}",
        f"Bet{1 if info_set.current_bet > 0 else 0}"
    ]
    
    return '|'.join(parts)


def normalize_suit_isomorphism(hole_cards: Tuple[int, int]) -> Tuple[int, int]:
    """
    Normalize cards for suit isomorphism.
    
    AcKc, AdKd, AhKh, AsKs are strategically identical preflop,
    so we map them all to the same representation.
    
    Args:
        hole_cards: Two cards as 0-51 integers
        
    Returns:
        Normalized cards with suits mapped to canonical form
    """
    card1, card2 = hole_cards
    rank1, suit1 = card1 // 4, card1 % 4
    rank2, suit2 = card2 // 4, card2 % 4
    
    # Ensure rank1 >= rank2 for canonical ordering
    if rank1 < rank2:
        rank1, rank2 = rank2, rank1
        suit1, suit2 = suit2, suit1
    
    # Normalize suits: first card always gets suit 0
    # Second card gets suit 0 if same suit, suit 1 if different
    if suit1 == suit2:
        # Suited: both cards get suit 0
        norm_card1 = rank1 * 4 + 0
        norm_card2 = rank2 * 4 + 0
    else:
        # Offsuit: first card suit 0, second card suit 1  
        norm_card1 = rank1 * 4 + 0
        norm_card2 = rank2 * 4 + 1
    
    return (norm_card1, norm_card2)


def get_street_specific_bucket(hole_cards: Tuple[int, int], 
                              community_cards: Tuple[int, ...], 
                              street: Street) -> int:
    """
    Get appropriate hand bucket for the current street.
    
    This prevents the critical mistake mentioned in the scaling plan:
    "Using flop buckets on turn values corrupts regret updates"
    
    Args:
        hole_cards: Player's hole cards
        community_cards: Board cards
        street: Current street
        
    Returns:
        Street-specific bucket index
    """
    if street == Street.PREFLOP:
        return CardAbstraction.get_preflop_bucket(hole_cards)
    elif street == Street.FLOP and len(community_cards) >= 3:
        return CardAbstraction.get_flop_bucket(hole_cards, community_cards[:3])
    elif street == Street.TURN and len(community_cards) >= 4:
        return CardAbstraction.get_turn_bucket(hole_cards, community_cards[:4])
    elif street == Street.RIVER and len(community_cards) >= 5:
        return CardAbstraction.get_river_bucket(hole_cards, community_cards[:5])
    else:
        # Fallback to preflop bucket
        return CardAbstraction.get_preflop_bucket(hole_cards)


if __name__ == "__main__":
    # Test the regret storage system
    print("Testing Texas Hold'em Regret Storage System")
    print("=" * 50)
    
    # Test card abstraction
    print("Testing card abstraction:")
    test_hands = [
        (48, 49),  # Ac, Ad - should be premium
        (44, 47),  # Kc, As - should be premium  
        (0, 4),    # 2c, 3c - should be trash
        (40, 41),  # Jc, Jd - should be strong
        (32, 36)   # 9c, Td - should be weak/trash
    ]
    
    for hand in test_hands:
        bucket = CardAbstraction.get_preflop_bucket(hand)
        bucket_name = [k for k, v in CardAbstraction.HAND_BUCKETS.items() if v == bucket][0]
        print(f"  {hand} -> bucket {bucket} ({bucket_name})")
    
    # Test regret table
    print(f"\nTesting regret table:")
    table = HoldemRegretTable()
    
    # Simulate some regret updates
    info_set_key = "P0|B0|St0|Pos0|Bet1"
    legal_actions = [0, 1, 2, 5]  # fold, call, raise quarter, all-in
    
    # Update some regrets
    table.update_regret(info_set_key, 1, 10.0)  # Call has positive regret
    table.update_regret(info_set_key, 0, -5.0)  # Fold has negative regret
    table.update_regret(info_set_key, 2, 3.0)   # Raise has small positive regret
    
    # Get strategy
    strategy = table.get_strategy(info_set_key, legal_actions)
    print(f"  Strategy after regret updates: {strategy}")
    
    # Update cumulative strategy
    table.update_strategy(info_set_key, strategy, 1.0)
    
    # Get average strategy
    avg_strategy = table.get_average_strategy(info_set_key, legal_actions)
    print(f"  Average strategy: {avg_strategy}")
    
    # Test strategy profile
    print(f"\nTesting strategy profile:")
    profile = HoldemStrategyProfile(num_players=2)
    
    # Create mock info set
    from core.holdem_info_set import HoldemInfoSet, Street
    mock_info_set = HoldemInfoSet(
        player=0,
        hole_cards=(48, 49),  # Pocket aces
        community_cards=(),
        street=Street.PREFLOP,
        betting_history=[],
        position=0,
        stack_sizes=(200, 200),
        pot_size=3,
        current_bet=2
    )
    
    # Test getting strategy from profile
    strategy = profile.get_strategy(0, mock_info_set)
    print(f"  Player 0 strategy: {strategy}")
    
    # Get stats
    stats = profile.get_total_stats()
    print(f"  Total stats: {stats}")
    
    print("✓ Regret storage system tests passed")