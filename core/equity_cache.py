"""
Show-down equity cache for fast hand evaluation.

This module provides precomputed equity lookups for common poker situations,
essential for the ≥3 billion evaluations/hour performance requirement.
"""

from typing import Dict, Tuple, List, Optional, Set
import pickle
import os
import hashlib
from dataclasses import dataclass
from threading import Lock
import numpy as np

from core.card_utils import (
    calculate_preflop_equity, get_preflop_hand_type, 
    generate_all_preflop_hands, card_to_string
)


@dataclass(frozen=True)
class HandMatchup:
    """Represents a heads-up hand matchup for equity caching."""
    hand1: Tuple[int, int]  # First hand (sorted)
    hand2: Tuple[int, int]  # Second hand (sorted)
    
    def __post_init__(self):
        """Ensure hands are in canonical order."""
        # Sort individual hands
        if self.hand1[0] > self.hand1[1]:
            object.__setattr__(self, 'hand1', (self.hand1[1], self.hand1[0]))
        if self.hand2[0] > self.hand2[1]:
            object.__setattr__(self, 'hand2', (self.hand2[1], self.hand2[0]))
        
        # Ensure hand1 <= hand2 lexicographically
        if self.hand1 > self.hand2:
            object.__setattr__(self, 'hand1', self.hand2)
            object.__setattr__(self, 'hand2', self.hand1)
    
    def __str__(self) -> str:
        h1_str = f"{card_to_string(self.hand1[0])}{card_to_string(self.hand1[1])}"
        h2_str = f"{card_to_string(self.hand2[0])}{card_to_string(self.hand2[1])}"
        return f"{h1_str} vs {h2_str}"


@dataclass
class EquityResult:
    """Result of equity calculation."""
    hand1_equity: float  # Equity for hand1 (0.0 to 1.0)
    hand2_equity: float  # Equity for hand2 (0.0 to 1.0)
    tie_probability: float  # Probability of tie
    
    @property
    def hand1_win_probability(self) -> float:
        """Win probability for hand1 (excluding ties)."""
        return self.hand1_equity - (self.tie_probability * 0.5)
    
    @property
    def hand2_win_probability(self) -> float:
        """Win probability for hand2 (excluding ties)."""
        return self.hand2_equity - (self.tie_probability * 0.5)


class EquityCache:
    """
    High-performance equity cache for poker hand evaluations.
    
    Provides fast lookup of precomputed equities to achieve the
    ≥3 billion evaluations/hour performance target.
    """
    
    def __init__(self, cache_file: Optional[str] = None):
        """
        Initialize equity cache.
        
        Args:
            cache_file: Path to cache file for persistent storage
        """
        self.cache_file = cache_file or "data/equity_cache.pkl"
        self._cache: Dict[HandMatchup, EquityResult] = {}
        self._lock = Lock()
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_lookups = 0
        
        # Load existing cache if available
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cache from disk if available."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
                print(f"Loaded {len(self._cache)} equity entries from cache")
            except Exception as e:
                print(f"Failed to load cache: {e}")
                self._cache = {}
        else:
            print("No existing cache found, will build new one")
    
    def save_cache(self) -> None:
        """Save cache to disk."""
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        
        with self._lock:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved {len(self._cache)} equity entries to cache")
            except Exception as e:
                print(f"Failed to save cache: {e}")
    
    def get_equity(self, hand1: Tuple[int, int], hand2: Tuple[int, int]) -> EquityResult:
        """
        Get equity for a hand matchup.
        
        Args:
            hand1: First hand as (card1, card2)
            hand2: Second hand as (card1, card2)
            
        Returns:
            EquityResult with hand1's equity
        """
        self.total_lookups += 1
        
        # Create canonical matchup
        matchup = HandMatchup(hand1, hand2)
        
        with self._lock:
            if matchup in self._cache:
                self.cache_hits += 1
                return self._cache[matchup]
        
        # Cache miss - compute equity
        self.cache_misses += 1
        equity = self._compute_equity(hand1, hand2)
        
        # Store in cache
        with self._lock:
            self._cache[matchup] = equity
        
        return equity
    
    def _compute_equity(self, hand1: Tuple[int, int], hand2: Tuple[int, int], 
                       num_simulations: int = 10000) -> EquityResult:
        """Compute equity using Monte Carlo simulation."""
        try:
            hand1_equity = calculate_preflop_equity(hand1, hand2, num_simulations)
            hand2_equity = 1.0 - hand1_equity
            
            # For now, assume tie probability is small (can be refined)
            tie_prob = 0.02  # Approximate tie rate in Hold'em
            
            return EquityResult(
                hand1_equity=hand1_equity,
                hand2_equity=hand2_equity,
                tie_probability=tie_prob
            )
        except Exception as e:
            print(f"Error computing equity for {hand1} vs {hand2}: {e}")
            # Return 50/50 as fallback
            return EquityResult(
                hand1_equity=0.5,
                hand2_equity=0.5,
                tie_probability=0.0
            )
    
    def precompute_all_preflop_equities(self, num_simulations: int = 50000, 
                                       save_interval: int = 1000) -> None:
        """
        Precompute all possible preflop hand matchups.
        
        This is computationally expensive but provides massive speedup
        for training by eliminating Monte Carlo simulation during MCCFR.
        
        Args:
            num_simulations: Simulations per matchup (higher = more accurate)
            save_interval: Save cache every N computations
        """
        print("Precomputing all preflop equities...")
        
        # Get all unique preflop hands (169 canonical hands)
        all_hands = []
        seen_hand_types = set()
        
        for hand in generate_all_preflop_hands():
            hand_type = get_preflop_hand_type(hand)
            if hand_type not in seen_hand_types:
                all_hands.append(hand)
                seen_hand_types.add(hand_type)
        
        print(f"Found {len(all_hands)} unique preflop hand types")
        
        # Compute all pairwise matchups
        total_matchups = len(all_hands) * (len(all_hands) - 1) // 2
        computed = 0
        
        for i in range(len(all_hands)):
            for j in range(i + 1, len(all_hands)):
                hand1, hand2 = all_hands[i], all_hands[j]
                
                # Check if already cached
                matchup = HandMatchup(hand1, hand2)
                if matchup not in self._cache:
                    equity = self._compute_equity(hand1, hand2, num_simulations)
                    
                    with self._lock:
                        self._cache[matchup] = equity
                
                computed += 1
                
                if computed % save_interval == 0:
                    self.save_cache()
                    hit_rate = self.cache_hits / max(self.total_lookups, 1) * 100
                    print(f"Computed {computed}/{total_matchups} matchups "
                          f"(Cache: {len(self._cache)} entries, {hit_rate:.1f}% hit rate)")
        
        print(f"Precomputation complete! {len(self._cache)} total entries")
        self.save_cache()
    
    def get_fast_preflop_equity(self, hand1: Tuple[int, int], hand2: Tuple[int, int]) -> float:
        """
        Fast preflop equity lookup optimized for MCCFR training.
        
        Returns only hand1's equity as a float for maximum performance.
        
        Args:
            hand1: First hand
            hand2: Second hand
            
        Returns:
            Hand1's equity (0.0 to 1.0)
        """
        result = self.get_equity(hand1, hand2)
        return result.hand1_equity
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache performance statistics."""
        hit_rate = self.cache_hits / max(self.total_lookups, 1) * 100
        
        return {
            'total_entries': len(self._cache),
            'total_lookups': self.total_lookups,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': hit_rate,
            'memory_usage_mb': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Estimate cache memory usage in MB."""
        # Rough estimate: each entry is ~100 bytes
        return len(self._cache) * 100 / (1024 * 1024)
    
    def clear_cache(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
            self.total_lookups = 0


# Global equity cache instance
_global_equity_cache: Optional[EquityCache] = None
_cache_lock = Lock()


def get_global_equity_cache() -> EquityCache:
    """Get or create the global equity cache instance."""
    global _global_equity_cache
    
    if _global_equity_cache is None:
        with _cache_lock:
            if _global_equity_cache is None:
                _global_equity_cache = EquityCache()
    
    return _global_equity_cache


def fast_preflop_equity(hand1: Tuple[int, int], hand2: Tuple[int, int]) -> float:
    """
    Ultra-fast preflop equity lookup using global cache.
    
    This is the function to use in MCCFR training for maximum performance.
    
    Args:
        hand1: First hand
        hand2: Second hand
        
    Returns:
        Hand1's equity (0.0 to 1.0)
    """
    cache = get_global_equity_cache()
    return cache.get_fast_preflop_equity(hand1, hand2)


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Test the equity cache
    print("Testing Equity Cache")
    print("=" * 50)
    
    # Create test cache (don't use global for testing)
    cache = EquityCache("test_equity_cache.pkl")
    
    # Test some hand matchups
    print("Testing equity calculations:")
    test_hands = [
        ((48, 49), (44, 47)),  # AA vs KK  
        ((48, 44), (0, 4)),    # AK vs 23
        ((32, 33), (16, 20)),  # 99 vs 55
    ]
    
    for hand1, hand2 in test_hands:
        equity_result = cache.get_equity(hand1, hand2)
        print(f"  {HandMatchup(hand1, hand2)}: "
              f"{equity_result.hand1_equity:.3f} vs {equity_result.hand2_equity:.3f}")
    
    # Test cache performance
    print(f"\nCache performance:")
    stats = cache.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test fast lookup
    print(f"\nTesting fast lookup:")
    fast_equity = cache.get_fast_preflop_equity((48, 49), (44, 47))
    print(f"  AA vs KK equity: {fast_equity:.3f}")
    
    # Save cache
    cache.save_cache()
    
    # Clean up test file
    if os.path.exists("test_equity_cache.pkl"):
        os.remove("test_equity_cache.pkl")
    
    print("✓ Equity cache tests completed")