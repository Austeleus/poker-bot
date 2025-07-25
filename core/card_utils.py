"""
Card utilities for Texas Hold'em poker.

This module provides essential card functionality including:
- Card representation and conversion (0-51 encoding)
- Hand evaluation and ranking
- Preflop equity calculation for heads-up play
- Hand strength computation

Designed for preflop heads-up poker with the foundation to scale to multi-street.
"""

from typing import List, Tuple, Dict, Optional, Any
import itertools
from dataclasses import dataclass
from enum import IntEnum


class HandRank(IntEnum):
    """Hand ranking enumeration (higher is better)."""
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8


@dataclass
class HandStrength:
    """Hand strength representation."""
    rank: HandRank
    primary_value: int  # Main ranking value (e.g., pair rank, high card)
    secondary_value: int = 0  # Secondary value (e.g., kicker)
    tertiary_value: int = 0  # Third value for tie-breaking
    
    def __lt__(self, other) -> bool:
        """Compare hand strengths."""
        if self.rank != other.rank:
            return self.rank < other.rank
        if self.primary_value != other.primary_value:
            return self.primary_value < other.primary_value
        if self.secondary_value != other.secondary_value:
            return self.secondary_value < other.secondary_value
        return self.tertiary_value < other.tertiary_value
    
    def __eq__(self, other) -> bool:
        """Check if hand strengths are equal."""
        return (self.rank == other.rank and 
                self.primary_value == other.primary_value and
                self.secondary_value == other.secondary_value and
                self.tertiary_value == other.tertiary_value)


def card_to_rank_suit(card: int) -> Tuple[int, int]:
    """
    Convert card integer (0-51) to rank and suit.
    
    Card encoding:
    - Ranks: 0=2, 1=3, ..., 11=K, 12=A
    - Suits: 0=clubs, 1=diamonds, 2=hearts, 3=spades
    - Card = rank * 4 + suit
    
    Args:
        card: Card as integer 0-51
        
    Returns:
        Tuple of (rank, suit)
    """
    if not (0 <= card <= 51):
        raise ValueError(f"Card must be 0-51, got {card}")
    
    rank = card // 4
    suit = card % 4
    return rank, suit


def rank_suit_to_card(rank: int, suit: int) -> int:
    """
    Convert rank and suit to card integer.
    
    Args:
        rank: Rank 0-12 (2-A)
        suit: Suit 0-3 (c/d/h/s)
        
    Returns:
        Card as integer 0-51
    """
    if not (0 <= rank <= 12):
        raise ValueError(f"Rank must be 0-12, got {rank}")
    if not (0 <= suit <= 3):
        raise ValueError(f"Suit must be 0-3, got {suit}")
    
    return rank * 4 + suit


def card_to_string(card: int) -> str:
    """Convert card integer to string representation."""
    rank, suit = card_to_rank_suit(card)
    
    rank_chars = '23456789TJQKA'
    suit_chars = 'cdhs'
    
    return f"{rank_chars[rank]}{suit_chars[suit]}"


def string_to_card(card_str: str) -> int:
    """Convert string representation to card integer."""
    if len(card_str) != 2:
        raise ValueError(f"Card string must be 2 characters, got '{card_str}'")
    
    rank_char, suit_char = card_str[0].upper(), card_str[1].lower()
    
    rank_map = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
                'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    suit_map = {'c': 0, 'd': 1, 'h': 2, 's': 3}
    
    if rank_char not in rank_map:
        raise ValueError(f"Invalid rank '{rank_char}'")
    if suit_char not in suit_map:
        raise ValueError(f"Invalid suit '{suit_char}'")
    
    return rank_suit_to_card(rank_map[rank_char], suit_map[suit_char])


def evaluate_hand_strength(cards: List[int]) -> HandStrength:
    """
    Evaluate hand strength from list of cards.
    
    For preflop heads-up, this handles 2-card hands.
    Designed to extend to 5+ card hands for multi-street.
    
    Args:
        cards: List of card integers (2 for preflop, up to 7 for full board)
        
    Returns:
        HandStrength object
    """
    if len(cards) < 2:
        raise ValueError("Need at least 2 cards to evaluate")
    
    # For preflop (2 cards), evaluate as high card or pair
    if len(cards) == 2:
        return _evaluate_preflop_hand(cards)
    
    # For multi-street (5+ cards), find best 5-card hand
    if len(cards) >= 5:
        return _evaluate_postflop_hand(cards)
    
    # For flop/turn (3-4 cards), evaluate what we have so far
    return _evaluate_partial_hand(cards)


def _evaluate_preflop_hand(cards: List[int]) -> HandStrength:
    """Evaluate preflop 2-card hand."""
    if len(cards) != 2:
        raise ValueError("Preflop evaluation requires exactly 2 cards")
    
    rank1, suit1 = card_to_rank_suit(cards[0])
    rank2, suit2 = card_to_rank_suit(cards[1])
    
    # Check for pair
    if rank1 == rank2:
        return HandStrength(
            rank=HandRank.PAIR,
            primary_value=rank1,  # Pair rank
            secondary_value=0,    # No kicker for preflop pairs
            tertiary_value=0
        )
    
    # High card hand - higher rank is primary, lower is secondary
    high_rank = max(rank1, rank2)
    low_rank = min(rank1, rank2)
    
    return HandStrength(
        rank=HandRank.HIGH_CARD,
        primary_value=high_rank,
        secondary_value=low_rank,
        tertiary_value=1 if suit1 == suit2 else 0  # Suited bonus
    )


def _evaluate_partial_hand(cards: List[int]) -> HandStrength:
    """Evaluate 3-4 card hand (flop/turn)."""
    # For now, just find highest pair or high card
    # This is a simplified implementation for future multi-street support
    
    ranks = [card_to_rank_suit(card)[0] for card in cards]
    rank_counts = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    # Find pairs/trips
    pairs = []
    trips = []
    for rank, count in rank_counts.items():
        if count >= 3:
            trips.append(rank)
        elif count >= 2:
            pairs.append(rank)
    
    if trips:
        return HandStrength(
            rank=HandRank.THREE_OF_A_KIND,
            primary_value=max(trips),
            secondary_value=max([r for r in ranks if r not in trips]),
            tertiary_value=0
        )
    
    if pairs:
        return HandStrength(
            rank=HandRank.PAIR,
            primary_value=max(pairs),
            secondary_value=max([r for r in ranks if r not in pairs]),
            tertiary_value=0
        )
    
    # High card
    sorted_ranks = sorted(ranks, reverse=True)
    return HandStrength(
        rank=HandRank.HIGH_CARD,
        primary_value=sorted_ranks[0],
        secondary_value=sorted_ranks[1] if len(sorted_ranks) > 1 else 0,
        tertiary_value=sorted_ranks[2] if len(sorted_ranks) > 2 else 0
    )


def _evaluate_postflop_hand(cards: List[int]) -> HandStrength:
    """Evaluate 5+ card hand (full hand evaluation)."""
    # This is a placeholder for full 5-card hand evaluation
    # For now, use simplified evaluation
    return _evaluate_partial_hand(cards[:5])


def get_preflop_hand_type(hole_cards: Tuple[int, int]) -> str:
    """
    Get preflop hand type string for heads-up play.
    
    Args:
        hole_cards: Two hole cards as integers
        
    Returns:
        Hand type string (e.g., "AA", "AKs", "AKo", "72o")
    """
    rank1, suit1 = card_to_rank_suit(hole_cards[0])
    rank2, suit2 = card_to_rank_suit(hole_cards[1])
    
    rank_chars = '23456789TJQKA'
    
    # Pair
    if rank1 == rank2:
        return f"{rank_chars[rank1]}{rank_chars[rank1]}"
    
    # Non-pair - higher rank first
    if rank1 > rank2:
        high_rank, low_rank = rank1, rank2
        high_suit, low_suit = suit1, suit2
    else:
        high_rank, low_rank = rank2, rank1
        high_suit, low_suit = suit2, suit1
    
    suited_char = 's' if suit1 == suit2 else 'o'
    return f"{rank_chars[high_rank]}{rank_chars[low_rank]}{suited_char}"


def calculate_preflop_equity(hero_cards: Tuple[int, int], 
                           villain_cards: Tuple[int, int],
                           num_simulations: int = 1000) -> float:
    """
    Calculate preflop equity between two hands in heads-up play.
    
    This runs Monte Carlo simulation to estimate equity.
    For preflop only, we assume all board runouts.
    
    Args:
        hero_cards: Hero's hole cards
        villain_cards: Villain's hole cards  
        num_simulations: Number of board simulations to run
        
    Returns:
        Hero's equity (0.0 to 1.0)
    """
    if set(hero_cards) & set(villain_cards):
        raise ValueError("Cards must be unique")
    
    # Get remaining deck
    used_cards = set(hero_cards) | set(villain_cards)
    remaining_deck = [card for card in range(52) if card not in used_cards]
    
    if len(remaining_deck) < 5:
        raise ValueError("Not enough cards remaining for simulation")
    
    wins = 0
    ties = 0
    
    # Run simulations
    import random
    for _ in range(num_simulations):
        # Sample 5 board cards
        board = random.sample(remaining_deck, 5)
        
        # Evaluate both hands
        hero_hand = list(hero_cards) + board
        villain_hand = list(villain_cards) + board
        
        hero_strength = evaluate_hand_strength(hero_hand)
        villain_strength = evaluate_hand_strength(villain_hand)
        
        if hero_strength > villain_strength:
            wins += 1
        elif hero_strength == villain_strength:
            ties += 1
    
    equity = (wins + 0.5 * ties) / num_simulations
    return equity


def get_preflop_hand_strength(hole_cards: Tuple[int, int]) -> float:
    """
    Get normalized preflop hand strength for heads-up play.
    
    This provides a quick hand strength estimate without simulation.
    Based on heads-up hand rankings.
    
    Args:
        hole_cards: Two hole cards as integers
        
    Returns:
        Hand strength from 0.0 (worst) to 1.0 (best)
    """
    rank1, suit1 = card_to_rank_suit(hole_cards[0])
    rank2, suit2 = card_to_rank_suit(hole_cards[1])
    
    # Ensure rank1 >= rank2
    if rank1 < rank2:
        rank1, rank2 = rank2, rank1
        suit1, suit2 = suit2, suit1
    
    is_suited = suit1 == suit2
    is_pair = rank1 == rank2
    
    # Base strength from high card
    strength = (rank1 + rank2) / 24.0  # Max is (12 + 12) / 24 = 1.0
    
    # Pair bonus
    if is_pair:
        pair_bonus = 0.3 + (rank1 / 12.0) * 0.4  # AA gets 0.7 bonus, 22 gets 0.3
        strength += pair_bonus
    else:
        # Suited bonus
        if is_suited:
            strength += 0.1
        
        # Connected bonus (for suited connectors)
        gap = rank1 - rank2
        if is_suited and gap <= 4:  # Suited cards within 4 ranks
            connector_bonus = (5 - gap) * 0.02  # Small bonus for connectedness
            strength += connector_bonus
    
    # Normalize to [0, 1] range
    return min(1.0, strength)


def generate_all_preflop_hands() -> List[Tuple[int, int]]:
    """
    Generate all possible preflop hands (169 unique hands in Hold'em).
    
    Returns hands in canonical form for easy iteration.
    
    Returns:
        List of all unique preflop hands
    """
    hands = []
    
    # All possible 2-card combinations from 52-card deck
    for card1, card2 in itertools.combinations(range(52), 2):
        hands.append((card1, card2))
    
    return hands


def get_hand_category(hole_cards: Tuple[int, int]) -> str:
    """
    Get hand category for strategic grouping.
    
    Categories are optimized for heads-up play.
    
    Args:
        hole_cards: Two hole cards as integers
        
    Returns:
        Hand category string
    """
    rank1, suit1 = card_to_rank_suit(hole_cards[0])
    rank2, suit2 = card_to_rank_suit(hole_cards[1])
    
    # Ensure rank1 >= rank2
    if rank1 < rank2:
        rank1, rank2 = rank2, rank1
    
    is_suited = suit1 == suit2
    is_pair = rank1 == rank2
    
    if is_pair:
        if rank1 >= 10:  # JJ+
            return "premium_pair"
        elif rank1 >= 6:  # 77-TT  
            return "medium_pair"
        else:  # 22-66
            return "small_pair"
    
    # Non-pairs
    if rank1 == 12:  # Ace
        if rank2 >= 10:  # AJ+
            return "premium_ace"
        elif rank2 >= 8:  # A9-AT
            return "medium_ace"
        else:  # A2-A8
            return "small_ace" if is_suited else "weak_ace"
    
    if rank1 >= 10:  # Face cards
        if rank2 >= 9:  # KQ, KJ, QJ
            return "premium_broadway"
        elif is_suited and rank2 >= 8:  # KTs, QTs
            return "suited_broadway"
        else:
            return "offsuit_broadway"
    
    # Suited connectors and one-gappers
    if is_suited:
        gap = rank1 - rank2
        if gap <= 1 and rank2 >= 4:  # 65s+
            return "suited_connector"
        elif gap <= 2 and rank2 >= 6:  # 86s+, 75s+
            return "suited_gap"
    
    return "trash"


# ==============================================================================
# Multi-Street Community Card Dealing
# ==============================================================================

def deal_flop_cards(used_cards: set) -> Tuple[int, int, int]:
    """
    Deal 3 flop cards that don't conflict with used cards.
    
    Args:
        used_cards: Set of card integers already in use (hole cards)
        
    Returns:
        Tuple of 3 flop card integers
        
    Raises:
        ValueError: If not enough cards available to deal flop
    """
    available_cards = [card for card in range(52) if card not in used_cards]
    
    if len(available_cards) < 3:
        raise ValueError(f"Not enough cards available to deal flop. Available: {len(available_cards)}")
    
    import random
    flop_cards = random.sample(available_cards, 3)
    return tuple(flop_cards)


def deal_turn_card(used_cards: set) -> int:
    """
    Deal 1 turn card that doesn't conflict with used cards.
    
    Args:
        used_cards: Set of card integers already in use (hole cards + flop)
        
    Returns:
        Turn card integer
        
    Raises:
        ValueError: If no cards available to deal turn
    """
    available_cards = [card for card in range(52) if card not in used_cards]
    
    if len(available_cards) < 1:
        raise ValueError("No cards available to deal turn")
    
    import random
    return random.choice(available_cards)


def deal_river_card(used_cards: set) -> int:
    """
    Deal 1 river card that doesn't conflict with used cards.
    
    Args:
        used_cards: Set of card integers already in use (hole cards + flop + turn)
        
    Returns:
        River card integer
        
    Raises:
        ValueError: If no cards available to deal river
    """
    available_cards = [card for card in range(52) if card not in used_cards]
    
    if len(available_cards) < 1:
        raise ValueError("No cards available to deal river")
    
    import random
    return random.choice(available_cards)


def get_board_texture(board: Tuple[int, ...]) -> Dict[str, Any]:
    """
    Analyze board texture for strategic abstraction.
    
    Args:
        board: Community cards (3 for flop, 4 for turn, 5 for river)
        
    Returns:
        Dict with board texture information:
        - 'flush_draw': bool, whether there's a flush draw
        - 'straight_draw': bool, whether there's a straight draw  
        - 'pair': bool, whether board is paired
        - 'trips': bool, whether board has trips
        - 'coordinated': float, how coordinated the board is (0-1)
    """
    if len(board) < 3:
        return {'flush_draw': False, 'straight_draw': False, 'pair': False, 
                'trips': False, 'coordinated': 0.0}
    
    # Extract ranks and suits
    ranks = [card_to_rank_suit(card)[0] for card in board]
    suits = [card_to_rank_suit(card)[1] for card in board]
    
    # Count suits for flush draws
    suit_counts = {}
    for suit in suits:
        suit_counts[suit] = suit_counts.get(suit, 0) + 1
    
    max_suit_count = max(suit_counts.values()) if suit_counts else 0
    flush_draw = max_suit_count >= 3
    
    # Count ranks for pairs/trips
    rank_counts = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    max_rank_count = max(rank_counts.values()) if rank_counts else 0
    pair = max_rank_count >= 2
    trips = max_rank_count >= 3
    
    # Check for straight draws (simplified)
    sorted_ranks = sorted(set(ranks))
    straight_draw = False
    if len(sorted_ranks) >= 3:
        # Check for consecutive ranks or near-consecutive
        for i in range(len(sorted_ranks) - 2):
            if sorted_ranks[i+2] - sorted_ranks[i] <= 4:  # Within 4 ranks = potential straight
                straight_draw = True
                break
    
    # Coordination score (higher = more coordinated)
    coordination = 0.0
    if flush_draw:
        coordination += 0.4
    if straight_draw:
        coordination += 0.4
    if pair:
        coordination += 0.3
    if trips:
        coordination += 0.5
    
    coordination = min(1.0, coordination)
    
    return {
        'flush_draw': flush_draw,
        'straight_draw': straight_draw,
        'pair': pair,
        'trips': trips,
        'coordinated': coordination
    }


def calculate_equity_with_board(hero_cards: Tuple[int, int], villain_cards: Tuple[int, int], 
                               board: Tuple[int, ...], num_simulations: int = 1000) -> float:
    """
    Calculate equity given known board cards.
    
    Args:
        hero_cards: Hero's hole cards
        villain_cards: Villain's hole cards  
        board: Known community cards (3+ cards)
        num_simulations: Number of Monte Carlo simulations
        
    Returns:
        Equity for hero (0.0 to 1.0)
    """
    if len(board) < 3:
        raise ValueError("Board must have at least 3 cards")
    
    # Cards already used
    used_cards = set(hero_cards + villain_cards + board)
    remaining_deck = [card for card in range(52) if card not in used_cards]
    
    if len(board) == 5:
        # River - no more cards to deal, just evaluate
        hero_hand = list(hero_cards) + list(board) 
        villain_hand = list(villain_cards) + list(board)
        
        hero_strength = evaluate_hand_strength(hero_hand)
        villain_strength = evaluate_hand_strength(villain_hand)
        
        if hero_strength > villain_strength:
            return 1.0
        elif hero_strength == villain_strength:
            return 0.5
        else:
            return 0.0
    
    # Need to deal remaining cards
    cards_needed = 5 - len(board)
    
    wins = 0
    ties = 0
    
    import random
    for _ in range(num_simulations):
        # Complete the board
        remaining_cards = random.sample(remaining_deck, cards_needed)
        full_board = board + tuple(remaining_cards)
        
        # Evaluate hands
        hero_hand = list(hero_cards) + list(full_board)
        villain_hand = list(villain_cards) + list(full_board)
        
        hero_strength = evaluate_hand_strength(hero_hand)
        villain_strength = evaluate_hand_strength(villain_hand)
        
        if hero_strength > villain_strength:
            wins += 1
        elif hero_strength == villain_strength:
            ties += 1
    
    equity = (wins + 0.5 * ties) / num_simulations
    return equity


if __name__ == "__main__":
    # Test the card utilities
    print("Testing Card Utilities")
    print("=" * 50)
    
    # Test card conversion
    print("Testing card conversion:")
    test_cards = [0, 12, 13, 25, 51]  # 2c, 4c, 2d, 7d, As
    for card in test_cards:
        card_str = card_to_string(card)
        back_to_int = string_to_card(card_str)
        rank, suit = card_to_rank_suit(card)
        print(f"  Card {card} -> '{card_str}' -> {back_to_int} (rank={rank}, suit={suit})")
    
    # Test preflop hand evaluation
    print(f"\nTesting preflop hand evaluation:")
    test_hands = [
        (48, 49),  # Ac, Ad (pocket aces)
        (44, 47),  # Kc, As (AK offsuit)
        (44, 46),  # Kc, Ah (AK suited)
        (0, 4),    # 2c, 3c (suited connector)
        (0, 1),    # 2c, 2d (pocket deuces)
    ]
    
    for hole_cards in test_hands:
        hand_str = get_preflop_hand_type(hole_cards)
        strength = get_preflop_hand_strength(hole_cards)
        category = get_hand_category(hole_cards)
        hand_strength_obj = evaluate_hand_strength(list(hole_cards))
        
        print(f"  {hole_cards} -> {hand_str}")
        print(f"    Strength: {strength:.3f}")
        print(f"    Category: {category}")
        print(f"    Hand rank: {hand_strength_obj.rank.name}")
        
    # Test equity calculation (small sample)
    print(f"\nTesting equity calculation:")
    hero_cards = (48, 49)  # Pocket aces
    villain_cards = (44, 47)  # AK offsuit
    
    try:
        equity = calculate_preflop_equity(hero_cards, villain_cards, num_simulations=100)
        print(f"  AA vs AKo: {equity:.3f} equity for AA")
    except Exception as e:
        print(f"  Equity calculation test failed: {e}")
    
    # Test hand generation
    all_hands = generate_all_preflop_hands()
    print(f"\nGenerated {len(all_hands)} unique preflop hands")
    
    print("✓ Card utilities tests completed")