"""
Information Set representation for Texas Hold'em poker games.

This module provides comprehensive information set functionality for Texas Hold'em,
designed to scale from 2-player to 6-player games.
"""

from typing import List, Tuple, Optional, Dict, Any, Set
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
import hashlib


class Street(IntEnum):
    """Enumeration for poker streets/rounds."""
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


class HoldemAction(IntEnum):
    """Standard Texas Hold'em actions."""
    FOLD = 0
    CHECK_CALL = 1
    RAISE_QUARTER_POT = 2
    RAISE_HALF_POT = 3
    RAISE_FULL_POT = 4
    ALL_IN = 5


@dataclass(frozen=True)
class BettingRound:
    """Represents betting actions for a single street."""
    street: Street
    actions: Tuple[int, ...]  # Sequence of player actions
    bet_amounts: Tuple[int, ...]  # Corresponding bet amounts
    
    def __str__(self) -> str:
        """String representation of betting round."""
        if not self.actions:
            return f"S{self.street.value}:"
        
        action_chars = []
        for action, amount in zip(self.actions, self.bet_amounts):
            if action == HoldemAction.FOLD:
                action_chars.append('F')
            elif action == HoldemAction.CHECK_CALL:
                action_chars.append('C')
            elif action == HoldemAction.RAISE_QUARTER_POT:
                action_chars.append('R1')
            elif action == HoldemAction.RAISE_HALF_POT:
                action_chars.append('R2')
            elif action == HoldemAction.RAISE_FULL_POT:
                action_chars.append('R3')
            elif action == HoldemAction.ALL_IN:
                action_chars.append('A')
            else:
                action_chars.append(f'X{amount}')
        
        return f"S{self.street.value}:{''.join(action_chars)}"


class HoldemInfoSet:
    """
    Information set for Texas Hold'em poker.
    
    Comprehensive representation including:
    - Hole cards and community cards
    - Betting history per street
    - Position and stack information
    - Pot and bet context
    
    Designed to scale from 2-player to 6-player games.
    """
    
    def __init__(self, 
                 player: int,
                 hole_cards: Tuple[int, int],
                 community_cards: Tuple[int, ...],
                 street: Street,
                 betting_history: List[BettingRound],
                 position: int,
                 stack_sizes: Tuple[int, ...],
                 pot_size: int,
                 current_bet: int,
                 small_blind: int = 1,
                 big_blind: int = 2,
                 num_players: int = 2,
                 action_pointer: Optional[int] = None,
                 chance_seed: Optional[int] = None,
                 deck_state: Optional[Tuple[int, ...]] = None):
        """
        Initialize Texas Hold'em information set.
        
        Args:
            player: Player index (0-based)
            hole_cards: Player's two hole cards (0-51 encoding)
            community_cards: Board cards (0-5 cards)
            street: Current street (preflop/flop/turn/river)
            betting_history: List of betting rounds
            position: Player's position (0=button in HU, small blind in multi-way)
            stack_sizes: Chip stacks for all players
            pot_size: Current pot size
            current_bet: Amount to call
            small_blind: Small blind amount
            big_blind: Big blind amount
            num_players: Number of players in game
            action_pointer: Whose turn to act (for multi-player)
            chance_seed: Seed for deterministic future card sampling in MCCFR
            deck_state: Remaining deck for deterministic sampling (cards not yet dealt)
        """
        # Validate inputs
        self._validate_inputs(player, hole_cards, community_cards, street, 
                            betting_history, stack_sizes, num_players)
        
        self.player = player
        self.hole_cards = tuple(sorted(hole_cards))  # Canonical ordering
        self.community_cards = tuple(community_cards)
        self.street = street
        self.betting_history = betting_history.copy()
        self.position = position
        self.stack_sizes = stack_sizes
        self.pot_size = pot_size
        self.current_bet = current_bet
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.num_players = num_players
        self.action_pointer = action_pointer if action_pointer is not None else player
        
        # MCCFR deterministic sampling support
        self.chance_seed = chance_seed
        self.deck_state = deck_state
        
        # Cache for expensive computations
        self._hash = None
        self._string_repr = None
    
    def _validate_inputs(self, player: int, hole_cards: Tuple[int, int], 
                        community_cards: Tuple[int, ...], street: Street,
                        betting_history: List[BettingRound], stack_sizes: Tuple[int, ...],
                        num_players: int) -> None:
        """Validate constructor inputs."""
        if not (0 <= player < num_players):
            raise ValueError(f"Invalid player {player}. Must be 0 to {num_players-1}")
        
        if len(hole_cards) != 2:
            raise ValueError(f"Must have exactly 2 hole cards, got {len(hole_cards)}")
        
        if not all(0 <= card <= 51 for card in hole_cards):
            raise ValueError("Hole cards must be in range 0-51")
        
        if not all(0 <= card <= 51 for card in community_cards):
            raise ValueError("Community cards must be in range 0-51")
        
        expected_community_cards = {
            Street.PREFLOP: 0,
            Street.FLOP: 3,
            Street.TURN: 4,
            Street.RIVER: 5
        }
        
        if len(community_cards) != expected_community_cards[street]:
            raise ValueError(f"Street {street.name} requires {expected_community_cards[street]} "
                           f"community cards, got {len(community_cards)}")
        
        if len(stack_sizes) != num_players:
            raise ValueError(f"Need {num_players} stack sizes, got {len(stack_sizes)}")
        
        # Check for card duplicates
        all_cards = set(hole_cards) | set(community_cards)
        if len(all_cards) != len(hole_cards) + len(community_cards):
            raise ValueError("Duplicate cards detected")
    
    def to_string(self) -> str:
        """Convert information set to string representation."""
        if self._string_repr is None:
            parts = []
            
            # Player
            parts.append(f"P{self.player}")
            
            # Cards (hole cards abstracted for opponent)
            hole_str = f"H{self.hole_cards[0]},{self.hole_cards[1]}"
            if self.community_cards:
                comm_str = f"B{','.join(map(str, self.community_cards))}"
                parts.append(f"{hole_str}|{comm_str}")
            else:
                parts.append(hole_str)
            
            # Current street
            parts.append(f"St{self.street.value}")
            
            # Betting history
            if self.betting_history:
                history_str = '|'.join(str(round_) for round_ in self.betting_history)
                parts.append(f"Hist[{history_str}]")
            
            # Position and stacks (for context)
            parts.append(f"Pos{self.position}")
            parts.append(f"Stacks{','.join(map(str, self.stack_sizes))}")
            parts.append(f"Pot{self.pot_size}")
            parts.append(f"Bet{self.current_bet}")
            
            # Blinds
            parts.append(f"Blinds{self.small_blind}/{self.big_blind}")
            
            self._string_repr = '|'.join(parts)
        
        return self._string_repr
    
    def get_legal_actions(self) -> List[int]:
        """
        Get legal actions available from this information set.
        
        Returns:
            List of legal action indices
        """
        legal_actions = []
        
        # Can always fold (except when can check)
        can_check = (self.current_bet == 0)
        
        if not can_check:
            legal_actions.append(HoldemAction.FOLD)
        
        # Can always check/call
        legal_actions.append(HoldemAction.CHECK_CALL)
        
        # Can raise if have chips and not facing all-in
        player_stack = self.stack_sizes[self.player]
        
        if player_stack > self.current_bet:
            # Determine possible bet sizes based on pot size
            pot_after_call = self.pot_size + self.current_bet
            
            # Quarter pot raise
            quarter_raise = pot_after_call // 4
            if self.current_bet + quarter_raise <= player_stack:
                legal_actions.append(HoldemAction.RAISE_QUARTER_POT)
            
            # Half pot raise
            half_raise = pot_after_call // 2
            if self.current_bet + half_raise <= player_stack:
                legal_actions.append(HoldemAction.RAISE_HALF_POT)
            
            # Full pot raise
            full_raise = pot_after_call
            if self.current_bet + full_raise <= player_stack:
                legal_actions.append(HoldemAction.RAISE_FULL_POT)
            
            # All-in always available if have chips
            legal_actions.append(HoldemAction.ALL_IN)
        
        return legal_actions
    
    def get_action_meaning(self, action: int) -> str:
        """Get the meaning of an action in current context."""
        if action == HoldemAction.FOLD:
            return "Fold"
        elif action == HoldemAction.CHECK_CALL:
            return "Check" if self.current_bet == 0 else "Call"
        elif action == HoldemAction.RAISE_QUARTER_POT:
            return "Raise 1/4 pot"
        elif action == HoldemAction.RAISE_HALF_POT:
            return "Raise 1/2 pot"
        elif action == HoldemAction.RAISE_FULL_POT:
            return "Raise pot"
        elif action == HoldemAction.ALL_IN:
            return "All-in"
        else:
            raise ValueError(f"Invalid action {action}")
    
    def is_decision_point(self) -> bool:
        """Check if this information set represents a decision point."""
        # All non-terminal info sets in Hold'em are decision points
        # Terminal detection would be done at game level
        return True
    
    def get_abstract_key(self, use_minimal_abstraction: bool = True) -> str:
        """
        Get abstracted key for regret storage.
        
        Args:
            use_minimal_abstraction: If True, use only suit isomorphism.
                                   If False, use the old arbitrary bucketing.
            
        Returns:
            Abstracted key string
        """
        if use_minimal_abstraction:
            # Use only provable equivalences (suit isomorphism)
            from core.holdem_regret_storage import normalize_suit_isomorphism, get_street_specific_bucket
            normalized_cards = normalize_suit_isomorphism(self.hole_cards)
            
            parts = [
                f"P{self.player}",
                f"H{normalized_cards[0]},{normalized_cards[1]}",
                f"St{self.street.value}",
                f"Pos{self.position}",
                f"Bet{1 if self.current_bet > 0 else 0}",
                f"Act{self.action_pointer}"  # Include action pointer to prevent aliasing
            ]
            
            # Add community card abstraction for multi-street
            if self.community_cards:
                board_bucket = get_street_specific_bucket(self.hole_cards, self.community_cards, self.street)
                parts.append(f"BB{board_bucket}")
            
            # Add chance seed for deterministic sampling (if available)
            if self.chance_seed is not None:
                # Use modulo to avoid too much key explosion while maintaining determinism
                parts.append(f"CS{self.chance_seed % 1000}")
            
            return '|'.join(parts)
        
        else:
            # Legacy abstraction with arbitrary strength buckets
            parts = []
            parts.append(f"H{self.hole_cards[0]},{self.hole_cards[1]}")
            
            if self.community_cards:
                parts.append(f"B{len(self.community_cards)}")
            
            parts.append(f"St{self.street.value}")
            parts.append(f"Pos{self.position}")
            
            return '|'.join(parts)
    
    def __str__(self) -> str:
        return self.to_string()
    
    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash(self.to_string())
        return self._hash
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, HoldemInfoSet):
            return False
        return self.to_string() == other.to_string()
    
    def get_remaining_deck(self) -> List[int]:
        """
        Get remaining cards in deck for sampling future community cards.
        
        Returns:
            List of card integers not yet dealt
        """
        if self.deck_state is not None:
            return list(self.deck_state)
        
        # Calculate remaining deck from known cards
        used_cards = set(self.hole_cards) | set(self.community_cards)
        
        # In multi-player, we need to account for other players' hole cards
        # For now, assume 2-player heads-up (4 cards dealt total)
        if self.num_players == 2:
            # Assume opponent has 2 cards we don't know
            remaining_cards = [card for card in range(52) if card not in used_cards]
            # Remove 2 more cards for opponent's hole cards (unknown to us)
            if len(remaining_cards) > 2:
                # This is handled properly in the MCCFR trainer with sampling
                pass
        else:
            remaining_cards = [card for card in range(52) if card not in used_cards]
        
        return remaining_cards
    
    def can_deterministic_sample(self) -> bool:
        """Check if this info set supports deterministic sampling."""
        return self.chance_seed is not None and (
            self.deck_state is not None or 
            len(self.get_remaining_deck()) > 0
        )
    
    def create_child_info_set(self, 
                             new_community_cards: Tuple[int, ...],
                             new_betting_history: List[BettingRound],
                             new_street: Street,
                             new_pot_size: int,
                             new_current_bet: int,
                             new_action_pointer: int,
                             new_chance_seed: Optional[int] = None) -> 'HoldemInfoSet':
        """
        Create a child information set after game progression.
        
        This maintains deterministic sampling capabilities while updating game state.
        """
        return HoldemInfoSet(
            player=self.player,
            hole_cards=self.hole_cards,
            community_cards=new_community_cards,
            street=new_street,
            betting_history=new_betting_history,
            position=self.position,
            stack_sizes=self.stack_sizes,  # Updated by caller if needed
            pot_size=new_pot_size,
            current_bet=new_current_bet,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            num_players=self.num_players,
            action_pointer=new_action_pointer,
            chance_seed=new_chance_seed or self.chance_seed,
            deck_state=tuple(self.get_remaining_deck()) if self.deck_state else None
        )


class HoldemInfoSetManager:
    """
    Manager for Texas Hold'em information sets.
    
    Provides functionality to create, store, and manage information sets
    with abstraction capabilities for scalable MCCFR training.
    """
    
    def __init__(self, num_players: int = 2):
        self.num_players = num_players
        self.info_sets: Dict[str, HoldemInfoSet] = {}
        self.player_info_sets: Dict[int, Set[str]] = {i: set() for i in range(num_players)}
        self.card_abstraction: Optional[Dict] = None
    
    def set_card_abstraction(self, abstraction: Dict) -> None:
        """Set card abstraction mapping for reduced state space."""
        self.card_abstraction = abstraction
    
    def create_info_set(self, 
                       player: int,
                       hole_cards: Tuple[int, int],
                       community_cards: Tuple[int, ...],
                       street: Street,
                       betting_history: List[BettingRound],
                       position: int,
                       stack_sizes: Tuple[int, ...],
                       pot_size: int,
                       current_bet: int,
                       small_blind: int = 1,
                       big_blind: int = 2,
                       action_pointer: Optional[int] = None) -> HoldemInfoSet:
        """Create a new Texas Hold'em information set."""
        
        return HoldemInfoSet(
            player=player,
            hole_cards=hole_cards,
            community_cards=community_cards,
            street=street,
            betting_history=betting_history,
            position=position,
            stack_sizes=stack_sizes,
            pot_size=pot_size,
            current_bet=current_bet,
            small_blind=small_blind,
            big_blind=big_blind,
            num_players=self.num_players,
            action_pointer=action_pointer
        )
    
    def get_or_create_info_set(self, info_set: HoldemInfoSet) -> Tuple[HoldemInfoSet, str]:
        """
        Get existing info set or store new one.
        
        Returns:
            Tuple of (info_set, key) where key is for regret storage
        """
        # Use abstract key for storage efficiency
        key = info_set.get_abstract_key(self.card_abstraction)
        
        if key not in self.info_sets:
            self.info_sets[key] = info_set
            self.player_info_sets[info_set.player].add(key)
        
        return self.info_sets[key], key
    
    def get_info_set(self, key: str) -> Optional[HoldemInfoSet]:
        """Get information set by key."""
        return self.info_sets.get(key)
    
    def get_player_info_sets(self, player: int) -> List[HoldemInfoSet]:
        """Get all information sets for a specific player."""
        return [self.info_sets[key] for key in self.player_info_sets.get(player, set())]
    
    def size(self) -> int:
        """Get total number of information sets."""
        return len(self.info_sets)
    
    def get_info_set_counts(self) -> Dict[int, int]:
        """Get count of information sets per player."""
        return {player: len(info_sets) for player, info_sets in self.player_info_sets.items()}
    
    def clear(self) -> None:
        """Clear all stored information sets."""
        self.info_sets.clear()
        self.player_info_sets = {i: set() for i in range(self.num_players)}


def card_to_string(card: int) -> str:
    """Convert card integer (0-51) to string representation."""
    if not (0 <= card <= 51):
        raise ValueError(f"Card must be in range 0-51, got {card}")
    
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['c', 'd', 'h', 's']
    
    rank = card // 4
    suit = card % 4
    
    return f"{ranks[rank]}{suits[suit]}"


def string_to_card(card_str: str) -> int:
    """Convert string representation to card integer (0-51)."""
    if len(card_str) != 2:
        raise ValueError(f"Card string must be 2 characters, got '{card_str}'")
    
    rank_char, suit_char = card_str[0].upper(), card_str[1].lower()
    
    ranks = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 
             'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
    suits = {'c': 0, 'd': 1, 'h': 2, 's': 3}
    
    if rank_char not in ranks:
        raise ValueError(f"Invalid rank '{rank_char}'")
    if suit_char not in suits:
        raise ValueError(f"Invalid suit '{suit_char}'")
    
    return ranks[rank_char] * 4 + suits[suit_char]


if __name__ == "__main__":
    # Test the Texas Hold'em information set implementation
    print("Testing Texas Hold'em Information Sets")
    print("=" * 50)
    
    # Test basic creation - preflop heads-up
    hole_cards = (0, 13)  # 2c, 3c
    community_cards = ()
    betting_history = []
    
    info_set = HoldemInfoSet(
        player=0,
        hole_cards=hole_cards,
        community_cards=community_cards,
        street=Street.PREFLOP,
        betting_history=betting_history,
        position=0,  # Button
        stack_sizes=(100, 100),
        pot_size=3,  # SB + BB
        current_bet=2,  # BB to call
        small_blind=1,
        big_blind=2
    )
    
    print(f"Preflop info set: {info_set}")
    print(f"Legal actions: {info_set.get_legal_actions()}")
    for action in info_set.get_legal_actions():
        print(f"  Action {action}: {info_set.get_action_meaning(action)}")
    
    print(f"Abstract key: {info_set.get_abstract_key()}")
    
    # Test flop info set
    community_cards = (4, 8, 12)  # 2s, 3s, 4s
    betting_round = BettingRound(
        street=Street.PREFLOP,
        actions=(HoldemAction.CHECK_CALL, HoldemAction.CHECK_CALL),
        bet_amounts=(2, 0)
    )
    
    flop_info_set = HoldemInfoSet(
        player=1,
        hole_cards=(16, 20),  # 5c, 6c
        community_cards=community_cards,
        street=Street.FLOP,
        betting_history=[betting_round],
        position=1,  # Big blind
        stack_sizes=(98, 98),
        pot_size=4,
        current_bet=0,
        small_blind=1,
        big_blind=2
    )
    
    print(f"\nFlop info set: {flop_info_set}")
    print(f"Legal actions: {flop_info_set.get_legal_actions()}")
    print(f"Abstract key: {flop_info_set.get_abstract_key()}")
    
    # Test manager
    print(f"\nTesting HoldemInfoSetManager:")
    manager = HoldemInfoSetManager(num_players=2)
    
    stored_info, key = manager.get_or_create_info_set(info_set)
    print(f"Stored info set with key: {key}")
    print(f"Manager size: {manager.size()}")
    
    counts = manager.get_info_set_counts()
    print(f"Info set counts: {counts}")
    
    # Test card conversion
    print(f"\nTesting card conversion:")
    test_cards = [0, 13, 26, 39, 51]  # 2c, 3c, 6c, Tc, As
    for card in test_cards:
        card_str = card_to_string(card)
        back_to_int = string_to_card(card_str)
        print(f"Card {card} -> '{card_str}' -> {back_to_int}")