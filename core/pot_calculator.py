"""
Pot and side-pot calculation utilities for Texas Hold'em.

This module handles complex pot calculations for all-in situations
with uneven stacks, ensuring correctness for tournament play.
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class PotType(Enum):
    """Types of pots in poker."""
    MAIN = "main"
    SIDE = "side"


@dataclass
class Pot:
    """Represents a pot or side pot."""
    amount: int
    eligible_players: List[int]  # Player indices eligible for this pot
    pot_type: PotType
    
    def __str__(self) -> str:
        return f"{self.pot_type.value.title()} pot: ${self.amount} (players: {self.eligible_players})"


@dataclass
class PlayerContribution:
    """Tracks a player's contribution to the current pot."""
    player_id: int
    amount_contributed: int
    stack_remaining: int
    is_all_in: bool = False
    
    @property
    def total_invested(self) -> int:
        """Total chips this player has put in the pot."""
        return self.amount_contributed


class PotCalculator:
    """
    Handles pot calculations for Texas Hold'em with side pots.
    
    Critical for correctness with uneven stacks as mentioned in the scaling plan:
    "Pot & side-pot calculator for correctness for uneven stacks"
    """
    
    def __init__(self, initial_stacks: List[int], small_blind: int = 1, big_blind: int = 2):
        """
        Initialize pot calculator.
        
        Args:
            initial_stacks: Starting stack sizes for each player
            small_blind: Small blind amount
            big_blind: Big blind amount
        """
        self.num_players = len(initial_stacks)
        self.initial_stacks = initial_stacks.copy()
        self.current_stacks = initial_stacks.copy()
        self.small_blind = small_blind
        self.big_blind = big_blind
        
        # Track contributions for current betting round
        self.contributions: List[PlayerContribution] = []
        for i, stack in enumerate(initial_stacks):
            self.contributions.append(PlayerContribution(
                player_id=i,
                amount_contributed=0,
                stack_remaining=stack
            ))
        
        # Track all pots (main + side pots)
        self.pots: List[Pot] = []
        self.total_pot_size = 0
        
        # Post blinds automatically
        self._post_blinds()
    
    def _post_blinds(self) -> None:
        """Post small and big blinds."""
        if self.num_players >= 2:
            # In heads-up, button is small blind (player 0)
            sb_player = 0 if self.num_players == 2 else 1
            bb_player = 1 if self.num_players == 2 else 2
            
            # Post small blind
            sb_amount = min(self.small_blind, self.current_stacks[sb_player])
            self.add_contribution(sb_player, sb_amount)
            
            # Post big blind  
            bb_amount = min(self.big_blind, self.current_stacks[bb_player])
            self.add_contribution(bb_player, bb_amount)
    
    def add_contribution(self, player_id: int, amount: int) -> bool:
        """
        Add a player's contribution to the pot.
        
        Args:
            player_id: Player making the contribution
            amount: Chips to add to pot
            
        Returns:
            True if contribution was valid, False otherwise
        """
        if player_id < 0 or player_id >= self.num_players:
            return False
        
        contribution = self.contributions[player_id]
        
        # Can't contribute more than remaining stack
        actual_amount = min(amount, contribution.stack_remaining)
        if actual_amount <= 0:
            return False
        
        # Update contribution tracking
        contribution.amount_contributed += actual_amount
        contribution.stack_remaining -= actual_amount
        self.current_stacks[player_id] -= actual_amount
        self.total_pot_size += actual_amount
        
        # Mark as all-in if no chips remaining
        if contribution.stack_remaining == 0:
            contribution.is_all_in = True
        
        return True
    
    def get_current_bet_to_call(self) -> int:
        """Get the current bet amount that needs to be called."""
        if not self.contributions:
            return 0
        
        max_contribution = max(c.amount_contributed for c in self.contributions)
        return max_contribution
    
    def get_amount_to_call(self, player_id: int) -> int:
        """
        Get amount a specific player needs to call.
        
        Args:
            player_id: Player to check
            
        Returns:
            Amount needed to call, or 0 if already matched or folded
        """
        if player_id < 0 or player_id >= self.num_players:
            return 0
        
        current_bet = self.get_current_bet_to_call()
        player_contribution = self.contributions[player_id].amount_contributed
        
        amount_to_call = current_bet - player_contribution
        return max(0, min(amount_to_call, self.current_stacks[player_id]))
    
    def calculate_side_pots(self, active_players: List[int]) -> List[Pot]:
        """
        Calculate main pot and side pots based on all-in situations.
        
        Args:
            active_players: Players still in the hand (not folded)
            
        Returns:
            List of pots (main + side pots) with eligible players
        """
        if not active_players:
            return []
        
        pots = []
        
        # Get contributions from active players only
        active_contributions = [
            (pid, self.contributions[pid].amount_contributed) 
            for pid in active_players
            if self.contributions[pid].amount_contributed > 0
        ]
        
        if not active_contributions:
            return []
        
        # Sort by contribution amount (ascending) 
        active_contributions.sort(key=lambda x: x[1])
        
        # Get unique contribution levels
        contribution_levels = []
        seen_amounts = set()
        for pid, amount in active_contributions:
            if amount not in seen_amounts:
                contribution_levels.append(amount)
                seen_amounts.add(amount)
        contribution_levels.sort()
        
        # Calculate pots level by level
        previous_level = 0
        
        for level in contribution_levels:
            level_increment = level - previous_level
            
            if level_increment > 0:
                # Find players eligible at this level (contributed >= level)
                eligible_players = [
                    pid for pid, amount in active_contributions 
                    if amount >= level
                ]
                
                if eligible_players:
                    pot_amount = level_increment * len(eligible_players)
                    pot_type = PotType.MAIN if len(pots) == 0 else PotType.SIDE
                    
                    pot = Pot(
                        amount=pot_amount,
                        eligible_players=sorted(eligible_players),
                        pot_type=pot_type
                    )
                    pots.append(pot)
            
            previous_level = level
        
        return pots
    
    def get_total_pot_size(self) -> int:
        """Get total size of all pots."""
        return self.total_pot_size
    
    def reset_betting_round(self) -> None:
        """Reset contributions for a new betting round (keeping total pot)."""
        for contribution in self.contributions:
            contribution.amount_contributed = 0
            # Don't reset all-in status - players stay all-in until hand ends
    
    def get_player_investment(self, player_id: int) -> int:
        """Get total amount a player has invested in this hand."""
        if player_id < 0 or player_id >= self.num_players:
            return 0
        
        return self.initial_stacks[player_id] - self.current_stacks[player_id]
    
    def distribute_winnings(self, winners_by_pot: Dict[int, List[int]]) -> Dict[int, int]:
        """
        Distribute winnings from all pots to winners.
        
        Args:
            winners_by_pot: Dict mapping pot index to list of winner player IDs
            
        Returns:
            Dict mapping player ID to total winnings
        """
        winnings = {i: 0 for i in range(self.num_players)}
        
        pots = self.calculate_side_pots(list(range(self.num_players)))
        
        for pot_index, pot in enumerate(pots):
            if pot_index in winners_by_pot:
                winners = winners_by_pot[pot_index]
                if winners:
                    per_winner = pot.amount // len(winners)
                    remainder = pot.amount % len(winners)
                    
                    for i, winner in enumerate(winners):
                        winnings[winner] += per_winner
                        # Give remainder to first winners
                        if i < remainder:
                            winnings[winner] += 1
        
        return winnings
    
    def get_pot_summary(self) -> str:
        """Get a summary string of all pots."""
        active_players = [i for i in range(self.num_players) 
                         if self.contributions[i].amount_contributed > 0]
        pots = self.calculate_side_pots(active_players)
        
        if not pots:
            return "No pot"
        
        summary_lines = []
        for pot in pots:
            summary_lines.append(str(pot))
        
        total = sum(pot.amount for pot in pots)
        summary_lines.append(f"Total: ${total}")
        
        return "\n".join(summary_lines)


def calculate_showdown_winnings(pot_calculator: PotCalculator, 
                               hand_strengths: Dict[int, float],
                               active_players: List[int]) -> Dict[int, int]:
    """
    Calculate winnings at showdown based on hand strengths.
    
    Args:
        pot_calculator: Pot calculator with current pot state
        hand_strengths: Dict mapping player ID to hand strength (higher = better)
        active_players: Players still in the hand
        
    Returns:
        Dict mapping player ID to winnings
    """
    pots = pot_calculator.calculate_side_pots(active_players)
    winners_by_pot = {}
    
    # For each pot, determine winners from eligible players
    for pot_index, pot in enumerate(pots):
        eligible_players = [p for p in pot.eligible_players if p in active_players]
        
        if not eligible_players:
            continue
        
        # Find best hand strength among eligible players
        best_strength = max(hand_strengths.get(p, 0) for p in eligible_players)
        
        # All players with best strength win this pot
        winners = [p for p in eligible_players 
                  if hand_strengths.get(p, 0) == best_strength]
        
        winners_by_pot[pot_index] = winners
    
    return pot_calculator.distribute_winnings(winners_by_pot)


if __name__ == "__main__":
    # Test the pot calculator
    print("Testing Pot Calculator")
    print("=" * 50)
    
    # Test heads-up scenario
    print("Testing heads-up with equal stacks:")
    initial_stacks = [100, 100]
    calc = PotCalculator(initial_stacks, small_blind=1, big_blind=2)
    
    print(f"After blinds:")
    print(f"  Total pot: ${calc.get_total_pot_size()}")
    print(f"  Current stacks: {calc.current_stacks}")
    print(f"  Amount to call for P0: ${calc.get_amount_to_call(0)}")
    print(f"  Amount to call for P1: ${calc.get_amount_to_call(1)}")
    
    # Player 0 calls
    calc.add_contribution(0, 1)  # Call the extra $1
    print(f"\nAfter P0 calls:")
    print(f"  Total pot: ${calc.get_total_pot_size()}")
    print(f"  Current stacks: {calc.current_stacks}")
    
    # Test uneven stacks scenario
    print(f"\nTesting uneven stacks with all-in:")
    initial_stacks_uneven = [50, 100, 25]  # 3-player with short stack
    calc2 = PotCalculator(initial_stacks_uneven, small_blind=1, big_blind=2)
    
    # Player 2 (short stack) goes all-in
    calc2.add_contribution(2, 23)  # All-in with remaining 23 chips
    print(f"After P2 all-in:")
    print(f"  Total pot: ${calc2.get_total_pot_size()}")
    print(f"  Current stacks: {calc2.current_stacks}")
    
    # Calculate side pots
    active_players = [0, 1, 2]
    pots = calc2.calculate_side_pots(active_players)
    print(f"\nSide pot calculation:")
    print(calc2.get_pot_summary())
    
    # Test showdown
    hand_strengths = {0: 0.3, 1: 0.8, 2: 0.6}  # P1 has best hand
    winnings = calculate_showdown_winnings(calc2, hand_strengths, active_players)
    print(f"\nShowdown winnings: {winnings}")
    
    print("✓ Pot calculator tests completed")