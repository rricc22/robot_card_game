import random
from dataclasses import dataclass, field
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUITS = ['♠', '♥', '♦', '♣']
RANK_VALUES = {r: i + 2 for i, r in enumerate(RANKS)}  # 2→2, J→11, Q→12, K→13, A→14


@dataclass
class Card:
    rank: str
    suit: str = '?'

    @property
    def value(self) -> int:
        return RANK_VALUES[self.rank]

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"


class Deck:
    def __init__(self):
        self.cards = [Card(r, s) for s in SUITS for r in RANKS]
        random.shuffle(self.cards)

    def split(self) -> tuple[list[Card], list[Card]]:
        """Split the 52-card deck evenly between two players."""
        mid = len(self.cards) // 2
        return self.cards[:mid], self.cards[mid:]


def compare_cards(a: Card, b: Card) -> int:
    """Returns 1 if a wins, -1 if b wins, 0 on tie."""
    if a.value > b.value:
        return 1
    if b.value > a.value:
        return -1
    return 0


@dataclass
class GameState:
    robot_hand: list[Card] = field(default_factory=list)
    human_hand: list[Card] = field(default_factory=list)
    robot_score: int = 0
    human_score: int = 0
    round_number: int = 0

    @property
    def game_over(self) -> bool:
        return not self.robot_hand or not self.human_hand
