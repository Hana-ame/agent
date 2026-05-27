"""
扑克牌核心表示：两副牌共108张 (含大小王)
"""

from enum import IntEnum, auto
from typing import List, Tuple, Optional
import random

# ── 花色 ──
class Suit(IntEnum):
    SPADES = 0       # ♠
    HEARTS = 1       # ♥
    CLUBS = 2        # ♣
    DIAMONDS = 3     # ♦

# ── 牌面数值（用于比较大小）──
# 数值越大牌越大
RANK_VALUES = {
    '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 15,
    'SJ': 16,  # Small Joker 小王
    'BJ': 17,  # Big Joker 大王
}

RANK_NAMES = {v: k for k, v in RANK_VALUES.items()}


class Card:
    """一张扑克牌"""

    def __init__(self, rank_value: int, suit: Optional[Suit] = None):
        """
        Args:
            rank_value: 3~15 for 3~2, 16=SJ, 17=BJ
            suit: 花色，王牌为 None
        """
        self.rank_value = rank_value
        self.suit = suit

    @property
    def rank_name(self) -> str:
        return RANK_NAMES.get(self.rank_value, '?')

    @property
    def is_joker(self) -> bool:
        return self.rank_value >= 16

    @property
    def is_small_joker(self) -> bool:
        return self.rank_value == 16

    @property
    def is_big_joker(self) -> bool:
        return self.rank_value == 17

    def __repr__(self) -> str:
        suit_symbol = {0: '♠', 1: '♥', 2: '♣', 3: '♦'}
        if self.is_small_joker:
            return '🃏S'  # Small Joker
        if self.is_big_joker:
            return '🃏B'  # Big Joker
        return f"{suit_symbol.get(self.suit, '?')}{self.rank_name}"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.rank_value == other.rank_value and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank_value, self.suit))

    def __lt__(self, other):
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank_value < other.rank_value

    def to_dict(self) -> dict:
        return {
            'rank': self.rank_name,
            'rank_value': self.rank_value,
            'suit': self.suit.name if self.suit is not None else None,
            'display': str(self),
        }

    @staticmethod
    def from_dict(d: dict) -> 'Card':
        rank_value = d['rank_value']
        suit = Suit[d['suit']] if d['suit'] else None
        return Card(rank_value, suit)


# ── 两副牌共108张 ──
def create_two_decks() -> List[Card]:
    """生成两副标准54张扑克牌（含大小王）"""
    suits = [Suit.SPADES, Suit.HEARTS, Suit.CLUBS, Suit.DIAMONDS]
    ranks = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']

    deck = []
    for _ in range(2):  # 两副牌
        for rank in ranks:
            for suit in suits:
                deck.append(Card(RANK_VALUES[rank], suit))
        deck.append(Card(16, None))  # 小王
        deck.append(Card(17, None))  # 大王
    return deck


def shuffle_and_deal() -> Tuple[List[Card], List[Card], List[Card], List[Card]]:
    """
    洗牌并发牌给4位玩家，每人27张
    Returns: (player0, player1, player2, player3) 各27张
    """
    deck = create_two_decks()
    random.shuffle(deck)
    return (
        sorted(deck[0:27]),
        sorted(deck[27:54]),
        sorted(deck[54:81]),
        sorted(deck[81:108]),
    )
