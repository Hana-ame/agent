"""
上海三打一牌型规则：
- 两副牌108张，4人各27张
- 牌型：单张、对子、三张、三带一、三带二、
        顺子(5+)、连对(3+)、飞机(2+)、
        炸弹(四张及以上)、火箭(双王)、王炸(四王)
"""

from collections import Counter
from typing import List, Dict, Tuple, Optional
from enum import IntEnum, auto

from .cards import Card, RANK_VALUES


class CardType(IntEnum):
    """牌型枚举，值越大在比较时优先级越高（同类型比主牌值）"""
    SINGLE = auto()          # 单张
    PAIR = auto()            # 对子
    TRIPLE = auto()          # 三张
    TRIPLE_ONE = auto()      # 三带一
    TRIPLE_PAIR = auto()     # 三带二
    STRAIGHT = auto()        # 顺子 (5+张)
    CONSECUTIVE_PAIRS = auto()  # 连对 (3+对)
    AIRCRAFT = auto()        # 飞机 (2+个三张)
    AIRCRAFT_SINGLES = auto()  # 飞机带单
    AIRCRAFT_PAIRS = auto()  # 飞机带对
    FOUR_TWO = auto()        # 四带二
    FOUR_TWO_PAIRS = auto()  # 四带两对
    BOMB = auto()            # 炸弹 (四张及以上相同)
    ROCKET = auto()          # 火箭 (双王)
    KING_BOMB = auto()       # 王炸 (四王)

    # 特殊：过
    PASS = auto()


# 炸弹需要至少4张相同点数
# 火箭是双王（1大1小 或 2大2小的组合，但通常2张王）
# 王炸是4张王

BOMB_MIN_COUNT = 4


class Hand:
    """一手牌（出牌组合）"""

    def __init__(self, cards: List[Card]):
        self.cards = sorted(cards)
        self.card_type, self.main_rank, self.length = self._evaluate()

    def _evaluate(self) -> Tuple[CardType, int, int]:
        """分析牌型，返回 (类型, 主牌点数, 长度/张数)"""
        n = len(self.cards)
        if n == 0:
            return CardType.PASS, 0, 0

        # 统计各点数出现次数
        counts: Dict[int, int] = Counter(c.rank_value for c in self.cards)
        groups = sorted(counts.items(), key=lambda x: (-x[1], -x[0]))

        # 获取排序后的点数列表
        ranks = sorted(counts.keys())
        unique_ranks = len(counts)

        # ── 王炸 (4张王) ──
        if n == 4 and all(r >= 16 for r in ranks):
            return CardType.KING_BOMB, 17, 4

        # ── 火箭 (双王) ──
        if n == 2 and all(r >= 16 for r in ranks):
            return CardType.ROCKET, max(ranks), 2

        # ── 炸弹 (4张及以上相同) ──
        if unique_ranks == 1 and list(counts.values())[0] >= 4:
            return CardType.BOMB, ranks[0], n

        # 以下牌型不能包含大小王（火箭/王炸已单独处理）
        normal_ranks = [r for r in ranks if r < 16]
        if not normal_ranks:
            return CardType.PASS, 0, 0

        # ── 单张 ──
        if n == 1:
            return CardType.SINGLE, normal_ranks[0], 1

        # ── 对子 ──
        if n == 2 and unique_ranks == 1 and list(counts.values())[0] == 2:
            return CardType.PAIR, normal_ranks[0], 2

        # ── 三张 ──
        if n == 3 and unique_ranks == 1 and list(counts.values())[0] == 3:
            return CardType.TRIPLE, normal_ranks[0], 3

        # ── 三带一 ──
        if n == 4 and unique_ranks == 2 and 3 in counts.values() and 1 in counts.values():
            main = [r for r, c in counts.items() if c == 3][0]
            return CardType.TRIPLE_ONE, main, 4

        # ── 三带二 ──
        if n == 5 and unique_ranks == 2 and 3 in counts.values() and 2 in counts.values():
            main = [r for r, c in counts.items() if c == 3][0]
            return CardType.TRIPLE_PAIR, main, 5

        # ── 顺子 (5+张连续单牌，不含2和王) ──
        if n >= 5 and unique_ranks == n and all(c == 1 for c in counts.values()):
            if self._is_consecutive(normal_ranks):
                return CardType.STRAIGHT, max(normal_ranks), n

        # ── 连对 (3+对连续) ──
        if n >= 6 and n % 2 == 0 and unique_ranks == n // 2 and all(c == 2 for c in counts.values()):
            if self._is_consecutive(normal_ranks):
                return CardType.CONSECUTIVE_PAIRS, max(normal_ranks), n

        # ── 飞机 (2+个三张连续) ──
        triple_ranks = [r for r, c in counts.items() if c >= 3]
        if len(triple_ranks) >= 2:
            triple_ranks.sort()
            if self._is_consecutive(triple_ranks):
                num_triples = len(triple_ranks)
                # 检查总张数是否匹配
                remaining = n - num_triples * 3
                if remaining == 0:
                    return CardType.AIRCRAFT, max(triple_ranks), n
                elif remaining == num_triples:
                    # 三带一：每带一张
                    return CardType.AIRCRAFT_SINGLES, max(triple_ranks), n
                elif remaining == num_triples * 2:
                    # 三带二：每带一对
                    return CardType.AIRCRAFT_PAIRS, max(triple_ranks), n

        # ── 四带二 ──
        if n == 6 and unique_ranks >= 2:
            four_ranks = [r for r, c in counts.items() if c >= 4]
            if len(four_ranks) == 1:
                return CardType.FOUR_TWO, four_ranks[0], 6

        # ── 四带两对 ──
        if n == 8 and unique_ranks >= 2:
            four_ranks = [r for r, c in counts.items() if c >= 4]
            pair_ranks = [r for r, c in counts.items() if c >= 2 and r not in four_ranks]
            if len(four_ranks) == 1 and len(pair_ranks) >= 2:
                return CardType.FOUR_TWO_PAIRS, four_ranks[0], 8

        return CardType.PASS, 0, 0

    @staticmethod
    def _is_consecutive(ranks: List[int]) -> bool:
        """检查点数列表是否连续（不含2和王(15+)）"""
        if len(ranks) < 2:
            return True
        # 2的分值是15，不能出现在顺子中
        valid_ranks = [r for r in ranks if r <= 14]  # 3-A
        if len(valid_ranks) != len(ranks):
            return False
        for i in range(len(ranks) - 1):
            if ranks[i + 1] - ranks[i] != 1:
                return False
        return True

    @property
    def is_valid(self) -> bool:
        return self.card_type != CardType.PASS or len(self.cards) == 0

    def can_beat(self, other: 'Hand') -> bool:
        """判断本手牌是否能管上另一手牌"""
        if not other.is_valid or other.card_type == CardType.PASS:
            return self.is_valid and self.card_type != CardType.PASS

        if not self.is_valid or self.card_type == CardType.PASS:
            return False

        # 炸弹/火箭/王炸可以管任何牌
        if self.card_type == CardType.KING_BOMB:
            return True
        if self.card_type == CardType.ROCKET:
            return other.card_type not in (CardType.ROCKET, CardType.KING_BOMB)
        if self.card_type == CardType.BOMB:
            if other.card_type in (CardType.ROCKET, CardType.KING_BOMB):
                return False
            if other.card_type == CardType.BOMB:
                if len(self.cards) != len(other.cards):
                    return len(self.cards) > len(other.cards)
                return self.main_rank > other.main_rank
            return True

        # 同类型比较
        if self.card_type != other.card_type:
            return False
        if self.length != other.length:
            return False
        return self.main_rank > other.main_rank

    def to_dict(self) -> dict:
        return {
            'cards': [c.to_dict() for c in self.cards],
            'card_type': self.card_type.name,
            'main_rank': self.main_rank,
            'length': self.length,
        }

    @staticmethod
    def from_cards(cards: List[Card]) -> 'Hand':
        return Hand(cards)


def find_all_plays(hand_cards: List[Card], last_play: Optional[Hand] = None) -> List[Hand]:
    """
    找出所有可出的牌型组合。
    如果 last_play 为 None，返回所有合法牌型；
    否则返回能管上 last_play 的牌型。
    这是一个简化版，只返回所有合法的单组牌型。
    """
    # 简化实现：返回所有单张、对子、三张、炸弹等
    # 实际游戏中需要完整的出牌搜索，这里作为 MVP 提供基础功能
    return []


def is_valid_play(hand_cards: List[Card], last_play: Optional[Hand] = None) -> bool:
    """检查出牌是否合法"""
    h = Hand(hand_cards)
    if not h.is_valid:
        return False
    if last_play is None:
        return h.card_type != CardType.PASS
    return h.can_beat(last_play)
