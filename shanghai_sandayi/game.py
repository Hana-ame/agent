"""
上海三打一游戏状态机

游戏流程：
1. 发牌（自动）
2. 叫地主（轮流叫分，最高3分）
3. 出牌（地主先出，逆时针）
4. 结束（有人出完所有牌）
"""

import uuid
from enum import IntEnum, auto
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field

from .cards import Card, shuffle_and_deal, Suit
from .rules import Hand, CardType, is_valid_play


class GamePhase(IntEnum):
    WAITING = auto()       # 等待开始
    DEALING = auto()       # 发牌中
    BIDDING = auto()       # 叫地主
    PLAYING = auto()       # 出牌中
    FINISHED = auto()      # 结束


class PlayerPosition(IntEnum):
    EAST = 0
    SOUTH = 1
    WEST = 2
    NORTH = 3


# 玩家名称
PLAYER_NAMES = ['东', '南', '西', '北']

# 叫分选项
BID_OPTIONS = [0, 1, 2, 3]  # 0 = 不叫


@dataclass
class Player:
    """玩家"""
    position: int
    name: str
    hand: List[Card] = field(default_factory=list)
    is_landlord: bool = False
    is_human: bool = True  # 默认都是人类（API控制）


@dataclass
class PlayRecord:
    """出牌记录"""
    position: int
    cards: List[Card]
    hand: 'Hand'
    timestamp: int = 0


@dataclass
class BidRecord:
    """叫分记录"""
    position: int
    bid: int  # 0-3


class Game:
    """上海三打一游戏实例"""

    def __init__(self, game_id: Optional[str] = None):
        self.game_id = game_id or uuid.uuid4().hex[:8]
        self.phase = GamePhase.WAITING
        self.players: List[Player] = []
        self.landlord_position: Optional[int] = None
        self.current_player: int = 0  # 当前行动玩家
        self.last_play: Optional[Hand] = None
        self.last_play_position: Optional[int] = None
        self.pass_count: int = 0  # 连续过牌次数
        self.play_history: List[PlayRecord] = []
        self.bid_history: List[BidRecord] = []
        self.bid_count: int = 0
        self.highest_bid: int = 0
        self.highest_bidder: Optional[int] = None
        self.winner: Optional[str] = None  # 'landlord' or 'farmers'
        self.turn_count: int = 0

    # ── 初始化 ──

    def init_game(self):
        """初始化游戏：创建4名玩家并发牌"""
        self.players = [
            Player(position=i, name=PLAYER_NAMES[i])
            for i in range(4)
        ]
        hands = shuffle_and_deal()
        for i, hand in enumerate(hands):
            self.players[i].hand = hand
        self.phase = GamePhase.BIDDING
        self.current_player = 0  # 东先叫
        self.bid_count = 0
        self.highest_bid = 0
        self.highest_bidder = None
        return self._state()

    # ── 叫地主 ──

    def player_bid(self, position: int, bid: int) -> Dict[str, Any]:
        """
        玩家叫分
        Returns: 更新后的游戏状态 or 错误
        """
        if self.phase != GamePhase.BIDDING:
            return {'error': '当前不是叫地主阶段'}
        if position != self.current_player:
            return {'error': f'当前轮到 {PLAYER_NAMES[self.current_player]}'}
        if bid not in BID_OPTIONS:
            return {'error': f'叫分必须是 {BID_OPTIONS} 之一'}
        if bid != 0 and bid <= self.highest_bid:
            return {'error': f'叫分必须高于当前最高分 {self.highest_bid}'}

        self.bid_history.append(BidRecord(position=position, bid=bid))
        self.bid_count += 1

        if bid > self.highest_bid:
            self.highest_bid = bid
            self.highest_bidder = position

        if self.bid_count >= 4:
            # 叫牌结束
            return self._finalize_bidding()
        else:
            self.current_player = (position + 1) % 4
            return self._state()

    def _finalize_bidding(self) -> Dict[str, Any]:
        """叫牌结束，确定地主"""
        if self.highest_bidder is None or self.highest_bid == 0:
            # 所有人都没叫，重新发牌
            self.init_game()
            return self._state()

        # 确定地主
        self.landlord_position = self.highest_bidder
        self.players[self.landlord_position].is_landlord = True
        self.phase = GamePhase.PLAYING
        self.current_player = self.landlord_position
        self.last_play = None
        self.last_play_position = None
        self.pass_count = 0
        return self._state()

    # ── 出牌 ──

    def player_play(self, position: int, card_indices: List[int]) -> Dict[str, Any]:
        """
        玩家出牌
        Args:
            position: 玩家位置
            card_indices: 手牌索引列表（在 hand 列表中的位置）
        Returns: 更新后的游戏状态
        """
        if self.phase != GamePhase.PLAYING:
            return {'error': '当前不是出牌阶段'}
        if position != self.current_player:
            return {'error': f'当前轮到 {PLAYER_NAMES[self.current_player]}'}

        player = self.players[position]
        hand = player.hand

        # 验证有出牌
        if not card_indices:
            return {'error': '必须出牌（不能空出）；想过牌请使用 player_pass'}

        # 验证索引
        try:
            play_cards = [hand[i] for i in card_indices]
        except IndexError:
            return {'error': '无效的牌索引'}

        # 验证牌型
        play_hand = Hand(play_cards)
        if not play_hand.is_valid:
            return {'error': f'无效的牌型: {play_cards}'}

        # 如果是第一个出牌或必须管上
        if self.last_play_position is None:
            # 自由出牌，任何合法牌型都可以
            pass
        elif position == self.last_play_position:
            # 新一轮自由出牌
            pass
        else:
            if not play_hand.can_beat(self.last_play):
                return {'error': '管不上'}

        # 从手牌中移除
        for i in sorted(card_indices, reverse=True):
            player.hand.pop(i)

        # 记录出牌
        record = PlayRecord(
            position=position,
            cards=play_cards,
            hand=play_hand,
            timestamp=self.turn_count,
        )
        self.play_history.append(record)
        self.last_play = play_hand
        self.last_play_position = position
        self.pass_count = 0
        self.turn_count += 1

        # 检查胜利
        if len(player.hand) == 0:
            self.phase = GamePhase.FINISHED
            self.winner = 'landlord' if player.is_landlord else 'farmers'
            return self._state()

        # 下一个玩家
        self.current_player = (position + 1) % 4
        return self._state()

    def player_pass(self, position: int) -> Dict[str, Any]:
        """玩家过牌（不出）"""
        if self.phase != GamePhase.PLAYING:
            return {'error': '当前不是出牌阶段'}
        if position != self.current_player:
            return {'error': f'当前轮到 {PLAYER_NAMES[self.current_player]}'}
        if position == self.last_play_position:
            return {'error': '自己出的牌不能过，必须出牌'}

        self.pass_count += 1
        self._check_round_end()
        return self._state()

    def _check_round_end(self):
        """检查一轮是否结束（其他三家都过了）"""
        if self.pass_count >= 3:
            # 上一轮出牌者自由出牌
            self.current_player = self.last_play_position
            self.last_play = None
            self.last_play_position = None
            self.pass_count = 0

    # ── 状态输出 ──

    def _state(self) -> Dict[str, Any]:
        """返回当前游戏状态（JSON可序列化）"""
        return {
            'game_id': self.game_id,
            'phase': self.phase.name,
            'players': [
                {
                    'position': p.position,
                    'name': p.name,
                    'hand_count': len(p.hand),
                    'hand': [c.to_dict() for c in p.hand],
                    'is_landlord': p.is_landlord,
                }
                for p in self.players
            ],
            'landlord_position': self.landlord_position,
            'current_player': self.current_player,
            'current_player_name': PLAYER_NAMES[self.current_player],
            'last_play': self.last_play.to_dict() if self.last_play else None,
            'last_play_position': self.last_play_position,
            'last_play_player': PLAYER_NAMES[self.last_play_position] if self.last_play_position is not None else None,
            'pass_count': self.pass_count,
            'turn_count': self.turn_count,
            'highest_bid': self.highest_bid,
            'highest_bidder': self.highest_bidder,
            'winner': self.winner,
            'bid_history': [
                {'position': b.position, 'name': PLAYER_NAMES[b.position], 'bid': b.bid}
                for b in self.bid_history
            ],
            'play_history_count': len(self.play_history),
        }

    def public_state(self, position: int) -> Dict[str, Any]:
        """
        返回指定玩家视角的状态（隐藏其他玩家手牌）
        """
        state = self._state()
        for p in state['players']:
            if p['position'] != position:
                p['hand'] = []  # 隐藏其他玩家手牌
        return state

    def get_player_hand(self, position: int) -> List[Card]:
        """获取指定玩家的手牌"""
        if 0 <= position < len(self.players):
            return self.players[position].hand
        return []


# ── 游戏管理器 ──

class GameManager:
    """管理多个游戏实例"""

    def __init__(self):
        self.games: Dict[str, Game] = {}

    def create_game(self) -> Game:
        """创建并初始化新游戏"""
        game = Game()
        game.init_game()
        self.games[game.game_id] = game
        return game

    def get_game(self, game_id: str) -> Optional[Game]:
        return self.games.get(game_id)

    def remove_game(self, game_id: str):
        if game_id in self.games:
            del self.games[game_id]


# 全局管理器
game_manager = GameManager()
