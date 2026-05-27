"""
FastAPI HTTP 接口
"""

from typing import List, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .game import GameManager, PLAYER_NAMES, GamePhase
from .rules import Hand, is_valid_play

app = FastAPI(
    title="上海三打一（两副牌斗地主）模拟器",
    description="上海三打一（两副牌斗地主）后端 API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = GameManager()


# ── 请求/响应模型 ──

class NewGameResponse(BaseModel):
    game_id: str
    phase: str
    players: list
    current_player: int
    current_player_name: str
    message: str = "游戏已创建"


class BidRequest(BaseModel):
    position: int = Field(..., ge=0, le=3, description="玩家位置 0=东 1=南 2=西 3=北")
    bid: int = Field(..., ge=0, le=3, description="叫分 0=不叫, 1/2/3=叫分")


class PlayRequest(BaseModel):
    position: int = Field(..., ge=0, le=3)
    card_indices: List[int] = Field(default_factory=list, description="手牌索引列表")
    action: str = Field(default="play", description="'play' 或 'pass'")


class GameStateResponse(BaseModel):
    game_id: str
    phase: str
    players: list
    landlord_position: Optional[int] = None
    current_player: int
    current_player_name: str
    last_play: Optional[dict] = None
    last_play_position: Optional[int] = None
    last_play_player: Optional[str] = None
    pass_count: int
    turn_count: int
    highest_bid: int
    highest_bidder: Optional[int] = None
    winner: Optional[str] = None
    bid_history: list
    play_history_count: int
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str


# ── API 端点 ──

@app.get("/")
def root():
    return {
        "name": "上海三打一（两副牌斗地主）模拟器",
        "version": "1.0.0",
        "endpoints": {
            "GET  /": "本页",
            "POST /game/new": "创建新游戏",
            "GET  /game/{game_id}": "获取游戏状态",
            "GET  /game/{game_id}/player/{position}": "获取玩家视角",
            "POST /game/{game_id}/bid": "叫地主",
            "POST /game/{game_id}/play": "出牌",
            "GET  /game/{game_id}/hand/{position}": "查看手牌",
        },
    }


@app.post("/game/new", response_model=NewGameResponse)
def create_game():
    """创建并初始化新游戏（发牌完成，等待叫地主）"""
    game = manager.create_game()
    state = game._state()
    return NewGameResponse(
        game_id=state["game_id"],
        phase=state["phase"],
        players=state["players"],
        current_player=state["current_player"],
        current_player_name=state["current_player_name"],
        message="游戏已创建，请东（位置0）开始叫地主",
    )


@app.get("/game/{game_id}", response_model=GameStateResponse)
def get_game_state(game_id: str):
    """获取游戏完整状态"""
    game = manager.get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="游戏不存在")
    state = game._state()
    return GameStateResponse(**state)


@app.get("/game/{game_id}/player/{position}", response_model=GameStateResponse)
def get_player_view(game_id: str, position: int):
    """获取指定玩家视角（仅显示自己的手牌）"""
    game = manager.get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="游戏不存在")
    if position < 0 or position > 3:
        raise HTTPException(status_code=400, detail="位置无效")
    state = game.public_state(position)
    state["message"] = f"玩家 {PLAYER_NAMES[position]} 的视角"
    return GameStateResponse(**state)


@app.post("/game/{game_id}/bid")
def player_bid(game_id: str, req: BidRequest):
    """玩家叫地主"""
    game = manager.get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="游戏不存在")

    result = game.player_bid(req.position, req.bid)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # 友好的提示信息
    bid_msg = f"玩家 {PLAYER_NAMES[req.position]} 叫了 {req.bid} 分"
    if game.phase == GamePhase.PLAYING:
        bid_msg = f"{bid_msg}，叫牌结束！地主是 {PLAYER_NAMES[game.landlord_position]}（叫了 {game.highest_bid} 分）"
        bid_msg += f"，请地主开始出牌"
    elif game.phase == GamePhase.BIDDING:
        bid_msg = f"{bid_msg}，轮到 {PLAYER_NAMES[game.current_player]}"

    result["message"] = bid_msg
    return result


@app.post("/game/{game_id}/play")
def player_play(game_id: str, req: PlayRequest):
    """玩家出牌"""
    game = manager.get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="游戏不存在")

    if req.action == "pass":
        result = game.player_pass(req.position)
    else:
        result = game.player_play(req.position, req.card_indices)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # 友好的提示信息
    if req.action == "pass":
        result["message"] = f"玩家 {PLAYER_NAMES[req.position]} 过"
    elif result.get("winner"):
        result["message"] = f"玩家 {PLAYER_NAMES[req.position]} 出完所有牌！{'地主' if game.players[req.position].is_landlord else '农民'} 胜利！"
    else:
        result["message"] = f"玩家 {PLAYER_NAMES[req.position]} 出牌，轮到 {PLAYER_NAMES[game.current_player]}"

    return result


@app.get("/game/{game_id}/hand/{position}")
def get_hand(game_id: str, position: int):
    """获取指定玩家的手牌"""
    game = manager.get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="游戏不存在")
    if position < 0 or position > 3:
        raise HTTPException(status_code=400, detail="位置无效")

    hand = game.get_player_hand(position)
    return {
        "game_id": game_id,
        "position": position,
        "player_name": PLAYER_NAMES[position],
        "hand_count": len(hand),
        "hand": [c.to_dict() for c in hand],
        "is_landlord": game.players[position].is_landlord if game.players else False,
    }


@app.get("/game/{game_id}/history")
def get_history(game_id: str):
    """获取出牌历史"""
    game = manager.get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="游戏不存在")

    history = []
    for record in game.play_history:
        history.append({
            "position": record.position,
            "player_name": PLAYER_NAMES[record.position],
            "cards": [c.to_dict() for c in record.cards],
            "card_type": record.hand.card_type.name if record.hand else None,
        })
    return {
        "game_id": game_id,
        "history": history,
        "total": len(history),
    }
