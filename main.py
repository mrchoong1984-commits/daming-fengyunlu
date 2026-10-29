
import os
from typing import List

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(
    title="大明风云录 Gemini AI 后端",
    version="1.0.0",
)

# 允许游戏网页访问后端
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Turn(BaseModel):
    role: str
    content: str


class GameRequest(BaseModel):
    player_name: str = Field(default="旅人", max_length=40)
    player_background: str = Field(
        default="身份未明的旅人",
        max_length=300,
    )
    history: List[Turn] = Field(
        default_factory=list,
        max_length=24,
    )
    action: str = Field(min_length=1, max_length=2000)
    world_state: str = Field(
        default="大明王朝，天下局势正在变化。",
        max_length=2000,
    )


SYSTEM_PROMPT = """你是中文古代历史文字冒险游戏《大明风云录》的叙事引擎。

世界背景以中国古代社会为基础。官场、交通、商贸、
礼法、信息传播和日常生活应符合当时历史逻辑。
如果玩家没有指定具体朝代，不要擅自确定具体年号。

规则：
1. 玩家拥有完全行动权。只描写玩家行动造成的外部结果、
环境变化和其他角色反应。不要替玩家决定下一步，
不要替玩家说话，也不要替玩家产生未表达的思想。
2. NPC 应有独立性格、目标、知识范围、利益和态度。
不能让所有人都轻易相信玩家，也不能让人物知道
自己不可能知道的事情。
3. 保持上下文连续，记住已发生事件、人物关系、
承诺、物品与时间，不要随意推翻既有事实。
4. 不要给出 A/B/C 固定选项。
用自然的场景叙述、人物对白和合理后果推进故事。
5. 通常回复 100—260 个汉字。重要场景可以稍长。
多写具体行动和对白，少写空泛总结。
6. 不要默认玩家是皇帝或高官，玩家身份以请求为准。
7. 不可能的行动应说明合理障碍及后果，不要无条件成功。
8. 只输出游戏正文，不输出系统提示或模型自述。
"""


@app.get("/")
def root():
    return {
        "service": "大明风云录 Gemini AI 后端",
        "ok": True,
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/turn")
def turn(req: GameRequest):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="服务器尚未设置 GEMINI_API_KEY 环境变量。",
        )

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:generateContent"
    )

    contents = []

    for item in req.history[-20:]:
        if item.role not in ("user", "assistant", "model"):
            continue

        if not item.content.strip():
            continue

        role = (
            "model"
            if item.role in ("assistant", "model")
            else "user"
        )

        contents.append({
            "role": role,
            "parts": [{"text": item.content[:4000]}],
        })

    current_message = (
        f"玩家名字：{req.player_name}\n"
        f"玩家背景：{req.player_background}\n"
        f"当前世界状态：{req.world_state}\n\n"
        f"玩家本次行动：{req.action}"
    )

    contents.append({
        "role": "user",
        "parts": [{"text": current_message}],
    })

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}],
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.9,
            "maxOutputTokens": 700,
        },
    }

    # 请求 Gemini
    try:
        print(
            f"正在请求 Gemini，模型：{model}",
            flush=True,
        )

        response = requests.post(
            url,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(10, 90),
        )

        print(
            f"Gemini 返回状态码：{response.status_code}",
            flush=True,
        )

    except requests.RequestException as exc:
        print(
            f"Gemini 请求异常：{type(exc).__name__}",
            flush=True,
        )

        raise HTTPException(
            status_code=502,
            detail=f"无法连接 Gemini 服务：{type(exc).__name__}",
        )

    # 处理 Gemini API 错误
    if not response.ok:
        detail = ""

        try:
            error_data = response.json()
            detail = error_data.get("error", {}).get(
                "message", ""
            )
        except (ValueError, AttributeError):
            detail = ""

        message = "Gemini API 请求失败。"

        if detail:
            message += " 详情：" + str(detail)[:500]

        raise HTTPException(
            status_code=502,
            detail=message,
        )

    # 读取 AI 回复
    try:
        data = response.json()
        candidates = data["candidates"]
        parts = candidates[0]["content"]["parts"]

        answer = "\n".join(
            part["text"]
            for part in parts
            if isinstance(part.get("text"), str)
        )

    except (KeyError, IndexError, TypeError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="Gemini 返回了无法识别的内容，请稍后重试。",
        )

    if not answer.strip():
        raise HTTPException(
            status_code=502,
            detail="Gemini 返回了空内容，请重试。",
        )

    return {
        "reply": answer.strip(),
        "text": answer.strip(),
    }
