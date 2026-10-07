"""Mock Cloud Run API for the ISP/IFSP generation workflow.

This service intentionally does not call OpenAI, Firestore, or Chroma. It lets the
frontend exercise the final HTTP contracts before real generation and RAG logic
are moved out of main_long.py.
"""

from __future__ import annotations

import os
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(
    title="Mumucare ISP AI Mock API",
    version="0.1.0",
    description="Mock endpoints for long-term goals, RAG search, and final generation.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:7862", "http://localhost:7862"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class LongTermGoalRequest(BaseModel):
    service_type: str = "成人"
    domain: str
    basic_info: str = ""
    summary: str = ""
    dream: str = ""
    custom_important_to: list[str] = Field(default_factory=list)
    custom_important_for: list[str] = Field(default_factory=list)


class LongTermGoalResponse(BaseModel):
    request_id: str
    important_to: list[str]
    important_for: list[str]
    mode: str = "mock"


class ReferenceSearchRequest(BaseModel):
    service_type: str = "成人"
    domain: str
    basic_info: str = ""
    selected_long_goal: str
    top_k: int = Field(default=6, ge=1, le=10)


class ReferenceCandidate(BaseModel):
    id: str
    text: str
    score: float
    kind: str


class ReferenceSearchResponse(BaseModel):
    request_id: str
    search_direction: list[str]
    short_term_references: list[ReferenceCandidate]
    strategy_references: list[ReferenceCandidate]
    mode: str = "mock"


class FinalGenerationRequest(BaseModel):
    service_type: str = "成人"
    domain: str
    basic_info: str = ""
    selected_long_goal: str
    selected_short_reference_ids: list[str] = Field(default_factory=list)
    selected_strategy_reference_ids: list[str] = Field(default_factory=list)


class GeneratedDimension(BaseModel):
    name: str
    short_term_goal: str
    strategies: list[str]


class FinalGenerationResponse(BaseModel):
    request_id: str
    dimensions: list[GeneratedDimension]
    mode: str = "mock"


REFERENCE_LIBRARY = {
    "short-1": "運用熟悉支持者共同安排日常活動，逐步維持穩定參與。",
    "short-2": "使用圖像化提醒工具辨識活動順序與休息時機。",
    "short-3": "透過示範、練習及提示褪除建立可持續的新技能。",
    "short-4": "運用個人既有偏好與成功經驗增加自主參與。",
    "strategy-1": "由家人、同儕或熟識人員提供陪同、提醒與自然回饋。",
    "strategy-2": "提供流程圖卡、計時器或檢核表協助完成活動。",
    "strategy-3": "依服務使用者表現調整提示層級並逐步減少協助。",
    "strategy-4": "調整空間動線、刺激量與物品位置以增加參與機會。",
}


def clean_lines(values: list[str]) -> list[str]:
    return [value.strip() for value in values if value and value.strip()]


def unique_first(values: list[str], limit: int) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
        if len(result) >= limit:
            break
    return result


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "Mumucare ISP AI Mock API", "docs": "/docs"}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "mode": "mock"}


@app.post("/v1/long-term-goals", response_model=LongTermGoalResponse)
def generate_long_term_goals(payload: LongTermGoalRequest) -> LongTermGoalResponse:
    supplied = any(
        value.strip()
        for value in (payload.basic_info, payload.summary, payload.dream)
    ) or payload.custom_important_to or payload.custom_important_for
    if not supplied:
        raise HTTPException(status_code=400, detail="請至少提供一項服務使用者資訊。")

    domain = payload.domain.strip() or "生活品質"
    important_to = unique_first(
        clean_lines(payload.custom_important_to)
        + [
            f"[選擇]符合個人期待的{domain}活動以[提升]生活參與",
            f"[表達]{domain}相關偏好以[增加]自主決定的機會",
            f"[參與]自己重視的{domain}安排以[維持]生活滿意度",
        ],
        5,
    )
    important_for = unique_first(
        clean_lines(payload.custom_important_for)
        + [
            f"[建立]穩定的{domain}支持安排以[維持]日常生活品質",
            f"[獲得]適切的{domain}協助以[提升]安全與持續參與",
            f"[運用]多元支持資源以[促進]{domain}目標的達成",
        ],
        5,
    )
    return LongTermGoalResponse(
        request_id=str(uuid4()),
        important_to=important_to,
        important_for=important_for,
    )


@app.post("/v1/references/search", response_model=ReferenceSearchResponse)
def search_references(payload: ReferenceSearchRequest) -> ReferenceSearchResponse:
    if not payload.selected_long_goal.strip():
        raise HTTPException(status_code=400, detail="請先選擇一項長程目標。")

    short_ids = ["short-1", "short-2", "short-3", "short-4"][: payload.top_k]
    strategy_ids = ["strategy-1", "strategy-2", "strategy-3", "strategy-4"][: payload.top_k]
    return ReferenceSearchResponse(
        request_id=str(uuid4()),
        search_direction=[
            f"{payload.domain}領域的可觀察改變",
            f"符合{payload.service_type}語境的多元支持",
            "服務使用者可實際運用的支持方法",
        ],
        short_term_references=[
            ReferenceCandidate(
                id=reference_id,
                text=REFERENCE_LIBRARY[reference_id],
                score=round(0.91 - index * 0.04, 2),
                kind="short_term",
            )
            for index, reference_id in enumerate(short_ids)
        ],
        strategy_references=[
            ReferenceCandidate(
                id=reference_id,
                text=REFERENCE_LIBRARY[reference_id],
                score=round(0.89 - index * 0.04, 2),
                kind="strategy",
            )
            for index, reference_id in enumerate(strategy_ids)
        ],
    )


@app.post("/v1/goals/generate", response_model=FinalGenerationResponse)
def generate_final_result(payload: FinalGenerationRequest) -> FinalGenerationResponse:
    if not payload.selected_long_goal.strip():
        raise HTTPException(status_code=400, detail="請先選擇一項長程目標。")

    unknown_ids = [
        reference_id
        for reference_id in (
            payload.selected_short_reference_ids
            + payload.selected_strategy_reference_ids
        )
        if reference_id not in REFERENCE_LIBRARY
    ]
    if unknown_ids:
        raise HTTPException(status_code=400, detail=f"無效的參考資料 ID：{unknown_ids}")

    domain = payload.domain.strip() or "生活品質"
    return FinalGenerationResponse(
        request_id=str(uuid4()),
        dimensions=[
            GeneratedDimension(
                name="自然支持",
                short_term_goal=(
                    f"[運用]自然支持網絡參與{domain}活動以[維持]穩定表現，"
                    "每月參與[4]次活動中能穩定完成[3]次以上。"
                ),
                strategies=[
                    "[獲得]熟悉人員共同討論活動安排以[增加]參與意願，每月[4]次安排中能表達選擇[3]次以上。",
                    "[參與]同儕共同進行的活動以[維持]互動表現，每月[4]次活動中能穩定參與[3]次以上。",
                ],
            ),
            GeneratedDimension(
                name="科技／輔具／工具",
                short_term_goal=(
                    f"[使用]視覺化工具安排{domain}活動以[提升]自主參與，"
                    "每月[4]次安排中能自主完成[3]次以上。"
                ),
                strategies=[
                    "[使用]圖像流程卡確認活動順序以[完成]日常安排，每次[4]個項目能完成[3]個以上。",
                    "[運用]計時提醒掌握開始與休息時機以[維持]參與，每月[4]次活動中能適時調整[3]次以上。",
                ],
            ),
        ],
    )


if __name__ == "__main__":
    uvicorn.run(
        "cloud_run_backend_mock:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=False,
    )
