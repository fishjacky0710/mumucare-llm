"""HTTP API for OpenAI generation and Chroma RAG retrieval.

The module is intentionally independent from Gradio. It uses the existing
ifsp_isp_ai service functions and exposes the stable /v1 contract exercised by
cloud_run_frontend_demo.py.
"""

from __future__ import annotations

import hmac
import logging
import os
import re
from typing import Literal
from uuid import uuid4

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from firestore_repository import (
    EnrollmentNotFoundError,
    FirestoreRepository,
    IspNotFoundError,
    StudentNotFoundError,
)
import ifsp_isp_ai as core


logger = logging.getLogger("mumucare.api")
API_KEY = os.getenv("MUMUCARE_API_KEY", "").strip()
cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://127.0.0.1:7862,http://localhost:7862",
    ).split(",")
    if origin.strip()
]

app = FastAPI(
    title="Mumucare ISP AI API",
    version="1.0.0",
    description="OpenAI generation and Chroma RAG endpoints for ISP/IFSP.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
)


class StudentIdentity(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    student_id: str = Field(alias="studentId", min_length=1, max_length=200, pattern=r"^[^/]+$")


class AdultIspIdentity(StudentIdentity):
    isp_id: str = Field(alias="ispId", min_length=1, max_length=200, pattern=r"^[^/]+$")
    domain: str = Field(min_length=1, max_length=50)
    service_type: Literal["成人"] = "成人"


class LongTermGoalRequest(AdultIspIdentity):
    custom_important_to: list[str] = Field(default_factory=list, max_length=10)
    custom_important_for: list[str] = Field(default_factory=list, max_length=10)


class LongTermGoalResponse(BaseModel):
    request_id: str
    important_to: list[str]
    important_for: list[str]
    mode: str = "live"


class ReferenceSearchRequest(AdultIspIdentity):
    selected_long_goal: str = Field(min_length=1, max_length=3_000)
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
    mode: str = "live"


class FinalGenerationRequest(AdultIspIdentity):
    selected_long_goal: str = Field(min_length=1, max_length=3_000)
    selected_short_reference_ids: list[str] = Field(default_factory=list, max_length=10)
    selected_strategy_reference_ids: list[str] = Field(default_factory=list, max_length=10)


class FinalGenerationResponse(BaseModel):
    request_id: str
    result: str
    mode: str = "live"


class RefineGenerationRequest(AdultIspIdentity):
    original: str = Field(min_length=1, max_length=60_000)
    refine_prompt: str = Field(min_length=1, max_length=5_000)


class RefineGenerationResponse(BaseModel):
    request_id: str
    result: str
    mode: str = "live"


class ServiceUserItem(BaseModel):
    id: str
    name: str


class ServiceUserListResponse(BaseModel):
    request_id: str
    service_users: list[ServiceUserItem]


class IspItem(BaseModel):
    id: str
    name: str


class AdultIspListResponse(BaseModel):
    request_id: str
    basic_info: str
    isps: list[IspItem]


class AdultIspContextResponse(BaseModel):
    request_id: str
    basic_info: str
    summary: str
    dream: str


DIRECT_IDENTIFIER_PATTERNS = (
    (re.compile(r"\b[A-Z][12]\d{8}\b", re.IGNORECASE), "**"),
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "**"),
    (re.compile(r"(?<!\d)09\d{2}[- ]?\d{3}[- ]?\d{3}(?!\d)"), "**"),
)


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if API_KEY and (not x_api_key or not hmac.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(status_code=401, detail="API 驗證失敗。")


def deidentify(text: str, extra_names: tuple[str, ...] = ()) -> str:
    value = core.replace_name_with_stars(text)
    names = {str(name).strip() for name in extra_names if str(name).strip()}
    for name in sorted(names, key=len, reverse=True):
        value = value.replace(name, "**")
        if len(name) >= 2:
            value = re.sub(re.escape(name[1:]), "**", value)
    for pattern, replacement in DIRECT_IDENTIFIER_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def validate_domain(service_type: str, domain: str) -> None:
    domains_by_type = {
        "成人": core.adult_domain_options,
        "兒童": core.child_domain_options,
        "社工": core.social_domain_options,
    }
    if domain not in domains_by_type[service_type]:
        raise HTTPException(
            status_code=400,
            detail=f"「{domain}」不是{service_type}可用的領域。",
        )


def service_unavailable(action: str, error: Exception) -> HTTPException:
    logger.exception("%s failed: %s", action, type(error).__name__)
    return HTTPException(
        status_code=503,
        detail=f"{action}暫時無法完成，請稍後再試。",
    )


def get_firestore_repository() -> FirestoreRepository:
    return FirestoreRepository()


def load_adult_context(payload: AdultIspIdentity):
    validate_domain(payload.service_type, payload.domain)
    try:
        context = get_firestore_repository().load_adult_isp_context(
            payload.student_id,
            payload.isp_id,
            payload.domain,
        )
    except EnrollmentNotFoundError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except (StudentNotFoundError, IspNotFoundError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise service_unavailable("Firestore 資料讀取", error) from error

    names = (context.student_name,)
    return {
        "basic_info": deidentify(context.basic_info, names),
        "summary": deidentify(context.summary, names),
        "dream": deidentify(context.dream, names),
    }


def reference_candidate(document, kind: str) -> ReferenceCandidate:
    text = deidentify(core.extract_content_only(document.page_content))
    raw_score = float(document.metadata.get("final_score", 0))
    score = max(min(raw_score / 5, 1.0), 0.0)
    return ReferenceCandidate(
        id=str(document.metadata["chroma_id"]),
        text=text,
        score=round(score, 4),
        kind=kind,
    )


def load_reference_texts(
    reference_ids: list[str],
    domain: str,
    allowed_types: set[str],
) -> list[str]:
    if not reference_ids:
        return []

    unique_ids = list(dict.fromkeys(reference_ids))
    result = core.get_chroma_collection().get(
        ids=unique_ids,
        include=["documents", "metadatas"],
    )
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    found = {
        str(document_id): (str(document), metadata or {})
        for document_id, document, metadata in zip(
            result.get("ids") or [],
            documents,
            metadatas,
        )
    }

    texts = []
    for reference_id in unique_ids:
        document_and_metadata = found.get(reference_id)
        if not document_and_metadata:
            raise HTTPException(status_code=400, detail="包含不存在的參考資料 ID。")
        document, metadata = document_and_metadata
        if metadata.get("domain") != domain or metadata.get("short") not in allowed_types:
            raise HTTPException(status_code=400, detail="參考資料與目前領域或類型不符。")
        texts.append(deidentify(core.extract_content_only(document)))
    return texts


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", str(uuid4()))


@app.exception_handler(HTTPException)
async def handle_http_error(request: Request, error: HTTPException):
    request_id = get_request_id(request)
    return JSONResponse(
        status_code=error.status_code,
        content={"detail": error.detail, "request_id": request_id},
        headers=error.headers,
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, error: RequestValidationError):
    request_id = get_request_id(request)
    logger.warning("Invalid API request [%s]", request_id)
    return JSONResponse(
        status_code=422,
        content={"detail": "請求資料格式不正確。", "request_id": request_id},
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, error: Exception):
    request_id = get_request_id(request)
    logger.exception("Unhandled API error [%s]: %s", request_id, type(error).__name__)
    return JSONResponse(
        status_code=500,
        content={"detail": "系統處理失敗，請稍後再試。", "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "Mumucare ISP AI API", "docs": "/docs"}


@app.get("/healthz")
def healthz() -> dict[str, object]:
    return {
        "status": "ok",
        "mode": "live",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "chroma_available": os.path.isdir(core.DBDIR),
        "api_key_required": bool(API_KEY),
        "model": core.OPENAI_MODEL,
    }


@app.get(
    "/v1/service-users",
    response_model=ServiceUserListResponse,
    dependencies=[Depends(verify_api_key)],
)
def list_service_users(request: Request) -> ServiceUserListResponse:
    try:
        service_users = get_firestore_repository().list_enrolled_students()
    except Exception as error:
        raise service_unavailable("服務使用者名單讀取", error) from error
    return ServiceUserListResponse(
        request_id=get_request_id(request),
        service_users=[ServiceUserItem(**item) for item in service_users],
    )


@app.post(
    "/v1/isps",
    response_model=AdultIspListResponse,
    dependencies=[Depends(verify_api_key)],
)
def list_adult_isps(
    request: Request,
    payload: StudentIdentity,
) -> AdultIspListResponse:
    try:
        basic_info, isps = get_firestore_repository().list_adult_isps(
            payload.student_id
        )
    except EnrollmentNotFoundError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except StudentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except Exception as error:
        raise service_unavailable("ISP 名單讀取", error) from error
    return AdultIspListResponse(
        request_id=get_request_id(request),
        basic_info=deidentify(basic_info),
        isps=[IspItem(**item) for item in isps],
    )


@app.post(
    "/v1/context",
    response_model=AdultIspContextResponse,
    dependencies=[Depends(verify_api_key)],
)
def get_adult_isp_context(
    request: Request,
    payload: AdultIspIdentity,
) -> AdultIspContextResponse:
    context = load_adult_context(payload)
    return AdultIspContextResponse(
        request_id=get_request_id(request),
        **context,
    )


@app.post(
    "/v1/long-term-goals",
    response_model=LongTermGoalResponse,
    dependencies=[Depends(verify_api_key)],
)
def generate_long_term_goals(
    request: Request,
    payload: LongTermGoalRequest,
) -> LongTermGoalResponse:
    context = load_adult_context(payload)
    if not any(
        value.strip()
        for value in (
            context["basic_info"],
            context["summary"],
            context["dream"],
        )
    ) and not payload.custom_important_to and not payload.custom_important_for:
        raise HTTPException(status_code=400, detail="請至少提供一項服務使用者資訊。")

    core_payload = core.LongTermGoalRequest(
        tab_name=payload.service_type,
        domain=payload.domain,
        basic_info=context["basic_info"],
        summary=context["summary"],
        dream=context["dream"],
        custom_to_goal="\n".join(deidentify(item) for item in payload.custom_important_to),
        custom_for_goal="\n".join(deidentify(item) for item in payload.custom_important_for),
    )
    try:
        result = core.api_long_term_goals(core_payload)
    except HTTPException:
        raise
    except Exception as error:
        raise service_unavailable("長程目標生成", error) from error

    return LongTermGoalResponse(
        request_id=get_request_id(request),
        important_to=[deidentify(goal) for goal in result.important_to],
        important_for=[deidentify(goal) for goal in result.important_for],
    )


@app.post(
    "/v1/references/search",
    response_model=ReferenceSearchResponse,
    dependencies=[Depends(verify_api_key)],
)
def search_references(
    request: Request,
    payload: ReferenceSearchRequest,
) -> ReferenceSearchResponse:
    context = load_adult_context(payload)
    basic_info = context["basic_info"]
    selected_goal = deidentify(payload.selected_long_goal)

    try:
        query_plan = core.ai_query_planner(
            payload.domain,
            basic_info,
            [selected_goal],
            payload.service_type,
        )
        short_type = "兒童短程目標" if payload.service_type == "兒童" else "短程目標"
        short_queries, short_documents = core.retrieve_candidates(
            payload.domain,
            short_type,
            basic_info,
            [selected_goal],
            payload.top_k,
            payload.service_type,
            query_plan,
        )
        strategy_queries: list[str] = []
        strategy_documents = []
        if payload.service_type != "兒童":
            strategy_queries, strategy_documents = core.retrieve_candidates(
                payload.domain,
                "策略",
                basic_info,
                [selected_goal],
                payload.top_k,
                payload.service_type,
                query_plan,
            )
    except HTTPException:
        raise
    except Exception as error:
        raise service_unavailable("RAG 參考資料搜尋", error) from error

    return ReferenceSearchResponse(
        request_id=get_request_id(request),
        search_direction=[
            deidentify(query)
            for query in dict.fromkeys(short_queries + strategy_queries)
        ],
        short_term_references=[
            reference_candidate(document, "short_term")
            for document in short_documents
        ],
        strategy_references=[
            reference_candidate(document, "strategy")
            for document in strategy_documents
        ],
    )


@app.post(
    "/v1/goals/generate",
    response_model=FinalGenerationResponse,
    dependencies=[Depends(verify_api_key)],
)
def generate_final_result(
    request: Request,
    payload: FinalGenerationRequest,
) -> FinalGenerationResponse:
    context = load_adult_context(payload)
    short_types = {"兒童短程目標"} if payload.service_type == "兒童" else {"短程目標"}
    try:
        short_references = load_reference_texts(
            payload.selected_short_reference_ids,
            payload.domain,
            short_types,
        )
        strategy_references = load_reference_texts(
            payload.selected_strategy_reference_ids,
            payload.domain,
            {"策略"},
        )
        selected_goal = deidentify(payload.selected_long_goal)
        structured_text = core.build_structured_text(
            context["basic_info"],
            selected_goal,
        )
        short_context = core.build_selected_context(short_references)
        strategy_context = core.build_selected_context(strategy_references)

        if short_context and strategy_context:
            messages = core.build_case2_prompt(
                payload.service_type,
                payload.domain,
                structured_text,
                short_context,
                strategy_context,
            )
        elif short_context:
            messages = core.build_case1_prompt(
                payload.service_type,
                payload.domain,
                structured_text,
                short_context,
            )
        else:
            messages = core.build_case3_prompt(
                payload.service_type,
                payload.domain,
                structured_text,
            )
        result = core.get_llm().invoke(messages).content.strip()
    except HTTPException:
        raise
    except Exception as error:
        raise service_unavailable("短程目標與策略生成", error) from error

    return FinalGenerationResponse(
        request_id=get_request_id(request),
        result=deidentify(result),
    )


@app.post(
    "/v1/goals/refine",
    response_model=RefineGenerationResponse,
    dependencies=[Depends(verify_api_key)],
)
def refine_final_result(
    request: Request,
    payload: RefineGenerationRequest,
) -> RefineGenerationResponse:
    load_adult_context(payload)
    original = deidentify(payload.original)
    refine_prompt = deidentify(payload.refine_prompt)

    try:
        intent = core.parse_refine_intent(refine_prompt)
        messages = core.prompt_templates.build_refine_answer_messages(
            original=original,
            refine_prompt=refine_prompt,
            intent=intent,
            service_type=payload.service_type,
            domain=payload.domain,
        )
        messages = core.add_service_type_rules(messages, payload.service_type)
        refined = core.get_llm().invoke(messages).content.strip()
        result, repair_prompt = core.prompt_templates.finalize_refined_answer(
            original=original,
            refine_prompt=refine_prompt,
            refined=refined,
        )
        if repair_prompt:
            repair_messages = messages + [
                ("assistant", refined),
                ("user", repair_prompt),
            ]
            refined = core.get_llm().invoke(repair_messages).content.strip()
            result, repair_prompt = core.prompt_templates.finalize_refined_answer(
                original=original,
                refine_prompt=refine_prompt,
                refined=refined,
            )
        if repair_prompt or result is None:
            raise ValueError("重新生成結果的面向或格式仍不完整")
    except HTTPException:
        raise
    except Exception as error:
        raise service_unavailable("生成結果調整", error) from error

    return RefineGenerationResponse(
        request_id=get_request_id(request),
        result=deidentify(result),
    )


if __name__ == "__main__":
    uvicorn.run(
        "cloud_run_backend:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8080")),
        reload=False,
    )
