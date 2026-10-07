import json
import os
import re
from functools import lru_cache
from typing import Any, Literal

import chromadb
import pandas as pd
import prompt as prompt_templates
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder, SentenceTransformer


load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DBDIR = os.getenv("CHROMA_DB_DIR", os.path.join(BASE_DIR, "chroma_db_115"))
COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_knowledge")
STUDENT_FILE = os.getenv("STUDENT_FILE", os.path.join(BASE_DIR, "學生名單.xlsx"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual")

adult_domain_options = ["身體福祉", "情緒福祉", "物質福祉", "個人發展", "自我決策", "人際關係", "權利", "社會融合"]
child_domain_options = ["健康與安全", "感官知覺", "精細動作", "粗大動作", "語言溝通", "認知", "生活自理", "社會適應"]
social_domain_options = ["醫療復健輔具", "教育安置", "經濟功能及福利輔助", "親職支持", "家庭支持系統(資援連結)"]

ICF_DISABILITY_TYPE_MAP = {
    "1": ["第一類", "第1類", "1類", "01類", "神經系統", "精神", "心智功能", "智能障礙", "自閉症", "自閉", "ASD", "精神障礙", "失智症"],
    "2": ["第二類", "第2類", "2類", "02類", "眼", "耳", "感官功能", "疼痛", "視覺障礙", "視障", "聽覺障礙", "聽障"],
    "3": ["第三類", "第3類", "3類", "03類", "聲音", "言語", "語言", "語言障礙", "語障"],
    "4": ["第四類", "第4類", "4類", "04類", "循環", "造血", "免疫", "呼吸"],
    "5": ["第五類", "第5類", "5類", "05類", "消化", "新陳代謝", "內分泌"],
    "6": ["第六類", "第6類", "6類", "06類", "泌尿", "生殖"],
    "7": ["第七類", "第7類", "7類", "07類", "神經肌肉骨骼", "神經、肌肉、骨骼", "移動相關", "肢體障礙", "肢障", "腦性麻痺", "腦麻"],
    "8": ["第八類", "第8類", "8類", "08類", "皮膚", "相關構造"],
}

LONG_TERM_SPLIT_PROMPT = prompt_templates.LONG_TERM_SPLIT_PROMPT
LONG_TERM_REWRITE_PROMPT = prompt_templates.LONG_TERM_REWRITE_PROMPT

SERVICE_TYPE_RULES = """
【服務對象類型強制規則】
目前服務對象類型是「{service_type}」。
所有短程目標、支持策略與搜尋方向都必須符合此年齡與角色語境。
若為成人，不得產生童話安撫、卡通獎勵、貼紙獎勵、故事哄睡、幼兒遊戲、乖寶寶或寶寶語氣等兒童化內容。
若為兒童，可使用兒童可理解的圖卡、遊戲化練習、故事情境與家庭支持，但仍須符合所選領域與可評量原則。
若為社工，內容應著重資源連結、家庭支持、社會參與、權益維護、支持網絡與服務協調。
""".strip()


class LongTermGoalRequest(BaseModel):
    tab_name: Literal["成人", "兒童", "社工"] = "成人"
    domain: str = Field(..., description="目前選擇的領域")
    basic_info: str = ""
    summary: str = ""
    dream: str = ""
    custom_to_goal: str = ""
    custom_for_goal: str = ""


class LongTermGoalResponse(BaseModel):
    domain: str
    important_to: list[str]
    important_for: list[str]


class SearchReferencesRequest(BaseModel):
    tab_name: Literal["成人", "兒童", "社工"] = "成人"
    domain: str
    basic_info: str = ""
    selected_long_goal: str
    topk: int = Field(default=6, ge=1, le=20)


class ReferenceItem(BaseModel):
    id: str
    type: str
    label: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


class SearchReferencesResponse(BaseModel):
    domain: str
    search_queries: list[str]
    short_references: list[ReferenceItem]
    strategy_references: list[ReferenceItem]


class GeneratePlanRequest(BaseModel):
    tab_name: Literal["成人", "兒童", "社工"] = "成人"
    domain: str
    basic_info: str = ""
    selected_long_goals: list[str] = Field(default_factory=list)
    selected_short_references: list[str] = Field(default_factory=list)
    selected_strategy_references: list[str] = Field(default_factory=list)


class GeneratePlanResponse(BaseModel):
    domain: str
    result: str


class RefineRequest(BaseModel):
    tab_name: Literal["成人", "兒童", "社工"] = "成人"
    domain: str = ""
    original: str
    refine_prompt: str


class RefineResponse(BaseModel):
    result: str
    intent: dict[str, Any]


cors_origins = [origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if origin.strip()]

app = FastAPI(
    title="IFSP/ISP AI API",
    version="1.0.0",
    description="FastAPI version of the IFSP/ISP generation flow from main_long.py.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or ["*"],
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0.2, max_tokens=3500)


@lru_cache(maxsize=1)
def get_planner_llm() -> ChatOpenAI:
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0)


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    # Jina's remote model code still imports the legacy module-level helper.
    # Transformers 4.55 moved it onto XLMRobertaEmbeddings.
    from transformers.models.xlm_roberta import modeling_xlm_roberta

    if not hasattr(modeling_xlm_roberta, "create_position_ids_from_input_ids"):
        modeling_xlm_roberta.create_position_ids_from_input_ids = (
            modeling_xlm_roberta.XLMRobertaEmbeddings.create_position_ids_from_input_ids
        )
    return CrossEncoder(RERANKER_MODEL, trust_remote_code=True)


@lru_cache(maxsize=1)
def get_chroma_collection():
    client = chromadb.PersistentClient(path=DBDIR)
    return client.get_collection(COLLECTION)


@lru_cache(maxsize=1)
def get_full_names() -> tuple[str, ...]:
    if not os.path.exists(STUDENT_FILE):
        return tuple()
    df = pd.read_excel(STUDENT_FILE)
    if "姓名" not in df.columns:
        return tuple()
    return tuple(df["姓名"].dropna().astype(str).tolist())


def replace_name_with_stars(text: Any) -> str:
    if text is None:
        value = ""
    elif isinstance(text, tuple):
        value = next((v for v in text if isinstance(v, str)), str(text))
    else:
        value = str(text)

    replaced = value
    full_names = get_full_names()
    name_lookup = {name[1:]: name for name in full_names if isinstance(name, str) and len(name) >= 2}

    for full_name in full_names:
        if full_name and full_name in replaced:
            replaced = replaced.replace(full_name, "**")

    for name_tail in name_lookup:
        replaced = re.sub(re.escape(name_tail), "**", replaced)
    return replaced


def parse_goal_lines(text: str) -> list[str]:
    goals = []
    for line in str(text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if "對他重要" in line or "為他重要" in line or line.startswith("領域"):
            continue
        line = re.sub(r"^長程目標\s*\d+\s*[:：]\s*", "", line)
        line = re.sub(r"^\d+\s*[\.、]\s*", "", line)
        line = re.sub(r"^[-•]\s*", "", line).strip()
        if line:
            goals.append(line)
    return goals


def split_lines(text: str) -> list[str]:
    return [x.strip() for x in str(text or "").splitlines() if x.strip()]


def dedupe_goals(goals: list[str]) -> list[str]:
    result = []
    seen = set()
    for goal in goals:
        key = re.sub(r"\s+", "", goal)
        if key and key not in seen:
            seen.add(key)
            result.append(goal)
    return result


def split_custom_goal_text(custom_text: str) -> list[str]:
    raw_goals = split_lines(custom_text)
    if not raw_goals:
        return []

    parts = []
    for goal in raw_goals:
        response = get_llm().invoke([
            ("system", LONG_TERM_SPLIT_PROMPT),
            ("user", goal),
        ])
        parsed = parse_goal_lines(response.content)
        parts.extend(parsed or [goal])
    return parts


def rewrite_one_long_goal(goal: str) -> str:
    response = get_llm().invoke([
        ("system", LONG_TERM_REWRITE_PROMPT),
        ("user", goal),
    ])
    goals = parse_goal_lines(response.content)
    return goals[0] if goals else response.content.strip()


def rewrite_custom_goals(custom_text: str) -> list[str]:
    rewritten = []
    for part in split_custom_goal_text(custom_text):
        goal = rewrite_one_long_goal(part)
        if goal:
            rewritten.append(replace_name_with_stars(goal))
    return dedupe_goals(rewritten)


def generate_extra_goals(
    domain: str,
    basic_info: str,
    summary: str,
    dream: str,
    goal_type: str,
    existing_goals: list[str],
    remain_count: int,
) -> list[str]:
    if remain_count <= 0:
        return []

    existing_text = "\n".join(f"{i + 1}. {goal}" for i, goal in enumerate(existing_goals))
    response = get_llm().invoke(
        prompt_templates.build_generate_extra_goals_messages(
            domain=domain,
            basic_info=basic_info,
            summary=summary,
            dream=dream,
            goal_type=goal_type,
            existing_text=existing_text,
            remain_count=remain_count,
        )
    )
    goals = parse_goal_lines(response.content)
    return [replace_name_with_stars(goal) for goal in goals[:remain_count]]


def build_structured_text(basic_info: str, long_goal_text: str) -> str:
    return f"""
基本資料:
{basic_info}

使用者勾選的長程目標:
{long_goal_text}
""".strip()


def extract_content_only(text: str) -> str:
    value = str(text or "").strip()
    if "內容:" in value:
        value = value.split("內容:", 1)[1].strip()
    elif "內容：" in value:
        value = value.split("內容：", 1)[1].strip()
    value = value.replace("\n", " ").strip()
    return re.sub(r"\s+", " ", value)


def extract_icf_disability_type(text: str) -> str:
    value = str(text or "").strip()
    match = re.search(r"障礙類別\s*[:：]?\s*([0-9.\s、,，]+)\s*類?", value)
    if match:
        nums = re.findall(r"\d+", match.group(1))
        nums = [str(int(n)) for n in nums if 1 <= int(n) <= 8]
        if nums:
            return ",".join(sorted(set(nums), key=lambda x: int(x)))

    nums = re.findall(r"(?:第\s*)?0?([1-8])\s*類", value)
    if nums:
        return ",".join(sorted(set(str(int(n)) for n in nums), key=lambda x: int(x)))

    found = []
    for code, keywords in ICF_DISABILITY_TYPE_MAP.items():
        if any(keyword and keyword in value for keyword in keywords):
            found.append(code)
    return ",".join(sorted(set(found), key=lambda x: int(x))) if found else ""


def extract_disability_level(text: str) -> str:
    value = str(text or "").strip()
    for level in ["極重度", "重度", "中度", "輕度"]:
        if level in value:
            return level
    return ""


def split_disability_types(value: str) -> list[str]:
    cleaned = str(value or "").replace("，", ",").replace("、", ",").replace(".", ",")
    return [str(int(part)) for part in cleaned.split(",") if part.strip().isdigit() and 1 <= int(part.strip()) <= 8]


def ai_query_planner(
    domain: str,
    basic_info: str,
    long_goal_text: list[str],
    service_type: str = "",
) -> dict[str, Any]:
    long_goal_block = "\n".join(f"{i + 1}. {goal}" for i, goal in enumerate(long_goal_text) if str(goal).strip())
    prompt = prompt_templates.build_ai_query_planner_prompt(
        domain=domain,
        basic_info=(
            f"{SERVICE_TYPE_RULES.format(service_type=service_type)}\n\n{basic_info}"
            if service_type
            else basic_info
        ),
        long_goal_block=long_goal_block,
    )
    try:
        response = get_planner_llm().invoke([
            ("system", "你只輸出合法 JSON，不要輸出任何解釋。"),
            ("user", prompt),
        ])
        text = response.content.strip()
        text = re.sub(r"^```json", "", text).strip()
        text = re.sub(r"^```", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        data = json.loads(text)
        queries = data.get("search_queries", [])
        if not isinstance(queries, list):
            raise ValueError("search_queries is not a list")
        data["search_queries"] = [str(query).strip() for query in queries if str(query).strip()]
        return data
    except Exception as e:
        print("⚠️ AI Query Planner 失敗:", e)

        fallback = " ".join([domain or "", basic_info or "", long_goal_block or ""]).strip()
        return {
            "core_intent": fallback,
            "domain_focus": domain,
            "search_focus": fallback,
            "search_queries": [fallback] if fallback else [],
        }


def retrieve_hybrid_with_meta(domain: str, typ: str, query: str, topk: int = 6) -> list[Document]:
    collection = get_chroma_collection()
    where_filter = {"$and": [{"domain": {"$eq": domain}}, {"short": {"$eq": typ}}]}
    qvec = get_embedder().encode([query], normalize_embeddings=True).tolist()
    vec_res = collection.query(
        query_embeddings=qvec,
        n_results=topk,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    result_docs = vec_res.get("documents", [[]])[0]
    result_metas = vec_res.get("metadatas", [[]])[0]
    result_distances = vec_res.get("distances", [[]])[0]
    result_ids = vec_res.get("ids", [[]])[0]
    for document_id, doc, meta, dist in zip(
        result_ids,
        result_docs,
        result_metas,
        result_distances,
    ):
        docs.append(
            Document(
                page_content=str(doc),
                metadata={
                    **(meta or {}),
                    "chroma_id": str(document_id),
                    "semantic_score": 1.0 - float(dist),
                },
            )
        )
    return docs


def rerank_docs(query: str, docs: list[Document], topk: int = 6) -> list[Document]:
    if not docs:
        return []
    scores = get_reranker().predict([[query, doc.page_content] for doc in docs])
    ranked = []
    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = float(score)
        ranked.append(doc)
    return sorted(ranked, key=lambda item: item.metadata["rerank_score"], reverse=True)[:topk]


def retrieve_candidates(
    domain: str,
    typ: str,
    basic_info: str,
    long_goal_text: list[str],
    topk: int = 6,
    service_type: str = "",
    plan: dict[str, Any] | None = None,
) -> tuple[list[str], list[Document]]:
    long_goal_block = "\n".join(f"{i + 1}. {goal}" for i, goal in enumerate(long_goal_text) if str(goal).strip())
    target_disability_type = extract_icf_disability_type(basic_info)
    target_disability_level = extract_disability_level(basic_info)
    target_types = split_disability_types(target_disability_type)
    if plan is None:
        plan = ai_query_planner(
            domain=domain,
            basic_info=basic_info,
            long_goal_text=long_goal_text,
            service_type=service_type,
        )
    planner_queries = plan.get("search_queries", [])

    full_query = f"""
領域：{domain or ""}
基本資料：{basic_info or ""}
勾選長程目標：{long_goal_block or ""}
""".strip()

    query_items = []
    if full_query:
        query_items.append(("full", full_query))
    for query in planner_queries:
        if str(query).strip():
            query_items.append(("planner", str(query).strip()))
    if basic_info:
        query_items.append(("basic_info", basic_info.strip()))
    if long_goal_block:
        query_items.append(("long_goal", long_goal_block.strip()))

    clean_query_items = []
    seen_queries = set()
    for tag, query in query_items:
        key = (tag, query)
        if query and key not in seen_queries:
            seen_queries.add(key)
            clean_query_items.append((tag, query))

    tag_weight = {"full": 3.0, "planner": 2.5, "long_goal": 2.0, "basic_info": 1.0}
    all_docs = []
    for tag, query in clean_query_items:
        docs = retrieve_hybrid_with_meta(domain=domain, typ=typ, query=query, topk=topk)
        docs = rerank_docs(query=query, docs=docs, topk=topk)
        for doc in docs:
            semantic_score = doc.metadata.get("semantic_score", 0)
            rerank_score = doc.metadata.get("rerank_score", 0)
            if semantic_score < 0.7 and rerank_score < 1.0:
                continue

            normalized_rerank = max(min(rerank_score / 10, 1.0), 0)
            final_score = (semantic_score * 0.4 + normalized_rerank * 0.6) * tag_weight.get(tag, 1.0)

            doc_types = split_disability_types(str(doc.metadata.get("disability_type", "")).strip())
            doc_level = str(doc.metadata.get("disability_level", "")).strip()
            disability_bonus = 1.0
            if target_types and doc_types and any(t in doc_types for t in target_types):
                disability_bonus *= 1.05
            if target_disability_level and doc_level and target_disability_level == doc_level:
                disability_bonus *= 1.03
            if target_types and target_disability_level and doc_level and any(t in doc_types for t in target_types) and target_disability_level == doc_level:
                disability_bonus *= 1.10

            final_score *= disability_bonus
            doc.metadata["matched_query"] = query
            doc.metadata["matched_tag"] = tag
            doc.metadata["target_disability_type"] = target_disability_type
            doc.metadata["target_disability_level"] = target_disability_level
            doc.metadata["disability_bonus"] = disability_bonus
            all_docs.append((doc, final_score))

    doc_map: dict[str, dict[str, Any]] = {}
    for doc, score in all_docs:
        key = doc.page_content
        if key not in doc_map:
            doc_map[key] = {"doc": doc, "score": 0.0, "matched_queries": [], "matched_tags": []}
        doc_map[key]["score"] = max(doc_map[key]["score"], score)
        doc_map[key]["matched_queries"].append(doc.metadata.get("matched_query", ""))
        doc_map[key]["matched_tags"].append(doc.metadata.get("matched_tag", ""))

    sorted_docs = sorted(doc_map.values(), key=lambda item: item["score"], reverse=True)
    final_docs = []
    for item in sorted_docs[:topk]:
        doc = item["doc"]
        doc.metadata["final_score"] = float(item["score"])
        doc.metadata["core_intent"] = plan.get("core_intent", "")
        doc.metadata["search_focus"] = plan.get("search_focus", "")
        doc.metadata["matched_queries"] = list(set(item["matched_queries"]))
        doc.metadata["matched_tags"] = list(set(item["matched_tags"]))
        final_docs.append(doc)

    display_queries = [query for _, query in clean_query_items]
    return display_queries, final_docs


def reference_items_from_docs(docs: list[Document], typ: str) -> list[ReferenceItem]:
    items = []
    for idx, doc in enumerate(docs):
        display_text = replace_name_with_stars(extract_content_only(doc.page_content))
        label_text = display_text[:120] + "..." if len(display_text) > 120 else display_text
        raw_score = float(doc.metadata.get("final_score", 0))
        display_score = max(min(raw_score / 5, 1.0), 0.0)
        disability_type = doc.metadata.get("disability_type", "")
        disability_level = doc.metadata.get("disability_level", "")
        disability_text = f"【類別:{disability_type or '-'} 等級:{disability_level or '-'}】 " if disability_type or disability_level else ""
        items.append(
            ReferenceItem(
                id=str(doc.metadata.get("chroma_id") or idx),
                type=typ,
                label=f"({display_score:.2f}) {disability_text}{label_text}",
                content=display_text,
                metadata=doc.metadata,
                score=display_score,
            )
        )
    return items


def build_selected_context(texts: list[str]) -> str:
    clean_texts = [str(text).strip() for text in texts if str(text).strip()]
    return "\n".join(f"{i + 1}. {text}" for i, text in enumerate(clean_texts))


def add_service_type_rules(messages, service_type: str):
    rules = SERVICE_TYPE_RULES.format(service_type=service_type)
    return [
        (role, f"{rules}\n\n{content}" if role == "user" else content)
        for role, content in messages
    ]


def build_case1_prompt(service_type: str, domain: str, structured_text: str, short_context: str):
    return add_service_type_rules(
        prompt_templates.build_case1_prompt(domain, structured_text, short_context),
        service_type,
    )


def build_case2_prompt(
    service_type: str,
    domain: str,
    structured_text: str,
    short_context: str,
    strategy_context: str,
):
    return add_service_type_rules(
        prompt_templates.build_case2_prompt(
            domain,
            structured_text,
            short_context,
            strategy_context,
        ),
        service_type,
    )


def build_case3_prompt(service_type: str, domain: str, structured_text: str):
    return add_service_type_rules(
        prompt_templates.build_case3_prompt(domain, structured_text),
        service_type,
    )


def parse_refine_intent(refine_prompt: str) -> dict[str, Any]:
    prompt = prompt_templates.build_parse_refine_intent_prompt(refine_prompt)
    try:
        response = get_llm().invoke([
            ("system", "你只能輸出 JSON"),
            ("user", prompt),
        ])
        text = response.content.strip()
        text = re.sub(r"^```json", "", text).strip()
        text = re.sub(r"^```", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        return json.loads(text)
    except Exception:
        return {"action": "other", "target": "", "extra": refine_prompt}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "openai_model": OPENAI_MODEL,
        "chroma_db_dir": DBDIR,
        "collection": COLLECTION,
    }


@app.get("/api/domains")
def domains() -> dict[str, list[str]]:
    return {
        "adult": adult_domain_options,
        "child": child_domain_options,
        "social": social_domain_options,
    }


@app.post("/api/long-term-goals", response_model=LongTermGoalResponse)
def api_long_term_goals(payload: LongTermGoalRequest) -> LongTermGoalResponse:
    if not any([
        payload.basic_info.strip(),
        payload.summary.strip(),
        payload.dream.strip(),
        payload.custom_to_goal.strip(),
        payload.custom_for_goal.strip(),
    ]):
        raise HTTPException(status_code=400, detail="請至少填寫基本資料、現況摘要、個人喜好期待與夢想成果，或自訂長程目標其中一項。")

    custom_to = rewrite_custom_goals(payload.custom_to_goal)
    custom_for = rewrite_custom_goals(payload.custom_for_goal)

    basic_info = replace_name_with_stars(payload.basic_info)
    summary = replace_name_with_stars(payload.summary)
    dream = replace_name_with_stars(payload.dream)

    remain_to = max(0, 5 - len(custom_to))
    extra_to = generate_extra_goals(payload.domain, basic_info, summary, dream, "對他重要", custom_to, remain_to)
    important_to = dedupe_goals(custom_to + extra_to)[:5]

    remain_for = max(0, 5 - len(custom_for))
    extra_for = generate_extra_goals(payload.domain, basic_info, summary, dream, "為他重要", custom_for, remain_for)
    important_for = dedupe_goals(custom_for + extra_for)[:5]

    return LongTermGoalResponse(domain=payload.domain, important_to=important_to, important_for=important_for)


@app.post("/api/search-references", response_model=SearchReferencesResponse)
def api_search_references(payload: SearchReferencesRequest) -> SearchReferencesResponse:
    selected_goals = [payload.selected_long_goal] if payload.selected_long_goal.strip() else []
    if not payload.domain.strip() or not selected_goals:
        raise HTTPException(status_code=400, detail="請提供領域與已選擇的長程目標。")

    all_queries: list[str] = []
    short_references: list[ReferenceItem] = []
    strategy_references: list[ReferenceItem] = []

    if payload.tab_name in ["成人", "社工"]:
        query_plan = ai_query_planner(
            payload.domain,
            payload.basic_info,
            selected_goals,
            payload.tab_name,
        )
        short_queries, short_docs = retrieve_candidates(
            payload.domain,
            "短程目標",
            payload.basic_info,
            selected_goals,
            payload.topk,
            payload.tab_name,
            query_plan,
        )
        strategy_queries, strategy_docs = retrieve_candidates(
            payload.domain,
            "策略",
            payload.basic_info,
            selected_goals,
            payload.topk,
            payload.tab_name,
            query_plan,
        )
        all_queries = list(dict.fromkeys(short_queries + strategy_queries))
        short_references = reference_items_from_docs(short_docs, "短程目標")
        strategy_references = reference_items_from_docs(strategy_docs, "策略")
    elif payload.tab_name == "兒童":
        short_queries, short_docs = retrieve_candidates(
            payload.domain,
            "兒童短程目標",
            payload.basic_info,
            selected_goals,
            payload.topk,
            payload.tab_name,
        )
        all_queries = short_queries
        short_references = reference_items_from_docs(short_docs, "兒童短程目標")

    return SearchReferencesResponse(
        domain=payload.domain,
        search_queries=all_queries,
        short_references=short_references,
        strategy_references=strategy_references,
    )


@app.post("/api/generate-plan", response_model=GeneratePlanResponse)
def api_generate_plan(payload: GeneratePlanRequest) -> GeneratePlanResponse:
    if not payload.domain.strip():
        raise HTTPException(status_code=400, detail="請提供領域。")
    if not payload.selected_long_goals:
        raise HTTPException(status_code=400, detail="請至少提供一個已選擇的長程目標。")

    long_goal_text = "\n".join(payload.selected_long_goals)
    structured_text = build_structured_text(payload.basic_info, long_goal_text)
    short_context = build_selected_context(payload.selected_short_references)
    strategy_context = build_selected_context(payload.selected_strategy_references)

    if short_context and strategy_context:
        messages = build_case2_prompt(
            payload.tab_name,
            payload.domain,
            structured_text,
            short_context,
            strategy_context,
        )
    elif short_context:
        messages = build_case1_prompt(
            payload.tab_name,
            payload.domain,
            structured_text,
            short_context,
        )
    else:
        messages = build_case3_prompt(payload.tab_name, payload.domain, structured_text)

    result = get_llm().invoke(messages).content.strip()
    return GeneratePlanResponse(domain=payload.domain, result=result)


@app.post("/api/refine", response_model=RefineResponse)
def api_refine(payload: RefineRequest) -> RefineResponse:
    if not payload.original.strip():
        raise HTTPException(status_code=400, detail="請提供原始內容。")
    if not payload.refine_prompt.strip():
        raise HTTPException(status_code=400, detail="請提供修改要求。")

    intent = parse_refine_intent(payload.refine_prompt)
    messages = prompt_templates.build_refine_answer_messages(
        original=payload.original,
        refine_prompt=payload.refine_prompt,
        intent=intent,
        service_type=payload.tab_name,
        domain=payload.domain,
    )
    messages = add_service_type_rules(messages, payload.tab_name)
    refined = get_llm().invoke(messages).content.strip()
    result, repair_prompt = prompt_templates.finalize_refined_answer(
        original=payload.original,
        refine_prompt=payload.refine_prompt,
        refined=refined,
    )
    if repair_prompt:
        repair_messages = messages + [
            ("assistant", refined),
            ("user", repair_prompt),
        ]
        refined = get_llm().invoke(repair_messages).content.strip()
        result, repair_prompt = prompt_templates.finalize_refined_answer(
            original=payload.original,
            refine_prompt=payload.refine_prompt,
            refined=refined,
        )
    if repair_prompt or result is None:
        raise HTTPException(status_code=502, detail="重新生成結果的面向或格式仍不完整。")
    return RefineResponse(result=result, intent=intent)
