import os
import re
import html

import langchain
import gradio as gr
import pandas as pd
import chromadb
from dotenv import load_dotenv

from chromadb import PersistentClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, CrossEncoder
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
)

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY 未設定，請在 .env 檔案中設定")

langchain.debug = os.environ.get("LANGCHAIN_DEBUG", "false").lower() == "true"

# ── 路徑與常數 ──────────────────────────────────────────────
BASE_DIR    = os.path.abspath(os.path.dirname(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db_115")
COLLECTION  = "rag_knowledge"
TOPK        = 4

# ── Prompt 規則 ─────────────────────────────────────────────
MEASURABLE_STRATEGY_RULES = """
請將每一項策略寫成「可評量策略」，每一點都必須同時包含以下四個要素：
1. 可觀察行為：要寫出服務對象實際要做出的行為，例如主動表達需求、完成洗手步驟、使用學習剪刀剪紙。
2. 情境條件：要寫出行為發生的場合或提示程度，例如在課堂活動中、於午餐後、在口頭提示下、在部分協助下。
3. 達成標準：要寫出可以判斷成功的標準，例如完成至少二個步驟、維持十五秒、正確完成八成以上。
4. 頻率或次數：要寫出每日或每週的發生頻率，例如每天至少二次、每週至少三次、連續四週達成。
每一點都必須能被教保員實際觀察、記錄與判斷是否達成。
不可只寫「提升能力」「增加參與」「改善表現」等籠統語句。
不可只寫教學方法，必須寫出可被評量的結果。
"""

MEASURABLE_TEMPLATE = """
每一點請盡量使用以下句型：
「在【情境條件】下，服務對象能【可觀察行為】，並達成【達成標準】，每【日或週】至少【次數】次。」
範例：
1. 在課堂活動中，服務對象能以口語主動表達自己的選擇，並能清楚說出要或不要，每天至少完成三次。
2. 於午餐前洗手情境中，服務對象能依提示完成洗手五步驟中的至少三個步驟，每天至少完成二次。
"""

# ── 模型初始化（module 層級，只載入一次）────────────────────
_chroma_client = PersistentClient(path=PERSIST_DIR)
_chroma_client.get_or_create_collection(name=COLLECTION)  # 確保 collection 存在

_embm     = SentenceTransformer("intfloat/multilingual-e5-base")
_reranker = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual", trust_remote_code=True)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# ── CSS ─────────────────────────────────────────────────────
CUSTOM_CSS = """
.clean-checkbox .wrap {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.clean-checkbox label {
    display: flex !important;
    align-items: flex-start !important;
    gap: 10px;
    padding: 10px 12px;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    background: #ffffff;
    line-height: 1.6;
    font-size: 15px;
}
.clean-checkbox label:hover {
    background: #f9fafb;
    border-color: #60a5fa;
}
.clean-checkbox input {
    margin-top: 5px;
}
.loading-overlay {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(2px);
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
}
.loading-card {
    background: white;
    padding: 20px 30px;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 18px;
    font-weight: bold;
    color: #2563eb;
}
.spinner {
    width: 26px; height: 26px;
    border: 4px solid #bfdbfe;
    border-top: 4px solid #2563eb;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
"""

# ── 檢索 ────────────────────────────────────────────────────

def retrieve_hybrid_with_meta(domain: str, typ: str, query: str, topk: int = 6) -> list[Document]:
    col = _chroma_client.get_collection(COLLECTION)
    where_filter = {
        "$and": [
            {"domain": {"$eq": domain}},
            {"short":  {"$eq": typ}},
        ]
    }
    qvec = _embm.encode([query], normalize_embeddings=True).tolist()

    # 若 collection 文件數少於 topk，ChromaDB 會報錯，先取上限
    total = col.count()
    n = min(topk, total) if total > 0 else 1
    vec_res = col.query(
        query_embeddings=qvec,
        n_results=n,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for doc, meta, dist in zip(
        vec_res.get("documents", [[]])[0],
        vec_res.get("metadatas", [[]])[0],
        vec_res.get("distances", [[]])[0],
    ):
        docs.append(Document(
            page_content=str(doc),
            metadata={**meta, "semantic_score": 1.0 - float(dist)},
        ))
    return docs


def rerank_docs(query: str, docs: list[Document], topk: int = 6) -> list[Document]:
    if not docs:
        return []
    scores = _reranker.predict([[query, d.page_content] for d in docs])
    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = float(score)
    return sorted(docs, key=lambda d: d.metadata["rerank_score"], reverse=True)[:topk]


def split_query(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"[，．,\n]", text)
    return [p.strip() for p in parts if len(p.strip()) >= 5]


def retrieve_candidates(
    domain, typ, ability, limitation, need,
    ability_weight, limitation_weight, need_weight,
    topk: int = 6,
) -> list[Document]:
    queries = []
    for tag, text in [("ability", ability), ("limitation", limitation), ("need", need)]:
        for q in split_query(text or ""):
            queries.append((tag, q))

    weight_map = {"ability": ability_weight, "limitation": limitation_weight, "need": need_weight}
    doc_map: dict[str, dict] = {}

    for tag, q in queries:
        docs = retrieve_hybrid_with_meta(domain, typ, q, topk=topk)
        docs = rerank_docs(q, docs, topk=topk)
        w = weight_map.get(tag, 1.0)

        for d in docs:
            sem   = d.metadata.get("semantic_score", 0)
            rerank = max(min(d.metadata.get("rerank_score", 0) / 10, 1.0), 0)
            score = (sem * 0.4 + rerank * 0.6) * w
            key = d.page_content
            if key not in doc_map:
                doc_map[key] = {"doc": d, "score": 0.0}
            doc_map[key]["score"] += score

    final = sorted(doc_map.values(), key=lambda x: x["score"], reverse=True)[:topk]
    for item in final:
        item["doc"].metadata["final_score"] = float(item["score"])
    return [item["doc"] for item in final]

# ── 文字工具 ─────────────────────────────────────────────────

def zh_len(s: str) -> int:
    return len(re.findall(r"[一-鿿]", s or ""))


def quick_check(text: str, min_zh: int = 20, max_items: int = 10) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out, seen = [], set()
    for ln in lines:
        ln = re.sub(r"<\|eot_id\|>|<\|eom_id\|>|<\|end_of_text\|>", "", ln).strip()
        m = re.match(r"^\d{1,2}\.\s*(.+)$", ln)
        body = m.group(1).strip() if m else ln
        if not body or "參考資料" in body:
            continue
        if zh_len(body) < min_zh:
            continue
        if body in seen:
            continue
        seen.add(body)
        out.append(body)
        if len(out) >= max_items:
            break
    return "\n".join(f"{i+1}. {t}" for i, t in enumerate(out[:max_items]))


def extract_content_only(text: str) -> str:
    text = str(text or "").strip()
    for sep in ("內容:", "內容："):
        if sep in text:
            text = text.split(sep, 1)[1].strip()
            break
    return re.sub(r"\s+", " ", text).strip()


def replace_name_with_stars(text, full_names: list) -> str:
    if isinstance(text, tuple):
        text = next((v for v in text if isinstance(v, str)), str(text))
    elif text is None:
        text = ""
    else:
        text = str(text)

    name_tails = {name[1:]: name for name in full_names if isinstance(name, str) and len(name) >= 2}
    for name in full_names:
        if isinstance(name, str) and name and name in text:
            text = text.replace(name, "**")
    for tail in name_tails:
        text = re.sub(re.escape(tail), "**", text)
    return text


def import_student_names() -> list:
    try:
        df = pd.read_excel("學生名單.xlsx")
        return df["姓名"].dropna().astype(str).tolist()
    except Exception:
        return []


def search_candidates_by_type(
    domain, typ, ability, limitation, need,
    ability_weight, limitation_weight, need_weight,
    topk: int = 6,
):
    candidates = retrieve_candidates(
        domain, typ, ability, limitation, need,
        ability_weight, limitation_weight, need_weight, topk=topk,
    )
    choices = []
    for i, r in enumerate(candidates):
        display = extract_content_only(r.page_content)
        display = replace_name_with_stars(display, full_names)
        if len(display) > 120:
            display = display[:120] + "..."
        score = max(min(r.metadata.get("final_score", 0) / 3, 1.0), 0.0)
        choices.append((f"({score:.2f}) {display}", str(i)))
    return candidates, gr.update(choices=choices, value=[])


def build_selected_context(selected_items, candidates) -> str:
    texts = []
    for item in (selected_items or []):
        try:
            txt = extract_content_only(candidates[int(item)].page_content)
            texts.append(txt)
        except Exception:
            continue
    return "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))

# ── 生成（評估用，UI 不直接呼叫）────────────────────────────

def _safe_ref_text(refs, max_chars: int = 240) -> str:
    out = []
    for r in (refs or []):
        txt = str(getattr(r, "page_content", r))
        if "內容:" in txt:
            txt = txt.split("內容:", 1)[1].strip()
        txt = re.sub(r"\s+", " ", txt).strip()
        if not txt:
            continue
        if len(txt) > max_chars:
            txt = txt[:max_chars].rstrip() + "…"
        out.append(txt)
    return "\n".join(f"{i+1}. {t}" for i, t in enumerate(out)) if out else "(無)"


def build_messages(domain: str, typ: str, info_sentence: str, refs: list) -> list:
    system = (
        "你是一位特教機構的專業教保員。"
        "所有回答一律使用繁體中文。"
        "只能依據使用者提供的資訊與參考資料推導；不得杜撰。"
        "禁止出現「參考資料」四字；不得貼上參考資料原文。"
    )
    ref_block = _safe_ref_text(refs)
    user = (
        f"【領域】{domain}\n【類型】{typ}\n\n"
        f"以下是一句話的資訊（不可在答案中出現原文）：\n{info_sentence}\n\n"
        f"以下是可用來推導答案的資訊（不可原文照抄）：\n{ref_block}\n\n"
        "請依據上述資訊撰寫「教學目標與策略」，並嚴格遵守以下規則：\n"
        "1) 只輸出條列，每行一點，格式必須為「1. 」「2. 」…\n"
        "2) 只寫 5~8 點；若資訊不足，寧可少點，絕對不要為湊點數而重複。\n"
        "3) 每一點至少 10 個中文字（不含標點與空白）。\n"
        "4) 每一點內容必須彼此不同；禁止同句改字、同義改寫、片語重複堆疊。\n"
        "5) 不可加入參考資料未出現的新專有名詞。\n"
        "6) 不要寫前言、結論、解釋；不要寫小標題。\n"
        "7) 句子必須完整；禁止輸出未完成句子或未完成條列。\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def gradio_generate(domain: str, typ: str, question: str, topk: int = TOPK):
    refs = retrieve_hybrid_with_meta(domain, typ, question, topk=topk)
    refs = rerank_docs(question, refs, topk=topk)
    msgs = build_messages(domain, typ, question, refs)
    response = llm.invoke([("system", msgs[0]["content"]), ("user", msgs[1]["content"])])
    raw = (response.content or "").strip()
    final = quick_check(raw, min_zh=3, max_items=10) or raw
    refs_text = "\n".join(
        f"{i}. {(d.page_content or '').replace(chr(10), ' ')}"
        for i, d in enumerate(refs[:topk], 1)
    )
    return final, refs_text


def run_ragas_eval(eval_samples: list, topk: int = 4):
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    rows = []
    for sample in eval_samples:
        try:
            domain, typ = sample["domain"], sample["typ"]
            question, ground_truth = sample["question"], sample["ground_truth"]
            refs = rerank_docs(question, retrieve_hybrid_with_meta(domain, typ, question, topk=topk), topk=topk)
            answer, _ = gradio_generate(domain, typ, question, topk=topk)
            rows.append({
                "user_input": question,
                "response": answer,
                "retrieved_contexts": [d.page_content for d in refs],
                "reference": ground_truth,
            })
        except Exception as e:
            print("❌ ragas eval error:", e)
    if not rows:
        raise ValueError("Ragas dataset is empty")
    return evaluate(
        EvaluationDataset.from_list(rows),
        metrics=[ContextPrecision(), LLMContextRecall(), Faithfulness(), ResponseRelevancy()],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

# ── 領域選項 & 學生名單 ──────────────────────────────────────
adult_domain_options  = ["身體福祉","情緒福祉","物質福祉","個人發展","自我決策","人際關係","權利","社會融合"]
child_domain_options  = ["健康與安全","感官知覺","精細動作","粗大動作","語言溝通","認知","生活自理","社會適應"]
social_domain_options = ["醫療復健輔具","教育安置","經濟功能及福利輔助","親職支持","家庭支持系統(資援連結)"]

full_names = import_student_names()


def get_active_domain(tab_name, adult, child, social) -> str:
    return {"成人": adult, "兒童": child, "社工": social}.get(tab_name, "")


def build_structured_text(ability, limitation, need) -> str:
    return f"能力:{ability}\n限制:{limitation}\n需求:{need}"

# ── Gradio UI ────────────────────────────────────────────────
with gr.Blocks(css=CUSTOM_CSS) as demo:
    loading_html = gr.HTML("", visible=False)

    def show_loading(text="處理中，請稍候..."):
        return gr.update(
            value=f"""
            <div class="loading-overlay"><div class="loading-card">
                <div class="spinner"></div><div>{text}</div>
            </div></div>""",
            visible=True,
        )

    def hide_loading():
        return gr.update(value="", visible=False)

    def ui_search_candidates(
        tab_name, adult, child, social,
        ability, limitation, need,
        ability_weight, limitation_weight, need_weight,
    ):
        domain = get_active_domain(tab_name, adult, child, social)
        empty_short    = gr.update(choices=[], value=[], visible=True)
        empty_strategy = gr.update(choices=[], value=[], visible=False)

        if not domain or (not ability and not limitation and not need):
            return [], [], empty_short, empty_strategy, gr.update(visible=False), gr.update(visible=False)

        if tab_name in ("成人", "社工"):
            short_cands,    short_upd    = search_candidates_by_type(domain, "短程目標", ability, limitation, need, ability_weight, limitation_weight, need_weight)
            strategy_cands, strategy_upd = search_candidates_by_type(domain, "策略",    ability, limitation, need, ability_weight, limitation_weight, need_weight)
            return short_cands, strategy_cands, short_upd, strategy_upd, gr.update(visible=True), gr.update(visible=True)

        if tab_name == "兒童":
            short_cands, short_upd = search_candidates_by_type(domain, "短程目標", ability, limitation, need, ability_weight, limitation_weight, need_weight)
            return short_cands, [], short_upd, gr.update(choices=[], value=[], visible=False), gr.update(visible=True), gr.update(visible=False)

        return [], [], empty_short, empty_strategy, gr.update(visible=False), gr.update(visible=False)

    def ui_generate_selected_results(
        tab_name, adult, child, social,
        ability, limitation, need,
        selected_short, short_cands,
        selected_strategy, strategy_cands,
    ):
        structured_text  = build_structured_text(ability, limitation, need)
        short_context    = build_selected_context(selected_short,    short_cands)
        strategy_context = build_selected_context(selected_strategy, strategy_cands)

        if short_context and not strategy_context:
            messages = build_case1_prompt(structured_text, short_context)
        elif short_context and strategy_context:
            messages = build_case2_prompt(structured_text, short_context, strategy_context)
        else:
            messages = build_case3_prompt(structured_text)

        result = llm.invoke(messages).content.strip()
        html_result = f"""
        <h3>🤖 生成結果</h3>
        <div style="border:1px solid #ccc;padding:1em;border-radius:8px;background:#f9f9f9;">
            <pre>{html.escape(result)}</pre>
        </div>"""
        return html_result, result, gr.update(visible=True)

    def build_case1_prompt(structured_text, short_context):
        return [
            ("system", "你是特殊教保員"),
            ("user", f"{structured_text}\n\n短程目標:\n{short_context}\n\n"
             "請為每一個短程目標生成2~3個支持策略。\n每個策略需包含：行為、條件、頻率。\n條列輸出"),
        ]

    def build_case2_prompt(structured_text, short_context, strategy_context):
        return [
            ("system", "你是特殊教保員"),
            ("user", f"{structured_text}\n\n短程目標:\n{short_context}\n\n策略:\n{strategy_context}\n\n"
             "請整合並優化：每個短程目標與對應策略（需可評量 + 頻率）。\n條列輸出"),
        ]

    def build_case3_prompt(structured_text):
        return [
            ("system", "你是特殊教保員"),
            ("user", f"【使用者填答內容】\n{structured_text}\n\n"
             "使用者沒有勾選任何參考資料，請自行生成：\n"
             "1. 至少 5 個短程目標。\n"
             "2. 每個短程目標對應 2 至 3 個支持策略。\n"
             "3. 每個策略必須可評量，且包含：情境條件、可觀察行為、達成標準、頻率或次數。\n"
             "4. 不要寫前言、結論或說明。\n"
             "5. 只輸出條列內容。"),
        ]

    def refine_answer(original, refine_prompt):
        messages = [
            ("system", "你是一位特教機構的專業教保員，請根據使用者要求重新調整內容。"),
            ("user",
             f"【原本回答】\n{original}\n\n【使用者希望調整的方向】\n{refine_prompt}\n\n"
             "請重新生成更符合需求的版本。\n規則：1.使用繁體中文 2.保持條列格式 3.不要寫前言或結論 4.策略需可評量且包含頻率"),
        ]
        refined = llm.invoke(messages).content
        html_result = f"""
        <h3>🤖 重新生成結果</h3>
        <div style="border:1px solid #ccc;padding:1em;border-radius:8px;background:#f9f9f9;">
            <pre>{html.escape(refined)}</pre>
        </div>"""
        return html_result, refined, gr.update(visible=True)

    # ── 版面 ──────────────────────────────────────────────────
    gr.Markdown("# 支援ISP/IFSP問答系統")
    current_tab = gr.State("成人")

    with gr.Tabs():
        with gr.Tab("成人") as adult_tab:
            adult_domain = gr.Dropdown(choices=adult_domain_options, label="領域", interactive=True)
        with gr.Tab("兒童") as child_tab:
            child_domain = gr.Dropdown(choices=child_domain_options, label="領域", interactive=True)
        with gr.Tab("社工") as social_tab:
            social_domain = gr.Dropdown(choices=social_domain_options, label="領域", interactive=True)

    adult_tab.select(lambda: "成人", outputs=current_tab)
    child_tab.select(lambda: "兒童", outputs=current_tab)
    social_tab.select(lambda: "社工", outputs=current_tab)

    gr.Markdown("## 搜尋的權重設定")
    with gr.Row():
        ability_weight    = gr.Slider(minimum=0.5, maximum=3.0, value=1.0, label="能力權重",    interactive=True)
        limitation_weight = gr.Slider(minimum=0.5, maximum=3.0, value=1.2, label="限制權重",    interactive=True)
        need_weight       = gr.Slider(minimum=0.5, maximum=3.0, value=2.0, label="需求權重",    interactive=True)

    ability    = gr.Textbox(label="A. 能力（個案目前可以做到的）", lines=2)
    limitation = gr.Textbox(label="B. 限制（目前的困難或不足）",  lines=2)
    need       = gr.Textbox(label="C. 需求（需要學習或改善的方向）", lines=2)

    short_candidate_state    = gr.State([])
    strategy_candidate_state = gr.State([])

    gr.Markdown("## 🧭 步驟 1：搜尋參考資料")
    search_btn = gr.Button("🔍 搜尋參考資料(請按我)", variant="primary")

    with gr.Column(visible=False) as short_section:
        gr.Markdown("## 🧭 步驟 2：勾選資料")
        gr.Markdown("""<div style="background:#fff7ed;padding:10px;border-radius:8px;
            border:1px solid #f59e0b;font-size:18px;font-weight:bold;color:#b45309;">
            📌 請勾選要納入生成的短程目標參考資料</div>""")
        short_checkbox = gr.CheckboxGroup(choices=[], interactive=True, elem_classes=["clean-checkbox"])

    with gr.Column(visible=False) as strategy_section:
        gr.Markdown("""<div style="background:#eff6ff;padding:10px;border-radius:8px;
            border:1px solid #3b82f6;font-size:18px;font-weight:bold;color:#1d4ed8;">
            📌 請勾選要納入生成的策略參考資料</div>""")
        strategy_checkbox = gr.CheckboxGroup(choices=[], interactive=True, elem_classes=["clean-checkbox"])

        gr.Markdown("## 🧭 步驟 3：生成結果")
        generate_btn = gr.Button("✨ AI回答(請按我)", variant="primary")
        output       = gr.HTML(label="系統回應")
        answer_state = gr.State("")

    with gr.Column(visible=False) as refine_section:
        gr.Markdown("## 🔧 不滿意AI生成的結果？")
        refine_prompt = gr.Textbox(
            label="請輸入希望調整的方向", lines=3,
            placeholder="例如：希望更強調生活自理、希望策略更具體...",
        )
        refine_btn = gr.Button("🔄 重新生成", variant="secondary")

    # ── 事件綁定 ───────────────────────────────────────────────
    refine_btn.click(
        fn=refine_answer,
        inputs=[answer_state, refine_prompt],
        outputs=[output, answer_state, refine_section],
    )

    search_btn.click(
        fn=lambda: show_loading("正在搜尋資料..."),
        outputs=loading_html,
    ).then(
        fn=ui_search_candidates,
        inputs=[current_tab, adult_domain, child_domain, social_domain,
                ability, limitation, need,
                ability_weight, limitation_weight, need_weight],
        outputs=[short_candidate_state, strategy_candidate_state,
                 short_checkbox, strategy_checkbox, short_section, strategy_section],
    ).then(fn=hide_loading, outputs=loading_html)

    generate_btn.click(
        fn=lambda: show_loading("AI正在生成..."),
        outputs=loading_html,
    ).then(
        fn=ui_generate_selected_results,
        inputs=[current_tab, adult_domain, child_domain, social_domain,
                ability, limitation, need,
                short_checkbox, short_candidate_state,
                strategy_checkbox, strategy_candidate_state],
        outputs=[output, answer_state, refine_section],
    ).then(fn=hide_loading, outputs=loading_html)

    # 切換分頁時清空所有欄位
    for tab in [adult_tab, child_tab, social_tab]:
        tab.select(
            fn=lambda: ("", "", "", "",
                        [], [],
                        gr.update(choices=[], value=[]),
                        gr.update(choices=[], value=[]),
                        gr.update(visible=False),
                        gr.update(visible=False)),
            outputs=[ability, limitation, need, output,
                     short_candidate_state, strategy_candidate_state,
                     short_checkbox, strategy_checkbox,
                     short_section, strategy_section],
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
