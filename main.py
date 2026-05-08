
import os, re, html
import langchain
import gradio as gr
import pandas as pd
import torch
import requests
import jieba
import numpy as np
import chromadb
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from chromadb import PersistentClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda


from sentence_transformers import SentenceTransformer, CrossEncoder


from rank_bm25 import BM25Okapi
from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    ResponseRelevancy,
    ContextPrecision,
)


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

/* 🔥 全畫面遮罩 */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(2px);
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
}
/* 🔥 中間卡片 */
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
/* 🔄 spinner */
.spinner {
    width: 26px;
    height: 26px;
    border: 4px solid #bfdbfe;
    border-top: 4px solid #2563eb;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
"""



if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY 未設定，請在 .env 檔案中設定")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
langchain.debug = os.environ.get("LANGCHAIN_DEBUG", "false").lower() == "true"
base_dir = os.path.abspath(os.path.dirname(__file__))
persist_dir = os.path.join(base_dir, "chroma_db")
# model_path = os.path.join(base_dir, "models", "gemma-2b-it.Q4_K_M.gguf")
#LLAMA_SERVER = "http://127.0.0.1:8080"
#LLAMA_SERVER = "https://llm-api-663386326083.asia-east1.run.app"
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={
        "device": "cpu",
    },                 # Apple Silicon GPU（若無MPS可改"cpu"）
    encode_kwargs={"normalize_embeddings": True},
)
# 新增知識
# sentence_embedder = SentenceTransformer("all-MiniLM-L6-v2")
#sentence_embedder = SentenceTransformer("intfloat/multilingual-e5-base")
client = PersistentClient(path=persist_dir)
collection = client.get_or_create_collection(name="rag_knowledge")
vectorstore = Chroma(
    client=client,
    collection_name="rag_knowledge",
    persist_directory=persist_dir,
    embedding_function=embedding_model,
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature = 0.2    
)

#直接LoRA推理
BASE="yentinglin/Taiwan-LLM-7B-v2.0-chat"
ADAPTER = "runs/student-lora-8b"
DBDIR = "chroma_db"
COLLECTION = "rag_knowledge"

TOPK=4
MAXLEN=512
MAX_NEW=512



STRICT_RULES = (
    "請嚴格遵守：\n"
    "A) 只輸出條列；每一行僅一個要點，不可在同一行再使用任何子編號或分點；\n"
    "B) 從 1. 開始連號到 n.，不得跳號；\n"
    "C) 最多 10 點；\n"
    "D) 每一點至少二十個『中文字』（不含標點與空白）；\n"
    "E) 禁止出現「參考資料」字樣；\n"
    "F) 不得加入任何前後語或解釋。"
)

# 可放一個簡短示例，強化“從1開始、字數夠、最多10點”的行為
FEWSHOT = (
    "範例（僅示意格式，不代表最終內容）：\n"
    "1. 這是一段超過二十個中文字的範例敘述展示條列格式與長度要求讓你理解規則。\n"
    "2. 這是一段超過二十個中文字的範例敘述展示條列格式與長度要求讓你理解規則。\n"
)

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


def zh_tokenize(text:str):
    text = str(text or "").strip()
    # jieba 對中文 BM25 會比 split() 好很多
    return [tok.strip() for tok in jieba.lcut(text) if tok.strip()]

def reciprocal_rank_fusion(rank_lists, k: int = 60):
    """
    rank_lists: list[list[str]]
        每個 list 裡放 doc_id，順序代表排名
    回傳:dict[doc_id, fused_score]
    """
    scores = {}
    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def _safe_ref_text(refs, max_chars=240):
    out = []
    for r in (refs or []):
        txt = getattr(r, "page_content", r)
        txt = str(txt)
        
        if "內容:" in txt:
            txt = txt.split("內容:", 1)[1].strip()
        txt = re.sub(r"\s+", " ", txt).strip()
        if not txt:
            continue
        
        if len(txt) > max_chars:
            txt = txt[:max_chars].rstrip() + "…"
        
        out.append(txt)
        
    if not out:
        return "(無)"
    return "\n".join([f"{i+1} .{t}" for i, t in enumerate(out)])

def build_structured_text(ability, limitation, need):
    return f"""
        能力:{ability}
        限制:{limitation}
        需求:{need}
    """

def build_messages(domain: str, typ: str, info_sentence: str, refs:list[str]):
    system = (
        "你是一位特教機構的專業教保員。"
        "所有回答一律使用繁體中文。"
        "只能依據使用者提供的資訊與參考資料推導；不得杜撰。"
        "禁止出現「參考資料」四字；不得貼上參考資料原文。"
    )

    ref_block = _safe_ref_text(refs)
    
    user = (
        f"【領域】{domain}\n"
        f"【類型】{typ}\n\n"
        "以下是一句話的資訊（不可在答案中出現原文）：\n"
        f"{info_sentence}\n\n"
        "以下是可用來推導答案的資訊（不可原文照抄）：\n"
        f"{ref_block}\n\n"
        "請依據上述資訊撰寫「教學目標與策略」，並嚴格遵守以下規則：\n"
        "1) 只輸出條列，每行一點，格式必須為「1. 」「2. 」…\n"
        "2) 只寫 5~8 點；若資訊不足，寧可少點，絕對不要為湊點數而重複。\n"
        "3) 每一點至少 10 個中文字（不含標點與空白）。\n"
        "4) 每一點內容必須彼此不同；禁止同句改字、同義改寫、片語重複堆疊。\n"
        "5) 不可加入參考資料未出現的新專有名詞（例如新量表、新技法、新機構名）。\n"
        "6) 不要寫前言、結論、解釋；不要寫小標題如「教學目標：」「教學策略：」。\n"
        "7) 句子必須完整；禁止輸出未完成句子或未完成條列。\n"
    )
    
    return [
        {"role":"system","content":system},
        {"role":"user","content":user}
    ]


_embm = SentenceTransformer("intfloat/multilingual-e5-base")

def split_query(text):
    if not text:
        return []
    
    parts = re.split(r"[，．,'\n]",text)
    return [p.strip() for p in parts if len(p.strip()) >= 5]

def retrieve_hybrid_with_meta(domain, typ, query, topk=6):
    client = chromadb.PersistentClient(path=DBDIR)
    col = client.get_collection(COLLECTION)
    
    where_filter = {
        "$and": [
            {"domain":{"$eq": domain}},
            {"short":{"$eq":typ}},    
        ]    
    }
    
    # # 1) 先抓出該 domain/typ 的所有候選文件
    # corpus = col.get(
    #     where=where_filter,
    #     include=["documents", "metadatas"]    
    # )
    # corpus_docs = corpus.get("documents", [])
    # corpus_metas = corpus.get("metadatas", [])
    # if not corpus_docs:
    #     return []
    
    # # 幫每篇文件做唯一 id
    # doc_ids = [f"doc_{i}" for i in range(len(corpus_docs))]
    
    # # 2) BM25 lexical search
    # tokenized_corpus = [zh_tokenize(doc) for doc in corpus_docs]
    # bm25 = BM25Okapi(tokenized_corpus)
    # bm25_scores = bm25.get_scores(zh_tokenize(query))
    
    # bm25_rank_idx = np.argsort(bm25_scores)[::-1][:min(bm25_topn, len(corpus_docs))]
    # bm25_rank_ids = [doc_ids[i] for i in bm25_rank_idx]

    # # 3)
    qvec = _embm.encode([query], normalize_embeddings=True).tolist()
    
    vec_res = col.query(
        query_embeddings=qvec,
        n_results=topk,
        where=where_filter,
        include=["documents", "metadatas", "distances"]    
    )
    
    vec_docs = []
    result_docs = vec_res.get("documents", [[]])[0]
    result_metas = vec_res.get("metadatas", [[]])[0]
    result_distances = vec_res.get("distances", [[]])[0]
    
    # 用內容 + metadata 對回原 corpus index
    for doc, meta, dist in zip(result_docs, result_metas, result_distances):
        similarity = 1.0 - float(dist)
        
        vec_docs.append(
            Document(
                page_content=str(doc),
                metadata={
                    **meta,
                    "semantic_score": similarity,    
                }                 
            )
        )
    return vec_docs
    #     for i, (cd, cm) in enumerate(zip(corpus_docs, corpus_metas)):
    #         if d == cd and m == cm:
    #             vec_rank_ids.append(doc_ids[i])
    #             break
    # # 4) RRF 融合
    # fused_scores = reciprocal_rank_fusion([bm25_rank_ids, vec_rank_ids], k=60)
    # fused_sorted_ids = sorted(fused_scores, key=lambda x:fused_scores[x], reverse=True)[:topk]
    
    # # 5) 轉回 Document
    # id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    # results = []
    # for doc_id in fused_sorted_ids:
    #     idx = id_to_idx[doc_id]
    #     results.append(
    #         Document(
    #             page_content=str(corpus_docs[idx]),
    #             metadata={
    #                 **corpus_metas[idx],
    #                 "score":float(fused_scores[doc_id])    
    #             }            
    #         )
    #     )
    # return results

_reranker = CrossEncoder("jinaai/jina-reranker-v2-base-multilingual",trust_remote_code=True)

def rerank_docs(query, docs, topk=6):
    if not docs:
        return[]
    pairs = [[query, d.page_content] for d in docs]
    scores = _reranker.predict(pairs)
    
    new_docs = []
    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = float(score)
        new_docs.append(doc)
        
    ranked = sorted(
        new_docs,
        key=lambda x: x.metadata["rerank_score"],
        reverse=True    
    )
    return ranked[:topk]

def run_ragas_eval(eval_samples, topk=4):
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    evaluator_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    rows = []
    
    for sample in eval_samples:
        try:
            domain = sample["domain"]
            typ = sample["typ"]
            question = sample["question"]
            ground_truth = sample["ground_truth"]
        
            refs = retrieve_hybrid_with_meta(domain, typ, question, topk=topk)
            refs = rerank_docs(question, refs, topk=topk)
        
            answer, _ = gradio_generate(domain, typ, question, topk=topk)
        
            rows.append({
                "user_input": question,
                "response": answer,
                "retrieved_contexts":[d.page_content for d in refs],
                "reference":ground_truth
            })
        except Exception as e:
            print("❌ error:", e)
    
    if len(rows) == 0:
        raise ValueError("Ragas dataset is empty！請檢查 gradio_generate 或 eval_samples")
    
    dataset = EvaluationDataset.from_list(rows)
    
    result = evaluate(
        dataset=dataset,
        metrics=[
            ContextPrecision(),
            LLMContextRecall(),
            Faithfulness(),
            ResponseRelevancy(),        
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    return result
        
def refs_to_text(refs, topk=TOPK):
    out = []
    for i, d in enumerate(refs[:topk], 1):
        txt = (d.page_content or "").replace("\n", " ").strip()
        if "內容:" in txt:
            txt = txt.split("內容:", 1)[1].strip()
        out.append(f"{i}. {txt}")
    return "\n".join(out)

def gradio_generate(domain, typ, question, topk=TOPK):
    #refs = retrieve_with_meta(domain, typ, question,topk=topk)
    refs = retrieve_hybrid_with_meta(domain, typ, question, topk=topk)
    refs = rerank_docs(question, refs, topk=topk)
    
    msgs = build_messages(domain, typ, question, refs)
    sys = msgs[0]["content"]
    usr = msgs[1]["content"]
    
    #改用 OpenAI生成
    messages = [
        ("system", sys),
        ("user", usr),    
    ]
    response = llm.invoke(messages)
    raw = (response.content or "").strip()
    
    final = quick_check(raw, min_zh=3, max_items=10)
    if not final:
        final = raw.strip()
        
    refs_text = refs_to_text(refs, topk=topk) if refs else ""
    return final, refs_text
   
    # raw, finish = llama_server_chat(sys, usr, stream=False)
    # raw = (raw or "").strip() 

    # if finish == "length":
    #     raw = _trim_incomplete_last_line(raw)
        
    # final = quick_check(raw, min_zh=3,max_items=10)
    # if not final:
    #     final = raw.strip()

    # refs_text = refs_to_text(refs, topk=topk) if refs else ""
    
    # return final,refs_text



def zh_len(s: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", s or ""))

def clean_artifacts(text:str):
    if not text:
        return text
    text = re.sub(r"<<\s*>>", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def quick_check(text: str, min_zh=20, max_items=10):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    out, seen = [], set()
    for ln in lines:
        # 去除模型可能產生的特殊 token
        ln = re.sub(r"<\|eot_id\|>|<\|eom_id\|>|<\|end_of_text\|>", "", ln).strip()
        m = re.match(r"^\d{1,2}\.\s*(.+)$", ln)
        body = m.group(1).strip() if m else ln
        if not body or "參考資料" in body: continue
        if zh_len(body) < min_zh: continue
        if body in seen: continue
        seen.add(body); out.append(body)
        if len(out) >= max_items: break
    return "\n".join(f"{i+1}. {t}" for i,t in enumerate(out[:max_items]))


def import_student_names() -> list:
    try:
        df = pd.read_excel("學生名單.xlsx")
        return df["姓名"].dropna().astype(str).tolist()
    except Exception:
        return []

adult_domain_options = ["身體福祉","情緒福祉","物質福祉","個人發展","自我決策","人際關係","權利","社會融合"]
child_domain_options = ["健康與安全","感官知覺","精細動作","粗大動作","語言溝通","認知","生活自理","社會適應"]
social_domain_options = ["醫療復健輔具","教育安置","經濟功能及福利輔助","親職支持","家庭支持系統(資援連結)"]
students_name = {}
generation = None
full_names = import_student_names()

# def add_knowledge(text, source="user_input", category="支持策略"):
#     embedding = sentence_embedder.encode(text).tolist()
#     doc_id = f"user_doc{int.from_bytes(os.urandom(4),'big')}"
#     collection.add(
#         documents=[text],
#         embeddings=[embedding],
#         ids=[doc_id],
#         metadatas=[{"source": source, "category": category}],
#     )
#     print(f"已加入知識(ID:{doc_id})")




# 替換人名
def replace_name_with_stars(text: str, full_names: list) -> str:
    
    if isinstance(text, tuple):
        for v in text:
            if isinstance(v, str):
                text = v
                break
        else:
            text = str(text)
    elif text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)
    replaced = text
    name_lookup = {name[1:]: name for name in full_names if isinstance(name, str) and len(name) >= 2}

    for full_name in full_names:
        
        if isinstance(full_name, str) and full_name and full_name in replaced:
            replaced = replaced.replace(full_name, "**")

    for name_tail, full_name in name_lookup.items():
        replaced = re.sub(re.escape(name_tail), "**", replaced)
    return replaced

def switch_toggle(state):
    global generation
    generation = state

def enforce_numbered_list(text):
    if not re.search(r"^\d+\.", text, re.M):
        sentences = re.split(r"[。！？]", text)
        lines = []
        seen = set()
        idx = 1
        for s in sentences:
            s = s.strip()
            if len(s) > 0 and s not in seen:
                lines.append(f"{idx}. {s}")
                seen.add(s)
                idx += 1
        return "\n".join(lines)
    else:
        lines = []
        seen = set()
        for line in text.splitlines():
            content = re.sub(r"^\d+\.\s*", "", line).strip()
            if content and content not in seen:
                lines.append(line)
                seen.add(content)
        return "\n".join(lines)
    
def retrieve_candidates(domain, typ, ability, limitation, need, ability_weight, limitation_weight, need_weight, topk=6):
    """
    執行二階段搜尋：
    1. Hybrid Search：BM25 + Vector
    2. CrossEncoder rerank
    最後回傳可供 Checkbox 顯示的候選資料
    """
    
    queries = []
    if ability:
        for q in split_query(ability):
            queries.append(("ability",q))
    if limitation:
        for q in split_query(limitation):
            queries.append(("limitation", q))
    if need:
        for q in split_query(need):
            queries.append(("need", q))
            
        
    all_docs=[]
    
    for tag, q in queries:
        docs = retrieve_hybrid_with_meta(domain, typ, q, topk=topk)
        docs = rerank_docs(q, docs, topk=topk)

        if tag == "ability":
            weight = ability_weight
        elif tag == "limitation":
            weight = limitation_weight
        elif tag == "need":
            weight = need_weight
        else:
            weight = 1.0
            
            
        for i, d in enumerate(docs):
            semantic_score = d.metadata.get("semantic_score",0)
            rerank_score = d.metadata.get("rerank_score", 0)

            rerank_score = max(min(rerank_score / 10, 1.0 ), 0)
            final_score = (semantic_score * 0.4 + rerank_score * 0.6)

            final_score *= weight            

            all_docs.append((d, final_score))
            
    doc_map = {}
    
    for d, s in all_docs:
        key = d.page_content
        
        if key not in doc_map:
            doc_map[key] = {
                "doc": d,
                "score": 0.0,    
            }
        doc_map[key]["score"] += s
        
    sorted_docs = sorted(
        doc_map.values(), 
        key=lambda x: x["score"], 
        reverse=True
    )
    
    final_docs = []
    
    for item in sorted_docs[:topk]:
        d = item["doc"]
        d.metadata["final_score"] = float(
            item["score"]
        )
        final_docs.append(d)
    return final_docs

def extract_content_only(text: str) -> str:
    text = str(text or "").strip()
    
    if "內容:" in text:
        text = text.split("內容:", 1)[1].strip()
    elif "內容：" in text:
        text = text.split("內容：", 1)[1].strip()
        
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    
    return text

def search_childidates_by_type(domain, typ, ability, limitation, need, ability_weight, limitation_weight, need_weight, topk=6):
    
    candidates = retrieve_candidates(domain, typ, ability, limitation, need, ability_weight, limitation_weight, need_weight, topk=topk)
    
    choices = []
    
    for i,r in enumerate(candidates):
        display_text = extract_content_only(r.page_content)
        
        display_text = replace_name_with_stars(display_text, full_names)
        
        if len(display_text) > 120:
            display_text = display_text[:120] + "..."
            
        score = r.metadata.get("final_score",0)
        
        score = max(min(score / 3, 1.0), 0.0)

        

        label = (
            
            f"({score:.2f}) "
            f"{display_text}"

        )

        choices.append(
            (
                label,
                str(i)
            )
        )
    return candidates, gr.update(choices=choices, value=[])
    

def build_selected_context(selected_item, candidates):
    """
    將使用者勾選的項目轉成 prompt context
    """
    if not selected_item:
        return ""
    
    selected_texts = []
    for item in selected_item:
        try:
            idx = int(item)
            txt = candidates[idx].page_content
            txt = extract_content_only(txt)
            selected_texts.append(txt)
        except Exception:
            continue
    return "\n".join(
        [f"{i+1}. {txt}" for i, txt in enumerate(selected_texts)]    
    )

def build_short_goal_messages(domain, goal, selected_context):
    system = (
        "你是一位特教機構的專業教保員。"
        "所有回答一律使用繁體中文。"
        "請根據使用者勾選的內容，協助撰寫短程目標。"
        "不得杜撰與上下文無關的內容。"
        "禁止出現「參考資料」四字。" 
    )
    
    user = (
        f"【領域】{domain}\n\n"
        "【使用者輸入的需求或目標】\n"
        f"{goal}\n\n"
        "【使用者勾選的短程目標參考內容】\n"
        f"{selected_context}\n\n"
        "請根據上述內容生成短程目標。\n\n"
        "輸出格式規則：\n"
        "1. 只輸出條列，每行一點，格式必須為「1. 」「2. 」...\n"
        "2. 請生成 3 至 6 點。\n"
        "3. 每一點必須是完整句。\n"
        "4. 不要寫前言、結論、說明或小標題。\n"
        "5. 不得直接照抄勾選內容。\n"
        "6. 此處不需要加入頻率、次數或可評量條件。\n"
    )
    return [ ("system", system),("user", user)]

def build_measurable_strategy_messages(domain, goal, selected_context):
    system = (
        "你是一位特教機構的專業教保員。"
        "所有回答一律使用繁體中文。"
        "請根據使用者勾選的內容，生成具體、可觀察、可記錄、可評量的支持策略。"
        "不得杜撰與上下文無關的內容。"
        "禁止出現「參考資料」四字。"
    )
    user = (
        f"【領域】{domain}\n"
        "【使用者輸入的需求或目標】\n"
        f"{goal}\n\n"
        "【使用者勾選的策略參考內容】\n"
        f"{selected_context}\n\n"
        "請根據上述勾選內容，生成可評量的支持策略。\n\n"
        f"{MEASURABLE_STRATEGY_RULES}\n\n"
        f"{MEASURABLE_TEMPLATE}\n\n"
        "輸出格式規則：\n"
        "1. 只輸出條列，每行一點，格式必須為「1. 」「2. 」...\n"
        "2. 請生成 3 至 6 點。\n"
        "3. 每一點必須是完整句。\n"
        "4. 每一點都必須包含情境條件、可觀察行為、達成標準、頻率或次數。\n"
        "5. 不要寫前言、結論、說明或小標題。\n"
        "6. 不得直接照抄勾選內容。\n"
    )
    
    return [
        ("system", system),
        ("user", user)    
    ]



def generate_short_goals(domain, goal, selected_items, candidates):
    selected_context = build_selected_context(selected_items, candidates)
    
    if not selected_context:
        return "請先勾選至少一筆短程目標參考內容"
    
    messages = build_short_goal_messages(
        domain=domain,
        goal=goal,
        selected_context=selected_context,
    )
    
    response = llm.invoke(messages)
    raw = (response.content or "").strip()
    final = quick_check(raw, min_zh=10, max_items=10)
    
    if not final:
        final = raw
    return final

def generate_strategies(domain, goal, selected_items, candidates):
    selected_context = build_selected_context(selected_items, candidates)
    
    if not selected_context:
        return "請先勾選至少一筆策略參考內容。"
    
    messages = build_measurable_strategy_messages(
        domain=domain,
        goal=goal,
        selected_context=selected_context
    )
    
    response = llm.invoke(messages)
    raw = (response.content or "").strip()
    
    final = quick_check(raw, min_zh=10, max_items=10)
    
    if not final:
        final = raw
    return final

def ask(domain, goal, category=["短程目標"], progress=gr.Progress()):
    strategy_html = ""
    displays = {"short": "", "strategy": ""}
    progress(0.2, desc="語意分析中...")
    full_names = import_student_names()

    progress(0.5, desc="資料庫檢索中...")
    if goal:
        if len(category) > 0:
            for t in category:
                # retriever = vectorstore.as_retriever(
                #     search_type="similarity_score_threshold",
                #     search_kwargs={
                #         "filter": {"$and": [{"domain": domain}, {"short": t}]},
                #         "k": 10,
                #         "score_threshold": 0.3,
                #     },
                # )
                # goal = f"""query:{goal.strip()}"""
                # docs = retriever.invoke(goal)
                goal_query = f"query:{goal.strip()}"
                # docs = retrieve_hybrid_with_meta(domain, t, goal_query, topk=10)
                # docs = rerank_docs(goal_query, docs, topk=10)
                docs = retrieve_candidates(domain, t, goal_query, topk=6)
                display = f""""""
                if len(docs) == 0:
                    if t == "短程目標":
                        displays["short"] = f"""沒有任何符合的資訊可供參考"""
                    elif t == "策略":
                        displays["strategy"] = f"""沒有任何符合的資訊可供參考"""
                else:
                    for i, doc in enumerate(docs, 1):
                        snippet = doc.page_content.replace("\n", " ")
                        snippet = re.sub(r"^\d+(?:\.\d+)*","",snippet.split("內容:")[1])
                        snippet = replace_name_with_stars(snippet, full_names)
                        display += f"""{i}.{snippet}\n"""
                    if t == "短程目標":
                        if generation:
                            final, refs_text = gradio_generate(domain, t, goal)
                            print("短程:使用生成回答" )
                            # result = chain.invoke({"ref_text":goal},min_new_tokens=300)
                            # messages = build_messages(domain=domain,typ=t, info_sentence=goal)
                            # out = generate(toks, models, messages, temperature=0.0, do_sample=False)
                            # displays["short"] = replace_name_with_stars(quick_check(out, min_zh=20, max_items=10), full_names)
                            
                            displays["short"] = replace_name_with_stars(final,full_names)
                        else:
                            print("短程:未使用生成回答" )
                            displays["short"] = display
                        print("==============")
                    elif t == "策略":
                        if generation:
                            final, refs_text = gradio_generate(domain, t, goal)
                            print("策略:使用生成回答" )
                            #result = chain.invoke({"ref_text": goal},min_new_tokens=300)
                            # messages = build_messages(domain=domain,typ=t, info_sentence=goal)
                            # out = generate(toks, models, messages, temperature=0.0, do_sample=False)
                            # displays["strategy"] = replace_name_with_stars(quick_check(out, min_zh=20, max_items=10),full_names)
                            displays["strategy"] = replace_name_with_stars(final,full_names)
                        else:
                            print("策略:未使用生成回答" )
                            displays["strategy"] = display
                        strategy_html = f"""
                            <div style='flex: 1;  border: 1px solid #ccc; padding: 1em; border-radius: 8px; background-color: #f9f9f9;'>
                                <p style= 'font-size:20px;'>-----支持策略參考----</p>
                                <pre>{html.escape(displays["strategy"])}</pre>
                                <p style='color:red; font-weight:bold;'>如果覺得回答有落差，可以改變寫法或寫下更多資訊，以提高系統回應的準確度。</p>
                            </div>
                            """

            progress(1.0, desc="完成")
            return f"""

                    <h3 style='margin-bottom:0.5em;'>🤖 系統回應</h3>
                    <h4>回答內容越前面的跟輸入文字越相關。</h4>
                    <h4> ** 可替換成服務對象名字</h4>
                    <div style='display: flex; gap: 1em;'>
                        <div style='flex: 1;   border: 1px solid #ccc; padding: 1em; border-radius: 8px; background-color: #f9f9f9;'>
                            <p style= 'font-size:20px;'>-----短程目標參考----</p>
                            <pre>{html.escape(displays["short"])}</pre>
                            <p style='color:red; font-weight:bold;'>如果覺得回答有落差，可以改變寫法或寫下更多資訊，以提高系統回應的準確度。</p>
                        </div>

                        {strategy_html}
                    </div>
                    """
        else:
            return f"""<h3 style='margin-bottom:0.5em;'>🤖 系統回應</h3>
                        <h4 style='color:red'>系統參數不足</h4>
                    """
    else:
        return f"""
                <h3 style='margin-bottom:0.5em;'>🤖 系統回應</h3>
                <pre>目標內容為空</pre>
            """

    # q = question.strip()
    # result = qa_chain.invoke({"question":q})
    # print(result["answer"])
    # print("\n======參考資料=======")
    # for doc in result["source_documents"]:
    #     print(f"- 來源: {doc.metadata.get('source','未知')}")
    #     print(f"- 內容: {doc.page_content}...\n")

def get_active_domain(tab_name, adult, child, social):
    if tab_name == "成人":
        return adult
    elif tab_name == "兒童":
        return child
    elif tab_name == "社工":
        return social
    else:
        return ""
    
def get_search_types_by_tab(tab_name):
    if tab_name in ["成人", "社工"]:
        return ["短程目標","策略"]
    elif tab_name == "兒童":
        return ["短程目標"]
    return []




def prepare_inputs(tab_name, adult, child, social, goal):
    if tab_name == "成人":
        return ask(adult, goal, ["短程目標", "策略"])
    elif tab_name == "兒童":
        return ask(child, goal)
    elif tab_name == "社工":
        return ask(social, goal, ["短程目標", "策略"])
    else:
        return "無法識別的選擇"


def clear_all_outputs():
    return [gr.update(value=""), gr.update(value="")]

# def show_adult_domain(options):
#     if options == "教保":
#         return gr.update(visible=True,interactive=True,choices=adult_domain_options,value=adult_domain_options[0])
#     return gr.update(visible=True,interactive=True,choices=social_domain_options,value=social_domain_options[0])
#
# def show_child_domain(options):
#     if options == "教保":
#         return gr.update(visible=True, interactive=True, choices=child_domain_options,value=child_domain_options[0])
#     return gr.update(visible=True,interactive=True,choices=social_domain_options,value=social_domain_options[0])


with gr.Blocks(css=CUSTOM_CSS) as demo:
    loading_html = gr.HTML("", visible=False)
    
    def show_loading(text="處理中，請稍候..."):
        return gr.update(
            value=f"""
            <div class="loading-overlay">
                <div class="loading-card">
                    <div class="spinner"></div>
                    <div>{text}</div>
                </div>
            </div>
            """,
            visible=True
        )

    def hide_loading():
        return gr.update(value="", visible=False)    

    def ui_search_candidates(tab_name, adult, child, social, ability, limitation, need, ability_weight, limitation_weight, need_weight):
        domain = get_active_domain(tab_name, adult, child, social)
        
        #structured_text = build_structured_text(ability, limitation, need)
        
        short_update = gr.update(choices=[], value=[], visible=True)
        strategy_update = gr.update(choices=[], value=[], visible=False)

        if not domain or (not ability and not limitation and not need):
            return [], [], short_update, strategy_update, gr.update(visible=False), gr.update(visible=False)
        
        # def build_query():
        #     parts = []
        #     if need:
        #         parts.append(need)
        #     if limitation:
        #         parts.append(limitation)
        #     if ability:
        #         parts.append(ability)
        #     return " ".join(parts)
        
        # query = build_query()

        if tab_name in ["成人", "社工"]:
            short_candidates, short_update = search_childidates_by_type(
                domain, "短程目標", ability, limitation, need, ability_weight, limitation_weight, need_weight, topk=6    
            )  
            strategy_candidates, strategy_update = search_childidates_by_type(
                domain, "策略", ability, limitation, need, ability_weight, limitation_weight, need_weight, topk=6
            )
            
            return (
                short_candidates,
                strategy_candidates,
                short_update,
                strategy_update,
                gr.update(visible=True),
                gr.update(visible=True)    
            )
        elif tab_name == '兒童':
            short_candidates, short_update = search_childidates_by_type(
                domain, "短程目標", ability, limitation, need, ability_weight, limitation_weight, need_weight, topk=6
            )
            return(
                short_candidates,
                [],
                short_update,
                gr.update(choices=[], value=[], visible=False),
                gr.update(visible=True),   # 🔥 顯示 short_section
                gr.update(visible=False)    # 🔥 顯示 strategy_section（視情況）
            )           

            

    def ui_generate_selected_results(tab_name, adult, child, social, ability, limitation, need, selected_short_items, short_candidates, selected_strategy_items, strategy_candidates):
        domain = get_active_domain(tab_name, adult, child, social)
        
        structured_text = build_structured_text(ability, limitation, need)
        
        short_context = build_selected_context(selected_short_items, short_candidates)
        strategy_context = build_selected_context(selected_strategy_items, strategy_candidates)
        
        #======== 情境1:只選短程 =======
        if short_context and not strategy_context:
            messages = build_case1_prompt(structured_text, short_context)
            
        #======== 情境2:短程 + 策略======
        elif short_context and strategy_context:
            messages = build_case2_prompt(structured_text, short_context, strategy_context)
            
        #======== 情境3:都沒選======
        else:
            messages = build_case3_prompt(structured_text)
            
        response = llm.invoke(messages)
        result = response.content.strip()
        
        html_result = f"""
        <h3>🤖 生成結果</h3>
        <div style="
            border:1px solid #ccc;
            padding:1em;
            border-radius:8px;
            background:#f9f9f9;
        ">
            <pre>{html.escape(result)}</pre>
        </div>
        """
        return(html_result, result, gr.update(visible=True))
    
    def build_case1_prompt(structured_text, short_context):
        return[
            ("system", "你是特殊教保員"),
            ("user", f"""
                {structured_text}

                短程目標:
                {short_context}

                請為每一個短程目標生成2~3個支持策略。
                
                每個策略需包含：
                - 行為
                - 條件
                - 頻率

                條列輸出
            """)    
        ]
    
    def build_case2_prompt(structured_text, short_context, strategy_context):
        return[
            ("system", "你是特殊教保員"),
            ("user", f"""
                {structured_text}

                短程目標:
                {short_context}
                
                策略:
                {strategy_context}
                
                請整合並優化：

                - 每個短程目標
                - 對應策略（需可評量 + 頻率）
                
                條列輸出
            """)    
        ]
    def build_case3_prompt(structured_text):
        return[
            ("system", "你是特殊教保員"),
            ("user", f"""
                【使用者填答內容】
                {structured_text}

                使用者沒有勾選任何短程目標或策略參考資料，因此請不要假設或引用任何檢索資料。

                請根據使用者填答內容自行生成：

                1. 至少 5 個短程目標。
                2. 每個短程目標對應 2 至 3 個支持策略。
                3. 每個策略必須可評量，且包含：
                    - 情境條件
                    - 可觀察行為
                    - 達成標準
                    - 頻率或次數
                4. 不要寫前言、結論或說明。
                5. 只輸出條列內容。
            """)    
        ]
    
    def refine_answer(original, refine_prompt):
        messages = [
            ("system", "你是一位特教機構的專業教保員，請根據使用者要求重新調整內容。"),
            ("user", f"""

                【原本回答】
                {original}
                
                【使用者希望調整的方向】
                {refine_prompt}

                請重新生成更符合需求的版本。

                規則：
                1. 使用繁體中文
                2. 保持條列格式
                3. 不要寫前言或結論
                4. 策略需可評量且包含頻率
                """
                    )
        ]
        refined = llm.invoke(messages).content
        html_result = f"""
            <h3>🤖 重新生成結果</h3>
            
            <div style="
                border:1px solid #ccc;
                padding:1em;
                border-radius:8px;
                background:#f9f9f9;
            ">
                <pre>{html.escape(refined)}</pre>
            </div>
            """
        return (
            html_result,
            refined,
            gr.update(visible=True)
        )
    
    gr.Markdown("# 支援ISP/IFSP問答系統")
    current_tab = gr.State("成人")
    tabs = gr.Tabs()
    with tabs:
        with gr.Tab("成人") as adult_tab:
            # adult_domain_radio= gr.Radio(visible=True,interactive=True,choices=radio_options,label="請選擇身份",value=radio_options[0])
            adult_domain = gr.Dropdown(visible=True,interactive=True,choices=adult_domain_options, label="領域")
        with gr.Tab("兒童") as child_tab:
            # child_domain_radio = gr.Radio(visible=True,interactive=True,choices=radio_options, label="請選擇身份",value=radio_options[0])
            child_domain = gr.Dropdown(visible=True,interactive=True,choices=child_domain_options, label="領域")
        with gr.Tab("社工") as social_tab:
            social_domain = gr.Dropdown(visible=True,interactive=True,choices=social_domain_options, label="領域")

    adult_tab.select(lambda: "成人", outputs=current_tab)
    child_tab.select(lambda: "兒童", outputs=current_tab)
    social_tab.select(lambda: "社工", outputs=current_tab)
    
    gr.Markdown("## 搜尋的權重設定")
    with gr.Row():
        ability_weight = gr.Slider(
            interactive = True,
            minimum = 0.5,
            maximum = 3.0,
            value = 1.0,
            label="能力權重"
        )
        limitation_weight = gr.Slider(
            interactive = True,
            minimum = 0.5,
            maximum = 3.0,
            value = 1.2,
            label="限制權重"
        )
        need_weight = gr.Slider(
            interactive = True,
            minimum = 0.5,
            maximum = 3.0,
            value = 2.0,
            label="需求權重"
        )

    ability = gr.Textbox(label="A. 能力（個案目前可以做到的）", lines=2)
    limitation = gr.Textbox(label="B. 限制（目前的困難或不足）", lines=2)
    need = gr.Textbox(label="C. 需求（需要學習或改善的方向）", lines=2)    
    
    short_candidate_state = gr.State([])
    strategy_candidate_state = gr.State([])
    
    gr.Markdown("## 🧭 步驟 1：搜尋參考資料")
    
    search_btn = gr.Button("🔍 搜尋參考資料(請按我)", variant="primary")
   
    
    

    with gr.Column(visible=False) as short_section:
        gr.Markdown("## 🧭 步驟 2：勾選資料")
        short_title = gr.Markdown("""
        <div style="
            background-color:#fff7ed;
            padding:10px;
            border-radius:8px;
            border:1px solid #f59e0b;
            font-size:18px;
            font-weight:bold;
            color:#b45309;
        ">
        📌 請勾選要納入生成的短程目標參考資料
        </div>
        """)
        short_checkbox = gr.CheckboxGroup(   
            choices=[],
            interactive=True, 
            elem_classes=["clean-checkbox"]     
        )
    with gr.Column(visible=False) as strategy_section:
        gr.Markdown("""
            <div style="
                background-color:#eff6ff;
                padding:10px;
                border-radius:8px;
                border:1px solid #3b82f6;
                font-size:18px;
                font-weight:bold;
                color:#1d4ed8;
            ">
            📌 請勾選要納入生成的策略參考資料
            </div>
            """
        )
        strategy_checkbox = gr.CheckboxGroup( 
            choices=[],
            interactive=True,
            elem_classes=["clean-checkbox"]  
        )
    
        gr.Markdown("## 🧭 步驟 3：生成結果")
        generate_btn = gr.Button("✨ AI回答(請按我)",variant="primary")
        output = gr.HTML(label="系統回應")
        answer_state = gr.State("") 

    with gr.Column(visible=False) as refine_section:
        
        gr.Markdown("## 🔧 不滿意AI生成的結果？")
        refine_prompt = gr.Textbox(
            label="請輸入希望調整的方向",
            lines=3,
            placeholder="例如：希望更強調生活自理、希望策略更具體..."
        )
    
        refine_btn = gr.Button(
            "🔄 重新生成",
            variant="secondary"
        )
        
    refine_btn.click(
        fn=refine_answer,
        inputs=[
            answer_state,
            refine_prompt,
        ],
        outputs=[
            output,
            answer_state,
            refine_section    
        ]    
    )
    
    search_btn.click(
        fn=lambda: show_loading("正在搜尋資料..."),
        outputs=loading_html
    ).then(
        fn=ui_search_candidates,
        inputs=[current_tab, adult_domain, child_domain, social_domain, ability, limitation, need, ability_weight, limitation_weight, need_weight ],
        outputs=[short_candidate_state, strategy_candidate_state, short_checkbox, strategy_checkbox, short_section, strategy_section],    
    ).then(
        fn=hide_loading,
        outputs=loading_html    
    )
    
    generate_btn.click(
        fn=lambda: show_loading("AI正在生成..."),
        outputs=loading_html
    ).then(
        fn=ui_generate_selected_results,
        inputs=[
            current_tab,
            adult_domain,
            child_domain,
            social_domain,
            ability,
            limitation,
            need,
            short_checkbox,
            short_candidate_state,
            strategy_checkbox,
            strategy_candidate_state    
        ],
        outputs=[output, answer_state, refine_section]    
    ).then(
        fn=hide_loading,
        outputs=loading_html    
    )

    for tab in [adult_tab, child_tab, social_tab]:
        tab.select(
            fn=lambda:(
                "",
                "",
                "",
                "",
                [],
                [],
                gr.update(choices=[], value=[]),
                gr.update(choices=[], value=[]),
                gr.update(visible=False),
                gr.update(visible=False),    
            ),
            outputs=[
                ability,
                limitation,
                need,
                output, 
                short_candidate_state, 
                strategy_candidate_state, 
                short_checkbox, 
                strategy_checkbox,
                short_section,
                strategy_section,
            ],
        )


        
# app, local_url, share_url = demo.launch(
#     server_name="0.0.0.0",
#     server_port=7860,
#     ssl_certfile="localhost.pem",
#     ssl_keyfile="localhost-key.pem",
#     prevent_thread_lock=True
# )
#
# @app.middleware("http")
# async def allow_iframe(request, call_next):
#     response = await call_next(request)
#     response.headers["X-Frame-Options"] = "ALLOWALL"
#     return response
if __name__ == "__main__":
    import json
    
    
    # uvicorn.run(app,host="0,0,0,0",port=7860)
    #get_model()
    
    # with open("export_jsonl/teacher_full_8b_cpp.jsonl", "r", encoding="utf-8") as f:
    #     for i, line in enumerate(f):
    #         ex = json.loads(line)
    #         if "BMI" in ex["messages"][2]["content"]:
    #             # 用 teacher prompt 測試 student
    #             prompt = ex["messages"][1]["content"]
    #             inputs = TOK(prompt, return_tensors="pt").to(MODEL.device)
    #             outputs = MODEL.generate(**inputs, max_new_tokens=300)
    #             print(TOK.decode(outputs[0], skip_special_tokens=True))
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
    # results = collection.get()
    # for i in range(len(results['ids'])):
    #     print(f"🧾 編號: {results['ids'][i]}")
    #     print(f"📄 文件: {results['documents'][i][:100]}...")  # 顯示前 100 字
    #     print(f"📚 Metadata: {results['metadatas'][i]}")
    #     print("-" * 40)

    # q = input("請輸入指令(a:問題 / k: 知識 / e:離開)：")
    # if q.lower() == "e":
    #     break
    # elif q.startswith("a"):
    #     ask(q[2:].strip())
    #     # query = "如何避免骨質流失"
    #     #
    #     #
    #     # retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"filter":{"domain":"身體福祉"},"k": 50})
    #     # docs = retriever.get_relevant_documents(query)
    #     # # docs = vectorstore.similarity_search_with_score(query,k=50)
    #     #
    #     #
    #     # print(f"針對查詢『{query}』，retriever 找到的文件內容：\n")
    #     #
    #     # for i, doc in enumerate(docs):
    #     #     print(f"--- 文件 {i + 1} ---")
    #     #     print(f"來源：{doc.metadata.get('source', '無來源資訊')}")
    #     #     print(doc.page_content)
    #     #     print()
    # elif q.startswith("k"):
    #     content = q[2:].strip()
    #     add_knowledge(content)
    # else:
    #     print("格式錯誤")
