
import os
import re
import pandas as pd
import uvicorn
from chromadb import PersistentClient
from fastapi import FastAPI,Query
from pydantic import BaseModel

# from langchain.llms import LlamaCpp
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory


from sentence_transformers import SentenceTransformer




base_dir = os.path.abspath(os.path.dirname(__file__))
persist_dir = os.path.join(base_dir, "chroma_db")
_model = None
embedding_model = None

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
)

app=FastAPI()

adult_domain_options = ["身體福祉","情緒福祉","物質福祉","個人發展","自我決策","人際關係","權利","社會融合"]
child_domain_options = ["健康與安全","感官知覺","精細動作","粗大動作","語言溝通","認知","生活自理","社會適應"]
social_domain_options = ["醫療復健輔具","教育安置","經濟功能及福利輔助","親職支持","家庭支持系統(資援連結)"]
students_name = {}


def import_student_names() -> list:
    df = pd.read_excel("學生名單.xlsx")
    return df["姓名"].dropna().astype(str).tolist()


# 替換人名
def replace_name_with_stars(text: str, full_names: list) -> str:
    replaced = text
    name_lookup = {name[1:]: name for name in full_names if len(name) >= 2}

    for full_name in full_names:
        if full_name in replaced:
            replaced = replaced.replace(full_name, "**")

    for name_tail, full_name in name_lookup.items():
        pattern = re.compile(re.escape(name_tail))
        replaced = pattern.sub("**", replaced)

    return replaced




 

def prepare_inputs(tab_name, adult, child, social, goal):
    if tab_name == "成人":
        return ask(adult, goal, ["短程目標", "策略"])
    elif tab_name == "兒童":
        return ask(child, goal)
    elif tab_name == "社工":
        return ask(social, goal, ["短程目標", "策略"])
    else:
        return "無法識別的選擇"

class QueryRequest(BaseModel):
    domain:str
    goal:str
    category:list[str]
    


@app.on_event("startup")
def load_model():
    global _model, embedding_model, vectorstore
    
    _model = SentenceTransformer("./hf_cache/bge-small-zh-v1.5")
    embedding_model = SentenceTransformerEmbeddings(model_name="./hf_cache/bge-small-zh-v1.5")
    client = PersistentClient(path=persist_dir)
    vectorstore = Chroma(
        client=client,
        collection_name="rag_knowledge",
        persist_directory=persist_dir,
        embedding_function=embedding_model,
    )
    



@app.post("/query")
def ask(req:QueryRequest):
    displays = {"short": "", "strategy": ""}
    full_names = import_student_names()

    if req.goal:
        if len(req.category) > 0:
            for t in req.category:
                retriever = vectorstore.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "filter": {"$and": [{"domain": req.domain}, {"short": t}]},
                        "k": 10,
                        "score_threshold": 0.3,
                    },
                )
                docs = retriever.invoke(req.goal.strip())
                display:str=f""""""
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
                        displays["short"] = display
                    elif t == "策略":
                        displays["strategy"] = display
            return displays



if __name__ == "__main__":
    print(123)
    port = int(os.environ.get("PORT",8080))
    uvicorn.run("main:app",host="0.0.0.0", port=port)
