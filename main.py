
import os
import re
import pandas as pd
import gradio as gr
from chromadb import PersistentClient
#from fastapi import FastAPI,Query
#from pydantic import BaseModel

# from langchain.llms import LlamaCpp
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory


#from sentence_transformers import SentenceTransformer




base_dir = os.path.abspath(os.path.dirname(__file__))
persist_dir = os.path.join(base_dir, "chroma_db")
embedding_model = SentenceTransformerEmbeddings(model_name="./hf_cache/bge-small-zh-v1.5")
client = PersistentClient(path=persist_dir)
vectorstore = Chroma(
    client=client,
    collection_name="rag_knowledge",
    persist_directory=persist_dir,
    embedding_function=embedding_model,
)

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
)


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



def clear_all_outputs():
    return [gr.update(value=""), gr.update(value="")]
 

def prepare_inputs(tab_name, adult, child, social, goal):
    if tab_name == "成人":
        return ask(adult, goal, ["短程目標", "策略"])
    elif tab_name == "兒童":
        return ask(child, goal)
    elif tab_name == "社工":
        return ask(social, goal, ["短程目標", "策略"])
    else:
        return "無法識別的選擇"

    
def ask(domain, goal, category=None, progress=None):
    strategy_html = ""
    displays = {"short": "", "strategy": ""}
    if category is None:
        category = ["短程目標"]
    if progress:
        progress(0.2, desc="語意分析中...")
    full_names = import_student_names()

    if progress:
        progress(0.5, desc="資料庫檢索中...")
    if goal:
        if len(category) > 0:
            for t in category:
                retriever = vectorstore.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "filter": {"$and": [{"domain": domain}, {"short": t}]},
                        "k": 10,
                        "score_threshold": 0.3,
                    },
                )
                docs = retriever.invoke(goal.strip())
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
                        displays["short"] = display
                    elif t == "策略":
                        displays["strategy"] = display
                        strategy_html = f"""
                            <div style='flex: 1;  border: 1px solid #ccc; padding: 1em; border-radius: 8px; background-color: #f9f9f9;'>
                                <p style= 'font-size:20px;'>-----支持策略參考----</p>
                                <pre>{displays["strategy"]}</pre>
                                <p style='color:red; font-weight:bold;'>如果覺得回答有落差，可以改變寫法或寫下更多資訊，以提高系統回應的準確度。</p>
                            </div>
                            """
            if progress:
                progress(1.0, desc="完成")
            return f"""
                    
                    <h3 style='margin-bottom:0.5em;'>🤖 系統回應</h3>
                    <h4>回答內容越前面的跟輸入文字越相關。</h4>
                    <h4> ** 可替換成服務對象名字</h4>
                    <div style='display: flex; gap: 1em;'>
                        <div style='flex: 1;   border: 1px solid #ccc; padding: 1em; border-radius: 8px; background-color: #f9f9f9;'>
                            <p style= 'font-size:20px;'>-----短程目標參考----</p>
                            <pre>{displays["short"]}</pre>
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

app = gr.Blocks()

with app:
    gr.Markdown("# 支援ISP/IFSP問答系統")
    current_tab = gr.State("成人")
    tabs = gr.Tabs()
    with tabs:
        with gr.Tab("成人") as adult_tab:
            adult_domain = gr.Dropdown(visible=True,interactive=True,choices=adult_domain_options, label="領域")
        with gr.Tab("兒童") as child_tab:
            child_domain = gr.Dropdown(visible=True,interactive=True,choices=child_domain_options, label="領域")
        with gr.Tab("社工") as social_tab:
            social_domain = gr.Dropdown(visible=True,interactive=True,choices=social_domain_options, label="領域")

    adult_tab.select(lambda: "成人", outputs=current_tab)
    child_tab.select(lambda: "兒童", outputs=current_tab)
    social_tab.select(lambda: "社工", outputs=current_tab)
    

    goal = gr.Textbox(label="支持目標內容簡述", lines=2, placeholder="請簡述...")
    # output = gr.Textbox(label="系統回應", interactive=False)
    #switch_button = gr.Checkbox(label="生成",value=False)
    #switch_button.change(fn=switch_toggle, inputs=switch_button)

    question_btn = gr.Button("送出")

    output = gr.HTML(label="系統回應")
    question_btn.click(
        fn=prepare_inputs,
        inputs=[current_tab,  adult_domain, child_domain, social_domain, goal],
        outputs=output,
    )

    for tab in [adult_tab, child_tab]:
        tab.select(fn=clear_all_outputs, outputs=[goal, output])

if __name__ == "__main__":
    port = int(os.environ.get("PORT",8080))
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        )
