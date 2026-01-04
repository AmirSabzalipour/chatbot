import time
from pathlib import Path
import base64

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from together import Together


# =========================
# BASIC APP
# =========================
st.set_page_config(page_title="Chatbot", layout="wide")

# =========================
# DEFAULTS
# =========================
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"  # Together model id
TOP_K = 5
DEBUG = False

# ✅ Control the width of the scrollable QA panel (smaller than the full-width top banner)
PANEL_WIDTH_PX = 860  # try 760 / 820 / 900


# =========================
# GLOBAL CSS (layout exactly as requested)
# =========================
st.markdown(
    f"""
<style>
/* Hide Streamlit chrome */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

/* Background */
.stApp {{ background: #f7f7f8; }}

/* -----------------------------
   Hide Streamlit toolbar/icons
-------------------------------- */
div[data-testid="stToolbar"],
div[data-testid="stToolbarActions"],
div[data-testid="stToolbarActionButton"],
div[data-testid="stStatusWidget"],
header[data-testid="stHeader"],
div[data-testid="stHeader"] {{
  display: none !important;
}}

/* Try to hide extra floating buttons some versions show */
button[title="View fullscreen"],
button[title="Open in new tab"],
button[title="Rerun"],
button[title="Settings"] {{
  display: none !important;
}}

/* Hide sidebar completely */
section[data-testid="stSidebar"] {{
  display: none !important;
}}

/* -----------------------------
   MAIN QA PANEL: centered + narrower
-------------------------------- */

/* Remove Streamlit default paddings so we can control layout */
.block-container {{
  max-width: {PANEL_WIDTH_PX}px !important;
  margin: 18px auto !important;
  padding: 22px 22px 100px 22px !important;  /* Extra bottom padding for input */
  
  /* Make the block-container itself the white panel */
  background: #ffffff !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
  border-radius: 26px !important;
  box-shadow: 0 10px 28px rgba(0,0,0,0.08) !important;
  
  height: calc(100vh - 36px) !important;
  overflow-y: auto !important;
  position: relative !important;
}}

/* Messages spacing */
div[data-testid="stChatMessage"] {{
  padding: 0.35rem 0 !important;
}}

/* Make the Streamlit chat input stay inside and at bottom */
div[data-testid="stChatInput"] {{
  position: fixed !important;
  bottom: 18px !important;
  left: 50% !important;
  transform: translateX(-50%) !important;
  width: {PANEL_WIDTH_PX - 44}px !important;
  max-width: {PANEL_WIDTH_PX - 44}px !important;
  padding: 0 !important;
  background: transparent !important;
  border-top: 0 !important;
  margin: 0 !important;
  z-index: 100 !important;
}}

div[data-testid="stChatInput"] > div {{
  background: #ffffff !important;
  border: 1px solid rgba(0,0,0,0.10) !important;
  border-radius: 28px !important;
  box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
  padding: 10px 14px !important;
}}

div[data-testid="stChatInput"] textarea {{
  border-radius: 20px !important;
  padding: 0.85rem 1rem !important;
}}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# LOAD DOC
# =========================
DOC_PATH = Path("data/document.txt")

@st.cache_data
def load_document(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8").strip()

DOCUMENT = load_document(str(DOC_PATH))
if not DOCUMENT:
    st.error("Document not found. Please add your file at: data/document.txt")
    st.stop()


# =========================
# RAG HELPERS
# =========================
def chunk_text_words(text: str, chunk_size: int = 120, overlap: int = 30):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        start = max(end - overlap, start + 1)
    return chunks

@st.cache_resource
def build_rag(document_text: str):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    chunks = chunk_text_words(document_text, 120, 30)
    embs = embedder.encode(chunks, convert_to_numpy=True)

    db = chromadb.Client()
    col = db.get_or_create_collection("rag", metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embs.tolist(),
    )

    api_key = st.secrets.get("TOGETHER_API_KEY", "")
    if not api_key:
        st.error("Missing TOGETHER_API_KEY in Streamlit secrets.")
        st.stop()

    llm = Together(api_key=api_key)
    return llm, embedder, col

def rag_answer(llm, embedder, col, query: str, model_name: str, top_k: int = 5):
    q = embedder.encode([query], convert_to_numpy=True)[0]
    res = col.query(query_embeddings=[q], n_results=top_k)
    chunks = res["documents"][0]
    ctx = "\n\n---\n\n".join(chunks)

    try:
        r = llm.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Answer ONLY using the provided context. If missing, say you don't know."},
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"},
            ],
            max_tokens=250,
            temperature=0.2,
        )
        return r.choices[0].message.content, chunks
    except Exception as e:
        # Prevent the whole app crashing; show short error
        return f"⚠️ Model request failed: {e}", chunks


# =========================
# INIT
# =========================
llm, embedder, col = build_rag(DOCUMENT)

# =========================
# CHAT STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me about the document."}]
messages = st.session_state.messages


# =========================
# CHAT MESSAGES
# =========================
for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# =========================
# CHAT INPUT (at bottom of container)
# =========================
prompt = st.chat_input("Ask about the document…")

if prompt:
    messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            ans, retrieved = rag_answer(llm, embedder, col, prompt, model_name=MODEL_NAME, top_k=TOP_K)
        st.markdown(ans)

        if DEBUG:
            with st.expander("Retrieved context"):
                st.write(retrieved)

    messages.append({"role": "assistant", "content": ans})
    st.rerun()
