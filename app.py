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
LEFT_PANEL_WIDTH_PX = 280  # width of left panel


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
.stApp {{ 
  background: #f7f7f8; 
  overflow: hidden;
}}

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

/* Hide default sidebar */
section[data-testid="stSidebar"] {{
  display: none !important;
}}

/* -----------------------------
   Main app wrapper
-------------------------------- */
.main {{
  padding: 0 !important;
}}

.main > div {{
  padding: 0 !important;
}}

/* -----------------------------
   LEFT PANEL: absolute positioning inside container
-------------------------------- */
.left-panel {{
  position: absolute;
  top: 18px;
  left: 18px;
  width: {LEFT_PANEL_WIDTH_PX}px;
  height: calc(100vh - 36px);
  background: #ffffff;
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 16px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  padding: 22px;
  overflow-y: auto;
  z-index: 10;
}}

.left-panel h3 {{
  margin: 0 0 16px 0;
  font-size: 16px;
  font-weight: 600;
  color: #111827;
}}

.left-panel ul {{
  list-style: none;
  padding: 0;
  margin: 0;
}}

.left-panel li {{
  padding: 10px 12px;
  margin-bottom: 4px;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.2s;
  font-size: 14px;
  color: #374151;
}}

.left-panel li:hover {{
  background: #f3f4f6;
}}

.left-panel li.active {{
  background: #e5e7eb;
  font-weight: 500;
}}

/* -----------------------------
   MAIN QA PANEL: offset by left panel
-------------------------------- */

/* Container wrapper */
.stApp > div:first-child {{
  position: relative;
  height: 100vh;
  overflow: hidden;
}}

/* Remove Streamlit default paddings so we can control layout */
.block-container {{
  position: absolute !important;
  top: 18px !important;
  left: {LEFT_PANEL_WIDTH_PX + 36}px !important;
  right: 18px !important;
  bottom: 18px !important;
  max-width: none !important;
  width: auto !important;
  margin: 0 !important;
  padding: 22px 22px 100px 22px !important;  /* Extra bottom padding for input */
  
  /* Make the block-container itself the white panel */
  background: #ffffff !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
  border-radius: 16px !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
  
  height: auto !important;
  overflow-y: auto !important;
}}

/* Messages spacing */
div[data-testid="stChatMessage"] {{
  padding: 0.35rem 0 !important;
}}

/* Make the Streamlit chat input stay inside and at bottom */
div[data-testid="stChatInput"] {{
  position: absolute !important;
  bottom: 22px !important;
  left: 22px !important;
  right: 22px !important;
  width: auto !important;
  max-width: none !important;
  padding: 0 !important;
  background: transparent !important;
  border-top: 0 !important;
  margin: 0 !important;
  z-index: 100 !important;
}}

div[data-testid="stChatInput"] > div {{
  background: #ffffff !important;
  border: 1px solid rgba(0,0,0,0.10) !important;
  border-radius: 24px !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
  padding: 8px 12px !important;
}}

div[data-testid="stChatInput"] textarea {{
  border-radius: 20px !important;
  padding: 0.7rem 1rem !important;
  font-size: 14px !important;
}}

/* Scrollbar styling */
::-webkit-scrollbar {{
  width: 8px;
  height: 8px;
}}

::-webkit-scrollbar-track {{
  background: transparent;
}}

::-webkit-scrollbar-thumb {{
  background: #d1d5db;
  border-radius: 4px;
}}

::-webkit-scrollbar-thumb:hover {{
  background: #9ca3af;
}}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# LEFT PANEL
# =========================
st.markdown(
    """
<div class="left-panel">
  <h3>Conversations</h3>
  <ul>
    <li class="active">Current Chat</li>
    <li>Previous Chat 1</li>
    <li>Previous Chat 2</li>
    <li>Previous Chat 3</li>
  </ul>
</div>
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
