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

/* -----------------------------
   Sidebar: fixed width + no collapse arrows
-------------------------------- */
button[data-testid="stSidebarCollapseButton"],
button[aria-label="Collapse sidebar"],
button[aria-label="Expand sidebar"],
button[title="Collapse sidebar"],
button[title="Expand sidebar"],
button[aria-label="Close sidebar"],
button[title="Close sidebar"],
[data-testid="collapsedControl"] {{
  display: none !important;
}}

section[data-testid="stSidebar"] {{
  visibility: visible !important;
  transform: none !important;

  width: 250px !important;
  min-width: 250px !important;
  max-width: 250px !important;

  transition: none !important;
  border-right: 1px solid rgba(0,0,0,0.08);
  padding: 0 !important;

  position: relative !important;
  overflow: visible !important;
  background: #f7f7f8 !important;
}}

section[data-testid="stSidebar"] > div {{
  padding: 0 !important;
  margin: 0 !important;
}}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
  padding: 0 !important;
  margin: 0 !important;
  gap: 0 !important;
}}

/* Reserve space for logo block */
section[data-testid="stSidebar"] > div {{
  padding-top: 92px !important;
}}

/* Logo + line (full width) */
.sidebar-logo-box {{
  position: absolute;
  top: 14px;     /* <- adjust logo vertical position */
  left: 0;
  right: 0;
  z-index: 9999;
}}

.sidebar-logo-img {{
  width: 44px;
  height: auto;
  display: block;
  margin-left: 22px;  /* <- adjust logo horizontal position */
}}

.sidebar-logo-box::after {{
  content: "";
  display: block;
  width: 100%;
  height: 1px;
  background: rgba(0,0,0,0.15);
  margin-top: 16px;   /* space above line */
  margin-bottom: 12px;/* space below line */
}}

/* -----------------------------
   TOP BANNER: full width, fixed
-------------------------------- */
.topbar {{
  position: fixed;      /* fixed so it stays on top always */
  top: 0;
  left: 0;
  right: 0;
  z-index: 2000;
  height: 56px;

  background: #ffffff;
  border-bottom: 1px solid rgba(0,0,0,0.10);

  display: flex;
  align-items: center;
}}

.topbar-row {{
  width: 100%;
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding: 0 18px;
}}

.topbar-left {{
  display:flex;
  align-items:center;
  gap: 8px;
  font-size: 18px;
  font-weight: 600;
  color: #111827;
  max-width: 75%;
  overflow: hidden;
}}

.topbar-left .sub {{
  font-weight: 500;
  opacity: 0.65;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}

.topbar-left .chev {{
  font-size: 14px;
  opacity: 0.65;
  transform: translateY(1px);
}}

.topbar-right {{
  display:flex;
  align-items:center;
  gap: 16px;
}}

.topbar-more {{
  font-size: 22px;
  opacity: 0.8;
  cursor: pointer;
}}

/* -----------------------------
   MAIN QA PANEL: centered + narrower
   Target the actual block-container that Streamlit creates
-------------------------------- */

/* Remove Streamlit default paddings so we can control layout */
.block-container {{
  max-width: {PANEL_WIDTH_PX}px !important;
  margin: 78px auto 18px auto !important;  /* push below fixed topbar */
  padding: 22px !important;
  
  /* Make the block-container itself the white panel */
  background: #ffffff !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
  border-radius: 26px !important;
  box-shadow: 0 10px 28px rgba(0,0,0,0.08) !important;
  
  height: calc(100vh - 56px - 78px - 18px) !important;
  overflow-y: auto !important;
}}

/* Messages spacing */
div[data-testid="stChatMessage"] {{
  padding: 0.35rem 0 !important;
}}

/* Make the Streamlit chat input look like a rounded floating input */
div[data-testid="stChatInput"] {{
  position: sticky !important;
  bottom: 0 !important;
  padding: 16px 0 0 0 !important;
  background: #ffffff !important;
  border-top: 0 !important;
  margin-left: -22px !important;
  margin-right: -22px !important;
  padding-left: 22px !important;
  padding-right: 22px !important;
  margin-bottom: -22px !important;
  padding-bottom: 22px !important;
}}

div[data-testid="stChatInput"] > div {{
  background: #ffffff !important;
  border: 1px solid rgba(0,0,0,0.10) !important;
  border-radius: 28px !important;
  box-shadow: 0 12px 30px rgba(0,0,0,0.10) !important;
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
# SIDEBAR (logo only)
# =========================
def img_to_base64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()

with st.sidebar:
    b64 = img_to_base64("assets/logo.png")
    st.markdown(
        f"""
        <div class="sidebar-logo-box">
          <img class="sidebar-logo-img" src="data:image/png;base64,{b64}" />
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
# TOP BAR (full width banner)
# =========================
st.markdown(
    f"""
<div class="topbar">
  <div class="topbar-row">
    <div class="topbar-left">
      <span>Chatbot</span>
      <span class="sub">{MODEL_NAME}</span>
      <span class="chev">▾</span>
    </div>
    <div class="topbar-right">
      <span class="topbar-more">⋯</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================
# CHAT MESSAGES (no custom wrappers)
# =========================
for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# =========================
# CHAT INPUT (sticky at bottom)
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
