import time
from pathlib import Path
import base64

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from together import Together


# ---------------- BASIC APP ----------------
st.set_page_config(page_title="Chatbot", layout="wide")

# ---------------- DEFAULTS (since sidebar controls are removed) ----------------
MODEL_NAME = "Meta-Llama-3.1-8B-Instruct-Turbo"
TOP_K = 5
DEBUG = False


# ---------------- GLOBAL CSS (ChatGPT-like) ----------------
st.markdown(
    """
<style>
/* Hide Streamlit default chrome */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* App background */
.stApp { background: #f7f7f8; }

/* Hide sidebar collapse/expand arrow button (different Streamlit versions) */
button[data-testid="stSidebarCollapseButton"],
button[aria-label="Collapse sidebar"],
button[aria-label="Expand sidebar"],
button[title="Collapse sidebar"],
button[title="Expand sidebar"],
button[aria-label="Close sidebar"],
button[title="Close sidebar"] {
  display: none !important;
}

/* Sidebar pinned open + width */
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] {
  visibility: visible !important;
  transform: none !important;
  width: 250px !important;
  min-width: 250px !important;
  max-width: 250px !important;
  transition: none !important;
  border-right: 1px solid rgba(0,0,0,0.08);
  padding: 0 !important;
  position: relative; /* IMPORTANT for absolute logo positioning */
}

/* Remove padding/margins on sidebar inner wrappers */
section[data-testid="stSidebar"] > div {
  padding: 0 !important;
  margin: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{
  padding: 0 !important;
  margin: 0 !important;
  gap: 0 !important;
}

/* Reserve space so nothing overlaps the logo (optional but recommended) */
section[data-testid="stSidebar"] > div{
  padding-top: 70px !important; /* adjust if logo is taller */
}

/* ---- PRECISE LOGO POSITIONING ---- */
.sidebar-logo-img{
  position: absolute;
  top: -70px;     /* ✅ change vertical position here */
  left: 70px;   /* ✅ change horizontal position here */
  width: 60px;  /* ✅ logo size */
  height: auto;
  z-index: 9999;
  margin: 0 !important;
  padding: 0 !important;
  display: block !important;
}

/* Full-width main content */
.block-container{
  max-width: 100% !important;
  padding-top: 0.6rem;
  padding-bottom: 7.5rem;
  padding-left: 0 !important;
  padding-right: 0 !important;
}

/* Chat message spacing */
div[data-testid="stChatMessage"] { padding: 0.35rem 0; }

/* Sticky chat input bar */
div[data-testid="stChatInput"] {
  position: sticky;
  bottom: 0;
  background: #f7f7f8;
  padding-top: 0.75rem;
  padding-bottom: 1rem;
  border-top: 1px solid rgba(0,0,0,0.08);
  z-index: 50;
  left: 0;
  right: 0;
  border-radius: 0 !important;
  box-shadow: none !important;
  padding-left: 24px;
  padding-right: 24px;
}

/* Round the textarea itself */
div[data-testid="stChatInput"] textarea {
  border-radius: 18px !important;
  padding: 0.85rem 1rem !important;
}

/* ---- Top bar (sticky) ---- */
.topbar {
  position: sticky;
  top: 0;
  z-index: 100;
  background: #f7f7f8;
  border-bottom: 1px solid rgba(0,0,0,0.08);
  padding: 0.65rem 0;
}

.topbar-row{
  display:flex;
  align-items:center;
  justify-content:space-between;
  max-width: 100% !important;
  margin:0 auto;
  padding-left: 24px !important;
  padding-right: 24px !important;
}

.topbar-title{
  font-size: 1rem;
  font-weight: 600;
  color: #111827;
  display:flex;
  gap: 0.35rem;
  align-items:center;
}

.topbar-sub{
  font-weight: 500;
  opacity: 0.65;
}

.topbar-actions{
  display:flex;
  gap: 0.6rem;
  align-items:center;
  font-size: 0.95rem;
}

/* Remove "card" styling around main content */
div[data-testid="stAppViewContainer"] > .main,
div[data-testid="stAppViewContainer"] > .main > div {
  background: transparent !important;
  box-shadow: none !important;
  border: 0 !important;
  border-radius: 0 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------- SIDEBAR (logo only, base64 HTML so CSS top/left works) ----------------
def img_to_base64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()

with st.sidebar:
    b64 = img_to_base64("assets/logo.png")
    st.markdown(
        f"""<img class="sidebar-logo-img" src="data:image/png;base64,{b64}" />""",
        unsafe_allow_html=True,
    )


# ---------------- LOAD DOC ----------------
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


# ---------------- RAG HELPERS ----------------
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


# ---------------- INIT ----------------
llm, embedder, col = build_rag(DOCUMENT)

# ---------------- CHAT STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me about the document."}]
messages = st.session_state.messages


# ---------------- TOP BAR (main area) ----------------
st.markdown(
    f"""
<div class="topbar">
  <div class="topbar-row">
    <div class="topbar-title">
      <span>Chatbot</span>
      <span class="topbar-sub">{MODEL_NAME}</span>
    </div>
    <div class="topbar-actions">
      <span style="opacity:.6;">⋯</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ---------------- MAIN CHAT ----------------
for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about the document…")
if prompt:
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            ans, _ = rag_answer(llm, embedder, col, prompt, model_name=MODEL_NAME, top_k=TOP_K)
        st.markdown(ans)

    messages.append({"role": "assistant", "content": ans})
