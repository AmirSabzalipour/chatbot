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

/* Hide Streamlit native toolbar (two-arrows etc.) */
div[data-testid="stToolbar"],
div[data-testid="stToolbarActions"],
div[data-testid="stToolbarAction"],
div[data-testid="stToolbarActionButton"],
div[data-testid="stHeader"],
header[data-testid="stHeader"],
button[kind="headerNoPadding"],
button[kind="header"]{
  display: none !important;
}

/* App background */
.stApp { background: #f7f7f8; }

/* Hide sidebar collapse/expand arrow button */
button[data-testid="stSidebarCollapseButton"],
button[aria-label="Collapse sidebar"],
button[aria-label="Expand sidebar"],
button[title="Collapse sidebar"],
button[title="Expand sidebar"],
button[aria-label="Close sidebar"],
button[title="Close sidebar"],
[data-testid="collapsedControl"]{
  display: none !important;
}

/* Sidebar pinned open + width */
section[data-testid="stSidebar"] {
  visibility: visible !important;
  transform: none !important;
  width: 250px !important;
  min-width: 250px !important;
  max-width: 250px !important;
  transition: none !important;
  border-right: 1px solid rgba(0,0,0,0.08);
  padding: 0 !important;
  position: relative;        /* for absolute positioning of logo box */
  overflow: visible !important; /* allow negative top without clipping */
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

/* Reserve space so nothing overlaps the logo+line (adjust if needed) */
section[data-testid="stSidebar"] > div{
  padding-top: 90px !important;
}

/* ---- PRECISE LOGO + FULL-WIDTH DIVIDER ---- */
.sidebar-logo-box{
  position: absolute;
  top: -100px;   /* ✅ can be negative */
  left: 0;
  right: 0;      /* ✅ full sidebar width */
  z-index: 9999;
}

/* Logo image position inside the full-width box */
.sidebar-logo-img{
  width: 40px;
  height: auto;
  display: block !important;
  margin: 0 0 0 70px !important; /* ✅ move logo horizontally */
  padding: 0 !important;
}

/* Full width divider line under logo with vertical spacing */
.sidebar-logo-box::after{
  content: "";
  display: block;
  width: 100%;                 /* ✅ from left edge to right edge */
  height: 1px;
  background: rgba(0,0,0,0.15);
  margin-top: 20px;            /* space between logo and line */
  margin-bottom: 12px;         /* space under the line */
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

/* Remove "card" styling around main content */
div[data-testid="stAppViewContainer"] > .main,
div[data-testid="stAppViewContainer"] > .main > div {
  background: transparent !important;
  box-shadow: none !important;
  border: 0 !important;
  border-radius: 0 !important;
}

/* -------- ChatGPT-like TOP BAR -------- */
.topbar{
  position: sticky;
  top: 0;
  z-index: 1000;
  background: #ffffff;                 /* ✅ white like ChatGPT */
  border-bottom: 1px solid rgba(0,0,0,0.10);
  height: 56px;
  display: flex;
  align-items: center;
}

.topbar-row{
  width: 100%;
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding: 0 18px;
}

/* left: title + model + chevron */
.topbar-left{
  display:flex;
  align-items:center;
  gap: 8px;
  color: #111827;
  font-size: 18px;
  font-weight: 600;
  line-height: 1;
}
.topbar-left .sub{
  font-weight: 500;
  opacity: 0.65;
  font-size: 18px;
}
.topbar-left .chev{
  font-size: 14px;
  opacity: 0.65;
  transform: translateY(1px);
}

/* right: Share + dots */
.topbar-right{
  display:flex;
  align-items:center;
  gap: 18px;
  font-size: 16px;
  color:#111827;
}
.topbar-share{
  display:flex;
  align-items:center;
  gap: 8px;
  cursor: pointer;
  user-select: none;
  text-decoration: none;
  color:#111827;
  opacity: 0.9;
}
.topbar-share:hover{ opacity: 1; }

.topbar-more{
  font-size: 22px;
  opacity: 0.8;
  cursor: pointer;
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
        f"""
        <div class="sidebar-logo-box">
          <img class="sidebar-logo-img" src="data:image/png;base64,{b64}" />
        </div>
        """,
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
    <div class="topbar-left">
      <span>Chatbot</span>
      <span class="sub">{MODEL_NAME}</span>
      <span class="chev">▾</span>
    </div>

    <div class="topbar-right">
      <a class="topbar-share" href="#" onclick="navigator.clipboard.writeText(window.location.href); return false;">
        <span style="font-size:18px;">⤴︎</span>
        <span>Share</span>
      </a>
      <span class="topbar-more">⋯</span>
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
