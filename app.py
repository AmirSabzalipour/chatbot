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

# ---------------- RESET CHAT (prevents old code blocks showing in chat) ----------------
# One-time init. If you want to force-clear on every refresh, uncomment the next line and remove the "if" block.
# st.session_state["messages"] = [{"role": "assistant", "content": "Hi! Ask me about the document."}]
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi! Ask me about the document."}]


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
  position: relative;          /* required for absolute logo positioning */
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

/* Reserve space so nothing overlaps the logo (adjust if needed) */
section[data-testid="stSidebar"] > div{
  padding-top: 110px !important; /* room for logo + line */
}

/* ---- PRECISE LOGO + LINE (use wrapper) ---- */
.sidebar-logo-box{
  position: absolute;
  top: -100px;   /* can be negative */
  left: 0px;     /* start from left edge of sidebar */
  right: 0px;    /* extend to right edge of sidebar */
  z-index: 9999;
}

/* Logo image size/position */
.sidebar-logo-img{
  width: 40px;
  height: auto;
  display: block !important;
  margin: 0 0 0 70px !important;  /* move logo right */
  padding: 0 !important;
}

/* Full-width line under logo */
.sidebar-logo-box::after{
  content: "";
  display: block;
  width: 100%;
  height: 1px;
  background: rgba(0,0,0,0.15);
  margin-top: 20px;
  margin-bottom: 10px;
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

/* Hide Streamlit top-right toolbar (icons like arrows / fullscreen / open) */
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

/* --- ChatGPT-like top header --- */
.topbar {
  position: sticky;
  top: 0;
  z-index: 1000;
  background: #ffffff;
  border-bottom: 1px solid rgba(0,0,0,0.10);
  height: 56px;
  display: flex;
  align-items: center;
}

/* full width row */
.topbar-row{
  width: 100%;
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding: 0 18px;
  min-width: 0; /* important for ellipsis */
}

/* left group: "Chatbot 5.2 Thinking ▾" */
.topbar-left{
  display:flex;
  align-items:center;
  gap: 8px;
  color: #111827;
  font-size: 18px;
  font-weight: 600;
  line-height: 1;
  min-width: 0;          /* allow truncation */
  flex-wrap: nowrap;     /* no wrapping */
  white-space: nowrap;   /* no wrapping */
}

/* model name: truncate instead of wrapping */
.topbar-left .sub{
  font-weight: 500;
  opacity: 0.65;
  font-size: 18px;

  display: inline-block;
  max-width: 420px;      /* adjust if you want */
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* chevron */
.topbar-left .chev{
  font-size: 14px;
  opacity: 0.65;
  transform: translateY(1px);
}

/* right group: Share + ⋯ */
.topbar-right{
  display:flex;
  align-items:center;
  gap: 18px;
  font-size: 16px;
  color:#111827;
}

/* Share button (look like link) */
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

/* 3-dots */
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
      <a class="topbar-share" href="#" onclick="return false;">
        <span style="font-size:18px;">⤴</span>
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
messages = st.session_state.messages

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
