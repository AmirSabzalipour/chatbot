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
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
TOP_K = 5
DEBUG = False

# =========================
# GLOBAL CSS (ChatGPT-like)
# =========================
st.markdown(
    """
<style>
:root{
  --sidebar-w: 200px;
  --topbar-h: 56px;
}

/* Hide Streamlit default chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* App background like ChatGPT */
.stApp { background: #f7f7f8; }

/* ---- Hide Streamlit toolbars / icons (top-right arrows etc.) ---- */
div[data-testid="stToolbar"],
div[data-testid="stToolbarActions"],
div[data-testid="stToolbarActionButton"],
div[data-testid="stStatusWidget"],
button[title="View fullscreen"],
button[title="Open in new tab"],
button[title="Settings"],
button[title="Rerun"]{
  display: none !important;
}
header[data-testid="stHeader"],
div[data-testid="stHeader"]{
  display: none !important;
}

/* Hide sidebar collapse/expand arrow */
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

/* =========================
   SIDEBAR (logo only)
   ========================= */
section[data-testid="stSidebar"] {
  visibility: visible !important;
  transform: none !important;

  width: var(--sidebar-w) !important;
  min-width: var(--sidebar-w) !important;
  max-width: var(--sidebar-w) !important;

  transition: none !important;
  border-right: 1px solid rgba(0,0,0,0.08);

  padding: 0 !important;
  position: relative !important; /* needed for absolute logo positioning */
  overflow: visible !important;  /* allow negative top */
  background: #f7f7f8 !important;
}

/* Remove padding/margins inside sidebar wrappers */
section[data-testid="stSidebar"] > div {
  padding: 0 !important;
  margin: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{
  padding: 0 !important;
  margin: 0 !important;
  gap: 0 !important;
}

/* Reserve space so nothing overlaps the logo+line */
section[data-testid="stSidebar"] > div{
  padding-top: 110px !important;   /* increase if logo overlaps other stuff */
}

/* ---- PRECISE LOGO + FULL-WIDTH LINE ---- */
.sidebar-logo-box{
  position: absolute;
  top: -30px;     /* ✅ can be negative */
  left: 0px;      /* ✅ full width line starts at left edge */
  right: 0px;     /* ✅ full width line ends at right edge */
  z-index: 9999;
  padding: 0 !important;
  margin: 0 !important;
}

/* Logo image placement (THIS is what you move) */
.sidebar-logo-img{
  width: 44px;
  height: auto;
  display: block !important;

  /* ✅ precise positioning knobs */
  margin-left: 78px !important;   /* + moves right, - moves left */
  margin-top:  0px !important;    /* + moves down, - moves up */
}

/* Divider line under logo (full sidebar width) */
.sidebar-logo-box::after{
  content: "";
  display: block;
  width: 100%;
  height: 1px;
  background: rgba(0,0,0,0.15);
  margin-top: 16px;     /* spacing above line */
  margin-bottom: 12px;  /* spacing below line */
}

/* =========================
   MAIN LAYOUT RESET
   ========================= */
/* IMPORTANT: do NOT make block-container a card (it wraps EVERYTHING) */
.block-container{
  max-width: 100% !important;
  margin: 0 !important;
  padding: calc(var(--topbar-h) + 14px) 0 0 0 !important; /* space below fixed topbar */
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  border-radius: 0 !important;
}

/* =========================
   TOP BAR (fixed header)
   ========================= */
.topbar{
  position: fixed;
  top: 0;
  left: var(--sidebar-w); /* start after sidebar */
  right: 0;
  height: var(--topbar-h);

  z-index: 5000;
  background: #ffffff;
  border-bottom: 1px solid rgba(0,0,0,0.10);

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

.topbar-left{
  display:flex;
  align-items:center;
  gap: 8px;
  color: #111827;
  font-size: 18px;
  font-weight: 600;
  line-height: 1;
  max-width: 75%;
  overflow: hidden;
}

.topbar-left .sub{
  font-weight: 500;
  opacity: 0.65;
  font-size: 18px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.topbar-left .chev{
  font-size: 14px;
  opacity: 0.65;
  transform: translateY(1px);
}

.topbar-right{
  display:flex;
  align-items:center;
  gap: 18px;
  font-size: 16px;
  color:#111827;
}

.topbar-more{
  font-size: 22px;
  opacity: 0.8;
  cursor: pointer;
}

/* =========================
   CHAT CARD (the ONLY container)
   ========================= */
.chat-card{
  width: min(920px, calc(100vw - var(--sidebar-w) - 40px));
  margin: 18px auto 18px auto;
  background: #ffffff;
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 26px;
  box-shadow: 0 10px 28px rgba(0,0,0,0.08);
  padding: 18px 22px 18px 22px;

  /* fixed height so middle area doesn’t become huge */
  height: calc(100vh - var(--topbar-h) - 150px);
  display: flex;
  flex-direction: column;
}

/* Scrollable messages area */
.chat-scroll{
  flex: 1;
  overflow-y: auto;
  padding-right: 6px;
}

/* Chat message spacing */
div[data-testid="stChatMessage"]{
  padding: 0.35rem 0 !important;
}

/* Input sits at bottom of the card */
div[data-testid="stChatInput"]{
  margin-top: 14px !important;
  background: transparent !important;
  border-top: 0 !important;
  padding: 0 !important;
}

/* Input wrapper becomes the rounded bar */
div[data-testid="stChatInput"] > div{
  background: #ffffff !important;      /* ✅ white background */
  border: 1px solid rgba(0,0,0,0.10) !important;
  border-radius: 28px !important;
  box-shadow: 0 12px 30px rgba(0,0,0,0.10) !important;
  padding: 10px 14px !important;
}

/* Textarea itself */
div[data-testid="stChatInput"] textarea{
  border-radius: 20px !important;
  padding: 0.85rem 1rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# SIDEBAR (logo only, base64 so CSS works)
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
        st.error(f"Together request failed.\n\nModel: {model_name}\n\nError: {repr(e)}")
        raise


# =========================
# INIT
# =========================
llm, embedder, col = build_rag(DOCUMENT)


# =========================
# CHAT STATE
# =========================
if "messages" not in st.session_state:
    # ✅ If you want to REMOVE the "Ask me about the document" box, keep this empty list:
    st.session_state.messages = []
messages = st.session_state.messages


# =========================
# TOP BAR (fixed header)
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
# CHAT CARD (the ONLY container)
# =========================
st.markdown('<div class="chat-card">', unsafe_allow_html=True)
st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)

for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

st.markdown('</div>', unsafe_allow_html=True)  # close chat-scroll

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

st.markdown('</div>', unsafe_allow_html=True)  # close chat-card
