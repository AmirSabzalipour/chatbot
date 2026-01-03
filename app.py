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
# ✅ Together usually needs provider-prefixed model ids
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
TOP_K = 5
DEBUG = False


# =========================
# GLOBAL CSS (ChatGPT-like)
# =========================
st.markdown(
    """
<style>
/* Hide Streamlit default chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* App background like ChatGPT */
.stApp { background: #f7f7f8; }

/* ---- Hide Streamlit top-right toolbars / icons ---- */
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

/* Some versions render a header container */
header[data-testid="stHeader"],
div[data-testid="stHeader"]{
  display: none !important;
}

/* Hide sidebar collapse/expand arrow button (different Streamlit versions) */
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

/* Sidebar width */
section[data-testid="stSidebar"] {
  visibility: visible !important;
  transform: none !important;

  width: 200px !important;
  min-width: 200px !important;
  max-width: 200px !important;

  transition: none !important;
  border-right: 1px solid rgba(0,0,0,0.08);

  padding: 0 !important;
  position: relative !important;    /* needed for absolute logo positioning */
  overflow: visible !important;     /* allow negative top if desired */
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

/* Reserve space so sidebar content wouldn't overlap the logo */
section[data-testid="stSidebar"] > div{
  padding-top: 90px !important;
}

/* ---- PRECISE LOGO + FULL-WIDTH LINE ---- */
.sidebar-logo-box{
  position: absolute;
  top: -130px;      /* can be negative if you want */
  left: 80px;
  right: 80px;
  z-index: 9999;
  padding: 0 !important;
  margin: 0 !important;
}

/* Logo image itself */
.sidebar-logo-img{
  width: 70px;
  height: auto;
  display: block !important;
  margin-left: 22px !important;   /* move logo horizontally */
  margin-top: 0px !important;     /* move logo vertically */
}

/* Full-width divider line under the logo */
.sidebar-logo-box::after{
  content: "";
  display: block;
  width: 100%;
  height: 1px;
  background: rgba(0,0,0,0.15);
  margin-top: 16px;
  margin-bottom: 12px;
}

/* =========================
   TOP BAR (main area)
   ========================= */
.topbar{
  position: sticky;
  top: 0;
  z-index: 1000;
  background: #ffffff;
  border-bottom: 1px solid rgba(0,0,0,0.10);
  height: 56px;
  display: flex;
  align-items: center;
}

.topbar-row{
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10 18px;
}

.topbar-left{
  display:flex;
  align-items:center;
  gap: 8px;
  color: #111827;
  font-size: 18px;
  font-weight: 600;
  line-height: 1;
  max-width: 70%;
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
   MAIN PANEL (ChatGPT-like card)
   ========================= */
.block-container{
  max-width: 1100px !important;
  margin: 18px auto 110px auto !important;
  padding: 18px 22px 26px 22px !important;

  background: #ffffff !important;
  border: 1px solid rgba(0,0,0,0.08) !important;
  border-radius: 26px !important;
  box-shadow: 0 10px 28px rgba(0,0,0,0.08) !important;
}

/* Chat message spacing */
div[data-testid="stChatMessage"]{
  padding: 0.35rem 0 !important;
}

/* Floating/sticky input inside the panel */
div[data-testid="stChatInput"]{
  position: sticky !important;
  bottom: 16px !important;
  z-index: 50 !important;
  background: transparent !important;
  border-top: 0 !important;
  padding: 0 !important;
}

/* Inner input wrapper becomes the rounded bar */
div[data-testid="stChatInput"] > div{
  background: #ffffff !important;
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
# SIDEBAR (logo only, base64 HTML)
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

    # ✅ show actual error details if Together rejects the request
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
        # This gives you the real reason (bad model name, permissions, etc.)
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
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me about the document."}]
messages = st.session_state.messages


# =========================
# TOP BAR (main area)
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
# MAIN CHAT
# =========================
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
