import streamlit as st
from together import Together
import chromadb
from sentence_transformers import SentenceTransformer
import time
from pathlib import Path
import base64
import json
from datetime import datetime
import re

# ---------------- UI CONFIG ----------------
BOT_NAME = "Orcabot"
BOT_ICON_PATH = "assets/orca.png"

st.set_page_config(page_title=BOT_NAME, page_icon=BOT_ICON_PATH, layout="centered")

st.markdown(
    """
<style>
/* =========================
   GLOBAL PAGE LOOK
========================= */
html, body { background: #f0f0f0 !important; }

.stApp { background-color: #f0f0f0; color: #000 !important; }
.stApp, .stApp * { color: #000 !important; }

/* Main Streamlit view containers (kills the black bottom strip) */
div[data-testid="stAppViewContainer"] { background: #f0f0f0 !important; }
div[data-testid="stAppViewBlockContainer"] { background: #f0f0f0 !important; }

/* Bottom/fixed area where chat_input is rendered */
div[data-testid="stBottom"],
div[data-testid="stBottomBlockContainer"] {
  background: #f0f0f0 !important;
}

/* =========================
   SIDEBAR - FIXED & NO HAMBURGER
========================= */
section[data-testid="stSidebar"] { 
  background-color: #e8e8e8 !important; 
  width: 260px !important;
  min-width: 260px !important;
  max-width: 260px !important;
}

/* HIDE ALL HAMBURGER/COLLAPSE BUTTONS */
button[kind="header"] {
  display: none !important;
}
button[data-testid="collapsedControl"] {
  display: none !important;
}
button[title="Collapse sidebar"],
button[title="Hide sidebar"],
button[title="Show sidebar"] {
  display: none !important;
}

/* Prevent sidebar from ever collapsing */
section[data-testid="stSidebar"][aria-expanded="false"] {
  display: block !important;
  margin-left: 0 !important;
}

/* Remove the hamburger icon completely */
.css-1dp5vir, .css-164nlkn {
  display: none !important;
}

/* =========================
   INPUTS / SELECTS (WHITE)
========================= */
input, textarea {
  background: #ffffff !important;
  color: #000000 !important;
  border: 1px solid #cfcfcf !important;
}

/* Selectbox (BaseWeb) */
div[data-baseweb="select"] > div {
  background: #ffffff !important;
  color: #000000 !important;
  border: 1px solid #cfcfcf !important;
}

/* Dropdown menu */
div[role="listbox"] { background: #ffffff !important; }
div[role="option"] { background: #ffffff !important; color: #000000 !important; }

/* =========================
   LAYOUT CONTAINER
========================= */
.block-container {
  padding-top: 2rem;
  max-width: 900px;
  /* Extra space so chat input doesn't overlap Streamlit Cloud "Manage app" */
  padding-bottom: 7rem !important;
}

/* =========================
   CHAT MESSAGES
========================= */
.stChatMessage {
  border-radius: 14px;
  padding: 6px 10px;
  background-color: transparent !important;
}

/* Optional: light styling for user/assistant bubbles */
[data-testid="stChatMessage"][data-testid*="user"] {
  background-color: #e3f2fd !important;
  border: 1px solid #90caf9 !important;
  border-radius: 14px !important;
}
[data-testid="stChatMessage"][data-testid*="assistant"] {
  background-color: #f5f5f5 !important;
  border: 1px solid #e0e0e0 !important;
  border-radius: 14px !important;
}

/* =========================
   CHAT INPUT (WHITE + NO BLACK BOX)
========================= */
/* Force chat input wrapper to be white */
[data-testid="stChatInput"] {
  background-color: #ffffff !important;
  background: #ffffff !important;
  border-radius: 12px !important;
  padding: 10px !important;
  border: 1px solid #cfcfcf !important;
  box-shadow: none !important;

  /* avoid overlap with Streamlit Cloud Manage app button */
  margin-right: 180px !important;
  margin-bottom: 20px !important;
}

/* wrapper div */
[data-testid="stChatInput"] > div:first-child {
  background-color: #ffffff !important;
  background: #ffffff !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}

/* everything inside */
[data-testid="stChatInput"] *,
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] input,
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] div {
  background: #ffffff !important;
  background-color: #ffffff !important;
  box-shadow: none !important;
}

/* input field itself */
[data-testid="stChatInput"] input,
[data-testid="stChatInput"] textarea {
  background: #ffffff !important;
  background-color: #ffffff !important;
  color: #000000 !important;
  border: none !important;
  box-shadow: none !important;
}

/* Streamlit sometimes uses this class */
.stChatInputContainer,
.stChatInputContainer *,
.stChatInputContainer > div {
  background: #f0f0f0 !important;   /* the container behind input */
  box-shadow: none !important;
}

/* =========================
   HEADER / TOOLBAR (KEEP ARROWS)
========================= */
header {
  background: #f0f0f0 !important;
  box-shadow: none !important;
}

/* Remove thin decoration strip that can look like a dark bar */
[data-testid="stDecoration"] { display: none !important; }

/* Keep toolbar, but match background */
[data-testid="stToolbar"] {
  background: #f0f0f0 !important;
  box-shadow: none !important;
}

/* =========================
   BUTTONS / RADIO / TEXT INPUT CONTAINERS (WHITE)
========================= */
/* Buttons */
.stButton > button,
button[kind="primary"],
button[kind="secondary"]{
  background: #ffffff !important;
  color: #000000 !important;
  border: 1px solid #cfcfcf !important;
  box-shadow: none !important;
}
.stButton > button:hover{ background: #f7f7f7 !important; }

/* Radio items */
div[role="radiogroup"] label{
  background: #ffffff !important;
  border: 1px solid #cfcfcf !important;
  border-radius: 10px !important;
  padding: 8px 10px !important;
  margin-bottom: 8px !important;
}
div[role="radiogroup"] label:hover{ background: #f7f7f7 !important; }

/* Text input container + eye button */
div[data-testid="stTextInput"] > div{
  background: #ffffff !important;
  border-radius: 10px !important;
}
div[data-testid="stTextInput"] button{
  background: #ffffff !important;
  border: 0 !important;
}

/* BaseWeb buttons */
div[data-baseweb="button"] button{
  background: #ffffff !important;
  color: #000000 !important;
  border: 1px solid #cfcfcf !important;
}

/* =========================
   SIDEBAR STYLING
========================= */
.sidebar-orca img { margin-bottom: -12px !important; }
.sidebar-title { margin-top: 0px !important; margin-bottom: 0px !important; }

</style>
""",
    unsafe_allow_html=True,
)

# ---------------- UTILITY FUNCTIONS ----------------
def img_to_base64(path: str) -> str:
    """Convert image to base64 string."""
    try:
        return base64.b64encode(Path(path).read_bytes()).decode()
    except Exception as e:
        st.warning(f"Could not load icon: {e}")
        return ""

# ---------------- DOC (from file) ----------------
DOC_PATH = Path("data/document.txt")

@st.cache_data
def load_document(path: str) -> str:
    """Load document from file path."""
    p = Path(path)
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception as e:
        st.error(f"Error reading document: {e}")
        return ""

DOCUMENT = load_document(str(DOC_PATH))
if not DOCUMENT:
    st.error("Document not found. Please add your file at: data/document.txt")
    st.stop()

# ---------------- HELPERS ----------------
def chunk_text_words(text, chunk_size=120, overlap=30):
    """Chunk text by word count with overlap."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        start = max(end - overlap, start + 1)
    return chunks

def dedup_near(texts, overlap_threshold=0.9):
    """Remove near-duplicate chunks while preserving order."""
    original = [x.strip() for x in texts if x and x.strip()]
    candidates = sorted(original, key=len, reverse=True)
    kept, kept_sets = [], []
    for t in candidates:
        w = set(t.lower().split())
        if any((len(w & ws) / max(1, min(len(w), len(ws)))) >= overlap_threshold for ws in kept_sets):
            continue
        kept.append(t)
        kept_sets.append(w)
    return [t for t in original if t in kept]

@st.cache_resource(show_spinner="🔄 Building RAG system…")
def build_rag(document_text: str):
    """Build RAG system with embeddings and vector database."""
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        chunks = chunk_text_words(document_text, 120, 30)
        
        if not chunks:
            return None, None, None
        
        embs = embedder.encode(chunks, convert_to_numpy=True)

        db = chromadb.Client()
        col = db.get_or_create_collection("rag", metadata={"hnsw:space": "cosine"})
        col.add(
            ids=[str(i) for i in range(len(chunks))],
            documents=chunks,
            embeddings=embs.tolist(),
        )

        api_key = st.secrets.get("TOGETHER_API_KEY")
        if not api_key:
            return None, None, None
        
        llm = Together(api_key=api_key)
        return llm, embedder, col
    
    except Exception as e:
        st.error(f"Error building RAG: {e}")
        return None, None, None

def rag_answer(llm, embedder, col, query, model_name, top_k=5):
    """Generate answer using RAG."""
    try:
        q = embedder.encode([query], convert_to_numpy=True)[0]
        res = col.query(query_embeddings=[q.tolist()], n_results=top_k)
        
        if not res["documents"] or not res["documents"][0]:
            return "I couldn't find relevant information in the document.", []
        
        chunks = dedup_near(res["documents"][0])
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
    
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "Sorry, I encountered an error while processing your question.", []

# ---------------- CHAT HISTORY (multi-session) ----------------
if "sessions" not in st.session_state:
    sid = str(int(time.time()))
    st.session_state.sessions = {
        sid: {
            "name": "Chat 1",
            "messages": [{"role": "assistant", "content": "Hi! I am here to help you 🙂"}],
            "created_at": datetime.now().isoformat()
        }
    }

if "active_session" not in st.session_state:
    st.session_state.active_session = list(st.session_state.sessions.keys())[-1]

def new_chat():
    """Create a new chat session."""
    sid = str(int(time.time()))
    n = len(st.session_state.sessions) + 1
    st.session_state.sessions[sid] = {
        "name": f"Chat {n}",
        "messages": [{"role": "assistant", "content": "Hi! I am here to help you 🙂"}],
        "created_at": datetime.now().isoformat()
    }
    st.session_state.active_session = sid
    st.rerun()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    # Logo and title
    logo_b64 = img_to_base64(BOT_ICON_PATH)
    
    if logo_b64:
        st.markdown(
            f"""
            <div style="display:flex; flex-direction:column; align-items:flex-start; gap:6px;">
              <img src="data:image/png;base64,{logo_b64}" style="width:48px; height:auto; margin:0;" />
              <div style="font-size:28px; font-weight:700; margin:0; line-height:1;">{BOT_NAME}</div>
              <div style="margin-top:0px; opacity:0.75;">Private demo</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.title(BOT_NAME)
        st.caption("Private demo")
    
    st.divider()

    # Model settings
    st.markdown("### ⚙️ Settings")
    MODEL_NAME = st.selectbox(
        "Model",
        [
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        ],
        index=0,
        key="model_select",
    )

    DEBUG = st.toggle("Show retrieved context", value=False, key="debug_toggle")
    st.divider()

    # Chat history
    st.markdown("### 💬 Chat History")
    
    if st.button("➕ New chat", key="new_chat_btn", use_container_width=True):
        new_chat()

    session_ids = sorted(
        st.session_state.sessions.keys(), 
        key=lambda x: st.session_state.sessions[x].get("created_at", ""),
        reverse=True
    )
    
    if session_ids:
        chosen = st.radio(
            "Sessions",
            options=session_ids,
            format_func=lambda x: st.session_state.sessions[x]["name"],
            label_visibility="collapsed",
            key="session_radio",
        )
        st.session_state.active_session = chosen

        # Rename chat
        new_name = st.text_input(
            "Rename chat",
            value=st.session_state.sessions[chosen]["name"],
            key=f"rename_{chosen}",
        )
        
        if new_name != st.session_state.sessions[chosen]["name"]:
            st.session_state.sessions[chosen]["name"] = new_name

        # Delete chat
        if st.button("🗑️ Delete this chat", key="delete_chat_btn", use_container_width=True):
            del st.session_state.sessions[chosen]
            if not st.session_state.sessions:
                new_chat()
            else:
                st.session_state.active_session = list(st.session_state.sessions.keys())[-1]
            st.rerun()

# ---------------- HEADER ----------------
st.title(f"💬 {BOT_NAME}")

# ---------------- PASSWORD GATE ----------------
pw_required = st.secrets.get("APP_PASSWORD", "")
if pw_required:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        pw = st.text_input(
            "🔒 Password required",
            type="password",
            placeholder="Enter password…",
            key="password_input_main",
        )
        
        if pw == pw_required:
            st.session_state.authenticated = True
            st.rerun()
        elif pw:
            st.error("Incorrect password")
        
        st.stop()

# ---------------- RAG INIT ----------------
llm, embedder, col = build_rag(DOCUMENT)

if llm is None or embedder is None or col is None:
    st.error("❌ Failed to initialize RAG system. Please check your configuration.")
    st.stop()

# Use active session messages
messages = st.session_state.sessions[st.session_state.active_session]["messages"]

# ---------------- CHAT UI ----------------
for i, m in enumerate(messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
prompt = st.chat_input("Ask about the document…")

if prompt:
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking…"):
            ans, retrieved_chunks = rag_answer(llm, embedder, col, prompt, model_name=MODEL_NAME)
        st.markdown(ans)

        if DEBUG and retrieved_chunks:
            with st.expander(f"📚 Retrieved {len(retrieved_chunks)} context chunks"):
                for idx, chunk in enumerate(retrieved_chunks, 1):
                    st.markdown(f"**Chunk {idx}:**")
                    st.markdown(chunk)
                    if idx < len(retrieved_chunks):
                        st.markdown("---")

    messages.append({"role": "assistant", "content": ans})
