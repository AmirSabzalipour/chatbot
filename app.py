import streamlit as st
from together import Together
import chromadb
from sentence_transformers import SentenceTransformer
import time
from pathlib import Path

# ---------------- UI CONFIG ----------------
BOT_NAME = "Orcabot"
BOT_ICON_PATH = "assets/orca.png"

st.set_page_config(page_title=BOT_NAME, page_icon=BOT_ICON_PATH, layout="centered")

st.markdown("""
<style>
/* --- Global background + text --- */
.stApp { background-color: #f0f0f0; color: #000 !important; }
.stApp, .stApp * { color: #000 !important; }

/* Sidebar background (keep sidebar visible!) */
section[data-testid="stSidebar"] { background-color: #e8e8e8 !important; }

/* Make inputs/selects look white (password + chat input + selectbox) */
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

/* Dropdown menu itself */
div[role="listbox"] { background: #ffffff !important; }
div[role="option"] { background: #ffffff !important; color: #000000 !important; }

/* Main container */
.block-container { padding-top: 2rem; max-width: 900px; }

/* Chat bubbles */
.stChatMessage { border-radius: 14px; padding: 6px 10px; }

/* Chat input container */
[data-testid="stChatInput"] { background: transparent !important; }

/* ---- IMPORTANT: Keep header/toolbar so sidebar + arrows stay ---- */
header { 
  background: #f0f0f0 !important;
  box-shadow: none !important;
}

/* Remove only the thin "decoration" strip (often the dark bar) */
[data-testid="stDecoration"] { display: none !important; }

/* Toolbar stays visible but match the page background */
[data-testid="stToolbar"] {
  background: #f0f0f0 !important;
  box-shadow: none !important;
}

/* ---------- Force WHITE backgrounds for Streamlit widgets ---------- */

/* Buttons (New chat / Delete) */
.stButton > button,
button[kind="primary"],
button[kind="secondary"]{
  background: #ffffff !important;
  color: #000000 !important;
  border: 1px solid #cfcfcf !important;
  box-shadow: none !important;
}
.stButton > button:hover{
  background: #f7f7f7 !important;
}

/* Radio items (Chat selector) */
div[role="radiogroup"] label{
  background: #ffffff !important;
  border: 1px solid #cfcfcf !important;
  border-radius: 10px !important;
  padding: 8px 10px !important;
  margin-bottom: 8px !important;
}
div[role="radiogroup"] label:hover{
  background: #f7f7f7 !important;
}

/* Password input container + eye button */
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
</style>
""", unsafe_allow_html=True)

# ---------------- DOC (from file) ----------------
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

# ---------------- HELPERS ----------------
def chunk_text_words(text, chunk_size=120, overlap=30):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunks.append(" ".join(words[start:end]))
        start = max(end - overlap, start + 1)
    return chunks

def dedup_near(texts, overlap_threshold=0.9):
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

@st.cache_resource(show_spinner="🔄 Thinking…")
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

    llm = Together(api_key=st.secrets["TOGETHER_API_KEY"])
    return llm, embedder, col

def rag_answer(llm, embedder, col, query, model_name, top_k=5):
    q = embedder.encode([query], convert_to_numpy=True)[0]
    res = col.query(query_embeddings=[q], n_results=top_k)
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

# ---------------- CHAT HISTORY (multi-session) ----------------
if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "active_session" not in st.session_state:
    sid = str(int(time.time()))
    st.session_state.sessions[sid] = {
        "name": "Chat 1",
        "messages": [{"role": "assistant", "content": "Hi! I am here to help you 🙂"}],
    }
    st.session_state.active_session = sid

def new_chat():
    sid = str(int(time.time()))
    n = len(st.session_state.sessions) + 1
    st.session_state.sessions[sid] = {
        "name": f"Chat {n}",
        "messages": [{"role": "assistant", "content": "Hi! I am here to help you 🙂"}],
    }
    st.session_state.active_session = sid
    st.rerun()

# ---------------- SIDEBAR (ALL-IN-ONE: logo + model + chat history + debug) ----------------
with st.sidebar:
    st.markdown("""
    <style>
    .sidebar-orca img { margin-bottom: -12px !important; }
    .sidebar-title { margin-top: 0px !important; margin-bottom: 0px !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-orca">', unsafe_allow_html=True)
    st.image(BOT_ICON_PATH, width=48)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="sidebar-title"><h2>{BOT_NAME}</h2></div>', unsafe_allow_html=True)
    st.caption("Private demo")
    st.divider()

    st.markdown("### Model")
    MODEL_NAME = st.selectbox(
        "Choose model",
        [
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        ],
        index=0,
        label_visibility="collapsed",
        key="model_select",
    )

    DEBUG = st.toggle("Show retrieved context", value=False, key="debug_toggle")
    st.divider()

    st.subheader("Chat history")
    if st.button("➕ New chat", key="new_chat_btn"):
        new_chat()

    session_ids = list(st.session_state.sessions.keys())[::-1]  # newest first
    chosen = st.radio(
        "Sessions",
        options=session_ids,
        format_func=lambda x: st.session_state.sessions[x]["name"],
        label_visibility="collapsed",
        key="session_radio",
    )
    st.session_state.active_session = chosen

    st.session_state.sessions[chosen]["name"] = st.text_input(
        "Rename chat",
        value=st.session_state.sessions[chosen]["name"],
        key=f"rename_{chosen}",
    )

    if st.button("🗑️ Delete this chat", key="delete_chat_btn"):
        del st.session_state.sessions[chosen]
        if not st.session_state.sessions:
            new_chat()
        else:
            st.session_state.active_session = list(st.session_state.sessions.keys())[-1]
            st.rerun()

# ---------------- HEADER ----------------
st.title(BOT_NAME)

# ---------------- PASSWORD GATE ----------------
pw_required = st.secrets.get("APP_PASSWORD", "")
if pw_required:
    pw = st.text_input(
        "Password",
        type="password",
        placeholder="Enter password…",
        key="password_input_main",
    )
    if pw != pw_required:
        st.stop()

# ---------------- RAG INIT ----------------
llm, embedder, col = build_rag(DOCUMENT)

# Use active session messages
messages = st.session_state.sessions[st.session_state.active_session]["messages"]

# ---------------- CHAT UI ----------------
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
            ans, retrieved_chunks = rag_answer(llm, embedder, col, prompt, model_name=MODEL_NAME)
        st.markdown(ans)

        if DEBUG:
            with st.expander("Retrieved context"):
                st.write(retrieved_chunks)

    messages.append({"role": "assistant", "content": ans})
