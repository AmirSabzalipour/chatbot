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
/* Page + text */
.stApp { background-color: #f0f0f0; color: #000 !important; }
.stApp, .stApp * { color: #000 !important; }

/* Sidebar background */
section[data-testid="stSidebar"] { background-color: #e8e8e8 !important; }

/* Make inputs/selects look white (password + chat input + selectbox) */
input, textarea { 
  background: #ffffff !important; 
  color: #000000 !important; 
  border: 1px solid #cfcfcf !important;
}
div[data-baseweb="select"] > div {
  background: #ffffff !important;
  color: #000000 !important;
  border: 1px solid #cfcfcf !important;
}

/* Fix dropdown menu itself */
div[role="listbox"] { background: #ffffff !important; }
div[role="option"] { background: #ffffff !important; color: #000000 !important; }

/* Main container */
.block-container { padding-top: 2rem; max-width: 900px; }

/* Chat bubbles */
.stChatMessage { border-radius: 14px; padding: 6px 10px; }

/* Chat input container (sometimes stays dark without this) */
[data-testid="stChatInput"] {
  background: transparent !important;
}
</style>
""", unsafe_allow_html=True)



# ---------------- DOC (from file) ----------------
DOC_PATH = Path("data/document.txt")

@st.cache_data
def load_document(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").strip()

DOCUMENT = load_document(str(DOC_PATH))

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

# ---------------- SIDEBAR (Model + Debug) ----------------
with st.sidebar:
    st.markdown("""
    <style>
    /* Reduce space under the sidebar image and above the title */
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
    )

    st.divider()
    DEBUG = st.toggle("Show retrieved context", value=False)

# ---------------- HEADER ----------------
st.title(BOT_NAME)

# ---------------- PASSWORD GATE ----------------
pw_required = st.secrets.get("APP_PASSWORD", "")
if pw_required:
    pw = st.text_input("Password", type="password", placeholder="Enter password…")
    if pw != pw_required:
        st.stop()

    

    pw = st.text_input("Password", type="password", placeholder="Enter password…")
    if pw != pw_required:
        st.stop()

# ---------------- RAG INIT (rebuilds when document changes) ----------------
llm, embedder, col = build_rag(DOCUMENT)

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
        "messages": [{"role": "assistant", "content": "Hi! Ask me anything about the document 🙂"}],
    }
    st.session_state.active_session = sid
    st.rerun()

with st.sidebar:
    st.subheader("Chat history")
    if st.button("➕ New chat"):
        new_chat()

    session_ids = list(st.session_state.sessions.keys())[::-1]  # newest first
    chosen = st.radio(
        "Sessions",
        options=session_ids,
        format_func=lambda x: st.session_state.sessions[x]["name"],
        label_visibility="collapsed",
    )
    st.session_state.active_session = chosen

    st.session_state.sessions[chosen]["name"] = st.text_input(
        "Rename chat",
        value=st.session_state.sessions[chosen]["name"],
    )

    if st.button("🗑️ Delete this chat"):
        del st.session_state.sessions[chosen]
        if not st.session_state.sessions:
            new_chat()
        else:
            st.session_state.active_session = list(st.session_state.sessions.keys())[-1]
            st.rerun()

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
