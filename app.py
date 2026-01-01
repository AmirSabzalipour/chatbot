import time
from pathlib import Path

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from together import Together


# ---------------- BASIC APP ----------------
st.set_page_config(page_title="Chatbot", layout="wide")

# --- ChatGPT-like CSS ---
st.markdown(
    """
<style>
/* Hide Streamlit default chrome */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* App background like ChatGPT */
.stApp {
  background: #f7f7f8;
}

/* Sidebar pinned open */
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] {
  visibility: visible !important;
  transform: none !important;
  width: 320px !important;
  min-width: 320px !important;
  max-width: 320px !important;
  transition: none !important;
  border-right: 1px solid rgba(0,0,0,0.08);
}

/* Center main content column like ChatGPT */
.block-container {
  max-width: 1050px;
  padding-top: 1.2rem;
  padding-bottom: 7.5rem; /* leave room for sticky input */
}

/* Make chat messages breathe a bit */
div[data-testid="stChatMessage"] {
  padding: 0.35rem 0;
}

/* Sticky chat input bar */
div[data-testid="stChatInput"] {
  position: sticky;
  bottom: 0;
  background: #f7f7f8;
  padding-top: 0.75rem;
  padding-bottom: 1rem;
  border-top: 1px solid rgba(0,0,0,0.08);
  z-index: 50;
}

/* Round the input like a pill */
div[data-testid="stChatInput"] textarea {
  border-radius: 18px !important;
  padding: 0.85rem 1rem !important;
}

/* Make buttons in sidebar full-width */
section[data-testid="stSidebar"] button {
  width: 100%;
}

/* Optional: tighten title spacing */
h1 { margin-top: 0.2rem; }
</style>
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

# ---------------- STATE (simple chat sessions) ----------------
if "chats" not in st.session_state:
    # each chat is: {"title": str, "messages": [...]}
    st.session_state.chats = [{
        "title": "New chat",
        "messages": [{"role": "assistant", "content": "Hi! Ask me about the document."}]
    }]
if "active_chat" not in st.session_state:
    st.session_state.active_chat = 0

def new_chat():
    st.session_state.chats.insert(0, {
        "title": "New chat",
        "messages": [{"role": "assistant", "content": "Hi! Ask me about the document."}]
    })
    st.session_state.active_chat = 0

active = st.session_state.active_chat
messages = st.session_state.chats[active]["messages"]

# ---------------- SIDEBAR (ChatGPT-like left panel) ----------------
with st.sidebar:
    st.markdown("### Chatbot")

    st.button("➕ New chat", on_click=new_chat)

    st.divider()
    st.markdown("**Settings**")

    MODEL_NAME = st.selectbox(
        "Model",
        [
            "Meta-Llama-3.1-8B-Instruct-Turbo",
            "Meta-Llama-3.1-70B-Instruct-Turbo",
        ],
        index=0,
    )

    TOP_K = st.slider("Top-k chunks", 1, 10, 5)
    DEBUG = st.toggle("Show retrieved context", value=False)

    st.divider()
    st.markdown("**History**")

    # Simple history list
    for i, chat in enumerate(st.session_state.chats[:10]):  # show first 10
        label = chat["title"] if chat["title"] else f"Chat {i+1}"
        if st.button(label, key=f"chat_{i}"):
            st.session_state.active_chat = i
            st.rerun()

    st.divider()
    if st.button("🧹 Clear this chat"):
        st.session_state.chats[st.session_state.active_chat]["messages"] = [
            {"role": "assistant", "content": "Hi! Ask me about the document."}
        ]
        st.rerun()

# ---------------- MAIN CHAT ----------------
st.title("Chatbot")

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
            ans, retrieved_chunks = rag_answer(
                llm, embedder, col, prompt, model_name=MODEL_NAME, top_k=TOP_K
            )
        st.markdown(ans)

        if DEBUG:
            with st.expander("Retrieved context"):
                st.write(retrieved_chunks)

    messages.append({"role": "assistant", "content": ans})

    # Update chat title after first user message
    if st.session_state.chats[active]["title"] == "New chat":
        st.session_state.chats[active]["title"] = prompt[:28] + ("…" if len(prompt) > 28 else "")
