import streamlit as st
from together import Together
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------- UI CONFIG ----------------
BOT_NAME = "AmirBot"
BOT_ICON = "🤖"

st.set_page_config(page_title=BOT_NAME, page_icon=BOT_ICON, layout="centered")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 900px;}
.stChatMessage {border-radius: 14px; padding: 6px 10px;}
[data-testid="stChatInput"] textarea {border-radius: 14px;}
/* Small pill style */
.model-pill {display:inline-block;padding:6px 10px;border-radius:999px;border:1px solid #ddd;font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

# ---------------- DOC ----------------
DOCUMENT = """
Our software allows users to generate automated reports.
Reports can be exported as PDF or CSV.
Admins can manage user permissions.
""".strip()

# ---------------- HELPERS ----------------
def chunk_text_words(text, chunk_size=80, overlap=20):
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
        kept.append(t); kept_sets.append(w)
    return [t for t in original if t in kept]

@st.cache_resource
def build_rag():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = chunk_text_words(DOCUMENT, 80, 20)
    embs = embedder.encode(chunks, convert_to_numpy=True)

    db = chromadb.Client()
    col = db.get_or_create_collection("rag", metadata={"hnsw:space": "cosine"})
    col.add(ids=[str(i) for i in range(len(chunks))], documents=chunks, embeddings=embs.tolist())

    llm = Together(api_key=st.secrets["TOGETHER_API_KEY"])
    return llm, embedder, col

def rag_answer(llm, embedder, col, query, model_name, top_k=5):
    q = embedder.encode([query], convert_to_numpy=True)[0]
    res = col.query(query_embeddings=[q], n_results=top_k)
    ctx = "\n\n---\n\n".join(dedup_near(res["documents"][0]))

    r = llm.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Answer ONLY using the provided context. If missing, say you don't know."},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"},
        ],
        max_tokens=250,
        temperature=0.2,
    )
    return r.choices[0].message.content

# ---------------- SIDEBAR (Model + Controls) ----------------
with st.sidebar:
    st.markdown(f"## {BOT_ICON} {BOT_NAME}")
    st.caption("Private demo for feedback")
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
    st.markdown("**Try these:**")
    st.markdown("- How can users export reports?\n- What can admins do?\n- What is the pricing?")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------- HEADER ----------------
st.title(f"{BOT_ICON} {BOT_NAME}")
st.caption("Ask questions about the document. If it’s not in the doc, I’ll say I don’t know.")
st.markdown(f"<div class='model-pill'>{MODEL_NAME}</div>", unsafe_allow_html=True)

# ---------------- PASSWORD GATE ----------------
pw_required = st.secrets.get("APP_PASSWORD", "")
if pw_required:
    pw = st.text_input("Password", type="password")
    if pw != pw_required:
        st.stop()

# ---------------- RAG INIT ----------------
llm, embedder, col = build_rag()

# ---------------- CHAT ----------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me anything about the document 🙂"}]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about the document…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            ans = rag_answer(llm, embedder, col, prompt, model_name=MODEL_NAME)
        st.markdown(ans)

    st.session_state.messages.append({"role": "assistant", "content": ans})
