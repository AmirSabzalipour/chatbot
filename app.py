import streamlit as st
from together import Together
import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

DOCUMENT = """
Our software allows users to generate automated reports.
Reports can be exported as PDF or CSV.
Admins can manage user permissions.
""".strip()

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

def rag_answer(llm, embedder, col, query, top_k=5):
    q = embedder.encode([query], convert_to_numpy=True)[0]
    res = col.query(query_embeddings=[q], n_results=top_k)
    ctx = "\n\n---\n\n".join(dedup_near(res["documents"][0]))
    r = llm.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role":"system","content":"Answer ONLY using the provided context. If missing, say you don't know."},
            {"role":"user","content":f"Context:\n{ctx}\n\nQuestion: {query}\nAnswer:"},
        ],
        max_tokens=250,
        temperature=0.2,
    )
    return r.choices[0].message.content

st.title("Private Doc Chatbot")

pw_required = st.secrets.get("APP_PASSWORD", "")
if pw_required:
    if st.text_input("Password", type="password") != pw_required:
        st.stop()

llm, embedder, col = build_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about the document…")
if prompt:
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    ans = rag_answer(llm, embedder, col, prompt)
    st.session_state.messages.append({"role":"assistant","content":ans})
    with st.chat_message("assistant"):
        st.markdown(ans)
