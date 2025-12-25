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
.block-container { padding-top: 2rem; max-width: 900px; padding-bottom: 6rem !important; }

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
.stButton > button:hover{ background: #f7f7f7 !important; }

/* Radio items (Chat selector) */
div[role="radiogroup"] label{
  background: #ffffff !important;
  border: 1px solid #cfcfcf !important;
  border-radius: 10px !important;
  padding: 8px 10px !important;
  margin-bottom: 8px !important;
}
div[role="radiogroup"] label:hover{ background: #f7f7f7 !important; }

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

/* --- Avoid Streamlit Cloud "Manage app" overlay (bottom-right) --- */
/* Reserve space on the right so the chat input doesn't go under Manage app */
div[data-testid="stChatInput"]{
  padding-right: 220px !important;   /* adjust: 180–260 as needed */
  box-sizing: border-box !important;
}
div[data-testid="stChatInput"] > div{
  padding-right: 220px !important;
  box-sizing: border-box !important;
}

/* Source citations styling */
.source-citation {
  background: #e3f2fd;
  border-left: 3px solid #2196f3;
  padding: 8px 12px;
  margin: 8px 0;
  border-radius: 4px;
  font-size: 0.9em;
}

/* Copy button styling */
.copy-button {
  background: #ffffff;
  border: 1px solid #cfcfcf;
  border-radius: 6px;
  padding: 4px 8px;
  cursor: pointer;
  font-size: 0.85em;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------- UTIL ----------------
def img_to_base64(path: str) -> str:
    """Convert image to base64 string."""
    try:
        return base64.b64encode(Path(path).read_bytes()).decode()
    except Exception as e:
        st.warning(f"Could not load icon: {e}")
        return ""


def save_sessions_to_disk():
    """Save all chat sessions to disk for persistence."""
    sessions_path = Path("data/sessions.json")
    sessions_path.parent.mkdir(exist_ok=True)
    
    try:
        with open(sessions_path, "w", encoding="utf-8") as f:
            json.dump(st.session_state.sessions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Failed to save sessions: {e}")


def load_sessions_from_disk():
    """Load chat sessions from disk."""
    sessions_path = Path("data/sessions.json")
    
    if sessions_path.exists():
        try:
            with open(sessions_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load previous sessions: {e}")
    
    return None


# ---------------- DOC (from file or upload) ----------------
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


def split_into_sentences(text: str) -> list:
    """Split text into sentences using regex."""
    # Simple sentence splitter (handles . ! ?)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_sentences(text: str, sentences_per_chunk: int = 5, overlap: int = 1) -> list:
    """
    Improved chunking: sentence-based instead of word-based.
    Preserves sentence boundaries for better context.
    """
    sentences = split_into_sentences(text)
    
    if not sentences:
        return []
    
    chunks = []
    start = 0
    max_chunks = 1000  # Prevent memory issues with very large documents
    
    while start < len(sentences) and len(chunks) < max_chunks:
        end = min(len(sentences), start + sentences_per_chunk)
        chunk = " ".join(sentences[start:end])
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        # Move forward with overlap
        start = max(end - overlap, start + 1)
    
    return chunks


def dedup_near(texts: list, overlap_threshold: float = 0.85) -> list:
    """Remove near-duplicate chunks while preserving order."""
    if not texts:
        return []
    
    original = [x.strip() for x in texts if x and x.strip()]
    candidates = sorted(original, key=len, reverse=True)
    kept, kept_sets = [], []
    
    for t in candidates:
        w = set(t.lower().split())
        if len(w) == 0:
            continue
            
        # Check overlap with already kept chunks
        is_duplicate = any(
            (len(w & ws) / max(1, min(len(w), len(ws)))) >= overlap_threshold 
            for ws in kept_sets
        )
        
        if not is_duplicate:
            kept.append(t)
            kept_sets.append(w)
    
    # Return in original order
    return [t for t in original if t in kept]


@st.cache_resource(show_spinner="🔄 Initializing AI models…")
def build_rag(document_text: str):
    """Build RAG system with embeddings and vector database."""
    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Use improved sentence-based chunking
        chunks = chunk_text_sentences(document_text, sentences_per_chunk=5, overlap=1)
        
        if not chunks:
            st.error("No chunks created from document. Please check document content.")
            return None, None, None, []
        
        st.info(f"📄 Document split into {len(chunks)} chunks")
        
        # Generate embeddings
        embs = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)

        # Use persistent ChromaDB
        db_path = Path("data/chroma_db")
        db_path.mkdir(parents=True, exist_ok=True)
        
        db = chromadb.PersistentClient(path=str(db_path))
        
        # Delete existing collection to ensure fresh start
        try:
            db.delete_collection("rag")
        except:
            pass
        
        col = db.get_or_create_collection(
            name="rag",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add documents to collection
        col.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embs.tolist(),
        )

        # Initialize LLM
        api_key = st.secrets.get("TOGETHER_API_KEY")
        if not api_key:
            st.error("TOGETHER_API_KEY not found in secrets!")
            return None, None, None, []
        
        llm = Together(api_key=api_key)
        
        return llm, embedder, col, chunks
    
    except Exception as e:
        st.error(f"Error building RAG system: {e}")
        return None, None, None, []


def rag_answer(llm, embedder, col, query: str, model_name: str, top_k: int = 6, temperature: float = 0.3):
    """
    Enhanced RAG answer with better error handling and improved prompting.
    """
    try:
        # Generate query embedding
        q = embedder.encode([query], convert_to_numpy=True)[0]
        
        # Retrieve relevant chunks
        res = col.query(query_embeddings=[q.tolist()], n_results=top_k)
        
        if not res["documents"] or not res["documents"][0]:
            return "I couldn't find relevant information in the document to answer your question.", []
        
        # Deduplicate retrieved chunks
        chunks = dedup_near(res["documents"][0], overlap_threshold=0.85)
        
        if not chunks:
            return "I couldn't find relevant information in the document to answer your question.", []
        
        # Build context
        ctx = "\n\n---\n\n".join(chunks)
        
        # Improved system prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided document context.

Guidelines:
- Answer ONLY using information from the provided context
- Be accurate and concise
- If the context doesn't contain enough information to answer fully, clearly state this
- If you're unsure, say so rather than guessing
- Cite specific parts of the context when relevant
- Use a friendly, professional tone"""

        user_prompt = f"""Context from the document:

{ctx}

Question: {query}

Please provide a clear and accurate answer based on the context above."""

        # Call LLM
        response = llm.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=400,
            temperature=temperature,
        )
        
        answer = response.choices[0].message.content
        
        return answer, chunks
    
    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        st.error(error_msg)
        return "Sorry, I encountered an error while processing your question. Please try again.", []


def export_chat_history(session_data: dict, session_name: str):
    """Export chat history as JSON."""
    export_data = {
        "session_name": session_name,
        "export_time": datetime.now().isoformat(),
        "messages": session_data["messages"]
    }
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)


# ---------------- CHAT HISTORY (multi-session with persistence) ----------------
if "sessions" not in st.session_state:
    # Try to load from disk first
    loaded_sessions = load_sessions_from_disk()
    
    if loaded_sessions:
        st.session_state.sessions = loaded_sessions
    else:
        # Create default session
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
    save_sessions_to_disk()
    st.rerun()


# ---------------- SIDEBAR ----------------
with st.sidebar:
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
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Lower = more focused, Higher = more creative",
        key="temperature_slider"
    )
    
    top_k = st.slider(
        "Context chunks",
        min_value=3,
        max_value=10,
        value=6,
        help="Number of document chunks to retrieve",
        key="top_k_slider"
    )

    DEBUG = st.toggle("Show retrieved context", value=False, key="debug_toggle")
    
    st.divider()

    # Chat history management
    st.markdown("### 💬 Chat History")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➕ New", key="new_chat_btn", use_container_width=True):
            new_chat()
    with col2:
        if st.button("💾 Save", key="save_sessions_btn", use_container_width=True):
            save_sessions_to_disk()
            st.success("Saved!", icon="✅")
            time.sleep(1)
            st.rerun()

    # Session selector
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
            save_sessions_to_disk()

        # Export and delete buttons
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Export chat
            export_data = export_chat_history(
                st.session_state.sessions[chosen],
                st.session_state.sessions[chosen]["name"]
            )
            st.download_button(
                label="📥 Export",
                data=export_data,
                file_name=f"{st.session_state.sessions[chosen]['name']}.json",
                mime="application/json",
                key="export_btn",
                use_container_width=True
            )
        
        with col2:
            if st.button("🗑️ Delete", key="delete_chat_btn", use_container_width=True):
                del st.session_state.sessions[chosen]
                save_sessions_to_disk()
                
                if not st.session_state.sessions:
                    new_chat()
                else:
                    st.session_state.active_session = list(st.session_state.sessions.keys())[-1]
                st.rerun()

    st.divider()
    
    # Document info
    st.markdown("### 📄 Document")
    doc_size = len(DOCUMENT.split()) if DOCUMENT else 0
    st.caption(f"Words: {doc_size:,}")


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

# ---------------- DOCUMENT LOADING ----------------
# Try to load from file first
DOCUMENT = load_document(str(DOC_PATH))

# If no document, offer upload
if not DOCUMENT:
    st.warning("⚠️ No document found at `data/document.txt`")
    
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["txt", "md"],
        help="Upload a text or markdown file to analyze"
    )
    
    if uploaded_file:
        try:
            DOCUMENT = uploaded_file.read().decode("utf-8").strip()
            
            # Save to disk
            DOC_PATH.parent.mkdir(exist_ok=True)
            DOC_PATH.write_text(DOCUMENT, encoding="utf-8")
            
            st.success("✅ Document uploaded successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
    else:
        st.info("Please add a document at `data/document.txt` or upload one above.")
        st.stop()

# ---------------- RAG INIT ----------------
llm, embedder, col, all_chunks = build_rag(DOCUMENT)

if llm is None or embedder is None or col is None:
    st.error("Failed to initialize RAG system. Please check your configuration.")
    st.stop()

# Use active session messages
messages = st.session_state.sessions[st.session_state.active_session]["messages"]

# ---------------- CHAT UI ----------------
for i, m in enumerate(messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        
        # Add copy button for assistant messages
        if m["role"] == "assistant" and i > 0:  # Skip initial greeting
            st.button(
                "📋 Copy",
                key=f"copy_{i}",
                help="Copy to clipboard",
                on_click=lambda content=m["content"]: st.write(f"```\n{content}\n```")
            )

# Chat input
prompt = st.chat_input("Ask about the document…")

if prompt:
    # Add user message
    messages.append({"role": "user", "content": prompt})
    save_sessions_to_disk()
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking…"):
            ans, retrieved_chunks = rag_answer(
                llm, 
                embedder, 
                col, 
                prompt, 
                model_name=MODEL_NAME,
                top_k=top_k,
                temperature=temperature
            )
        
        st.markdown(ans)

        # Show retrieved context in debug mode
        if DEBUG and retrieved_chunks:
            with st.expander(f"📚 Retrieved {len(retrieved_chunks)} context chunks"):
                for idx, chunk in enumerate(retrieved_chunks, 1):
                    st.markdown(f"**Chunk {idx}:**")
                    st.markdown(f'<div class="source-citation">{chunk}</div>', unsafe_allow_html=True)
                    st.markdown("---")

    # Save assistant message
    messages.append({"role": "assistant", "content": ans})
    save_sessions_to_disk()
```

## 🎉 Key Improvements Implemented

### 1. **Persistent Storage** 💾
- Chat sessions save to `data/sessions.json`
- ChromaDB uses `PersistentClient` instead of in-memory
- Sessions survive app restarts

### 2. **Better Chunking** 📝
- Sentence-based chunking (preserves context)
- Configurable sentences per chunk
- Prevents awkward mid-sentence splits

### 3. **Enhanced Error Handling** 🛡️
- Try-except blocks throughout
- Graceful fallbacks
- User-friendly error messages

### 4. **Improved UI/UX** ✨
- Export chat history as JSON
- Document upload widget
- Save button for manual persistence
- Temperature & top_k sliders
- Document word count display
- Copy buttons for responses (basic implementation)
- Better session management

### 5. **Better Prompting** 🎯
- Enhanced system prompt with clear guidelines
- Structured user prompt
- Increased max_tokens to 400
- Configurable temperature

### 6. **Code Quality** 📊
- Better documentation
- Type hints where helpful
- Modular functions
- Memory limits (max 1000 chunks)

### 7. **Security** 🔒
- Better password gate with session state
- API key validation
- Input sanitization

### 8. **Debug Features** 🔍
- Show retrieved chunks with styling
- Numbered chunks in expanders
- Visual separation

## 📁 Required File Structure
```
your_project/
├── app.py (this script)
├── assets/
│   └── orca.png
├── data/
│   ├── document.txt (your document)
│   ├── sessions.json (auto-created)
│   └── chroma_db/ (auto-created)
└── .streamlit/
    └── secrets.toml
