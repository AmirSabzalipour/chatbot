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
CV — Fardokht Rezayi
Last update: 12/2025

CONTACT
- Name: Fardokht Rezayi
- Title: Researcher in Chemistry
- Email: Rezayisf@gmail.com
- LinkedIn: linkedin.com/in/fardokhtrz
- Phone: +32 484412315
- Location: 2000 Antwerpen, Belgium
- Work authorization: Valid work permit
- Driving license: Belgian driving license

TECHNICAL EXPERTISE
Analytical / Characterization Techniques:
- Biochemical Oxygen Demand (BOD)
- Chemical Oxygen Demand (COD)
- Electron Paramagnetic Resonance (EPR)
 ' Infrared (IR)
- Ultraviolet-Visible (UV-Vis)
- Diffuse Reflectance Spectroscopy (DRS-UV)
- BET surface area analysis
- High-Performance Liquid Chromatography (HPLC)

PROFILE SUMMARY
Proactive and internationally minded chemist with 5+ years of experience in spectroscopy, catalysis, and porous material chemistry. Skilled in experimental design, data analysis, and interdisciplinary research. Initiated an internship at Pidpa in the water sector; gained experience in an ISO-certified laboratory environment and improved Dutch proficiency to navigate local environmental policies and regulations. Focused on sustainability, process optimization, and data-driven decision-making in R&D environments.

WORK EXPERIENCE

1) Trainee – Laboratory (Inorganic Analysis)
- Company: Pidpa
- Location: Antwerp, Belgium
- Type: Part-time
- Dates: 07/2025 – 12/2025 (ongoing)
Responsibilities / Achievements:
- Inorganic water quality analyses (Cl⁻, SO₄²⁻, NO₂⁻, NO₃⁻, NH₄⁺) using Discrete Analysis technique
- BOD and COD testing; measurement of suspended solids in wastewater and TAM
- Worked with ISO 17025 and WAC documentation; LIMS registration
- Supported environmental monitoring and accurate reporting for wastewater and drinking water safety

2) Chemistry Private Tutor
- Company: Dr Ship Tutoring & Consulting
- Type: Part-time
- Dates: 11/2023 – 10/2025
Responsibilities / Achievements:
- Simplified chemistry concepts to build strong foundations
- Developed critical thinking via tailored, progressively challenging questions

3) PhD Researcher — Spectroscopy and Catalysis Chemistry
- Institution: Cardiff University (UK) – University of Antwerp (Belgium)
- Dates: 07/2019 – 06/2024
Responsibilities / Achievements:
- Operated and troubleshot advanced techniques: EPR, IR, UV-Vis, DRS-UV, BET, HPLC
- Designed and optimized catalytic processes (Design of Experiments / DOE); catalyst synthesis
- Project work with multidisciplinary teams and external partners
- Data analysis, reporting, publications, presentations
- Expertise: glycerol valorisation with homo/heterogeneous catalysts

4) Sales Engineer (B2B) — Technical Machinery Solutions (Pharma Industry)
- Company: Amiz Sazan Ari (Official Representative of ACG Pharma)
- Location: Tehran, Iran
- Dates: 03/2017 – 02/2019
Responsibilities / Achievements:
- Introduced machines to 20+ pharma companies
- 4+ on-site installations with ACG engineers
- Prepared technical documentation and reports for management

TRAININGS AND SKILLS

Soft Skills:
- Excellent organisational, communication, and interpersonal abilities
- Fast learner; scientific curiosity; self-steering; flexible; creative

Laboratory / Practical Skills:
- Setup, maintenance, troubleshooting of cryogenic systems (liquid helium/nitrogen setups, pumps, compressors)
- Operating batch pressure reactors
- In-situ and ex-situ analysis of reaction products

Trainings:
- Industrial aspect of catalytic process — LyondellBasell (Online), Italy — 02/2021
- EPR spectroscopy — Bruker (Online), Germany — 11/2020
- LCA (Life Cycle Assessment) — Zaragoza University, Spain — 09/2020
- Self-study: Circular Economy — Sustainable Materials Management (Coursera)

Languages:
- English: Full professional proficiency
- Persian: Native or bilingual proficiency
- Dutch: Professional working proficiency
- French: Limited working proficiency

Software:
- LIMSLink (LIMS)
- MATLAB (EasySpin toolbox)
- ChemDraw
- Avogadro
- Origin
- Microsoft Office

EDUCATION

1) PhD in Chemistry (MSCA Joint Program: PARACAT)
- Institutions: Cardiff University, UK — University of Antwerp, Belgium
- Dates: 07/2019 – 06/2024
- Thesis: Advanced EPR Investigation of Copper Complexes in Catalysis
- Publication: Book chapter

2) M.Sc. in Organic Chemistry
- Institution: Kharazmi University, Tehran, Iran
- Dates: 01/2015 – 01/2017
- Publication: Thesis project paper on a bi-functional metal/solid acid catalyst for direct reductive amination of nitroarenes on mesoporous carbon (CMK-8)

3) B.Sc. in Chemistry
- Institution: University of Arak, Arak, Iran
- Dates: 02/2009 – 02/2013

ADDITIONAL NOTES
- Flexible to relocate or commute as needed
- Happy to travel for work and engage in international collaborations

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
# ---------------- PASSWORD GATE ----------------
pw_required = st.secrets.get("APP_PASSWORD", "")
if pw_required:
    st.markdown("### 👋 Welcome to AmirBot")
    st.write("Please insert your password, dear 🙂")

    pw = st.text_input("Password", type="password", placeholder="Enter password…")
    if pw != pw_required:
        st.info("If you don’t have the password, message Amir.")
        st.stop()

# ---------------- RAG INIT ----------------
llm, embedder, col = build_rag()  # shows 🔄 Thinking… while building (via @st.cache_resource)

# ---------------- CHAT HISTORY (multi-session) ----------------
import time

if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "active_session" not in st.session_state:
    sid = str(int(time.time()))
    st.session_state.sessions[sid] = {
        "name": "Chat 1",
        "messages": [{"role": "assistant", "content": "Hi! Ask me anything about the document 🙂"}],
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
            ans = rag_answer(llm, embedder, col, prompt, model_name=MODEL_NAME)
        st.markdown(ans)

    messages.append({"role": "assistant", "content": ans})
