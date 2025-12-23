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
AboutAbout
An enthusiastic researcher with a passion for learning and a proven track record of developing analytical and problem-solving skills. My experience in Catalysis Chemistry includes applying various characterization methods for both homogeneous and heterogeneous systems. This has equipped me with the ability to understand, optimize, and design complex systems and reactions.
My passion for discovery extends beyond the lab. Working on diverse research projects across multiple groups and countries has enriched my communication, planning, and time management skills. This international exposure has also taught me the importance of adaptability and collaboration.


Experience
Pidpa logo
Trainee
Trainee
Pidpa · InternshipPidpa · Internship
Jul 2025 - Present · 6 mosJul 2025 to Present · 6 mos
Antwerp, Flemish Region, Belgium · On-siteAntwerp, Flemish Region, Belgium · On-site
-Conducted inorganic water quality analyses (Cl⁻, SO₄²⁻, NO₂⁻, NO₃⁻, NH₄⁺) using Discret Analysis technique

-Performed BOD and COD testing, as well as measurement of suspended solids in wastewater and TAM

-Gained experience with laboratory procedures and quality assurance standards (ISO & WAC documentation)

-Contributed to environmental monitoring and support accurate reporting for wastewater and drinking water safety
-Conducted inorganic water quality analyses (Cl⁻, SO₄²⁻, NO₂⁻, NO₃⁻, NH₄⁺) using Discret Analysis technique -Performed BOD and COD testing, as well as measurement of suspended solids in wastewater and TAM -Gained experience with laboratory procedures and quality assurance standards (ISO & WAC documentation) -Contributed to environmental monitoring and support accurate reporting for wastewater and drinking water safety
Skills: ISO 17025 · ISO 5667 · Interdisciplinary Collaboration · Reporting & Analysis · Dutch language · waste water analysis · Drinking Water Quality · Laboratory Information Management System (LIMS)
Skills: ISO 17025 · ISO 5667 · Interdisciplinary Collaboration · Reporting & Analysis · Dutch language · waste water analysis · Drinking Water Quality · Laboratory Information Management System (LIMS)
Personal Goal Pursuit: Sustainability and Green Innovation
Personal Goal Pursuit: Sustainability and Green Innovation
Personal Goal Pursuit: Sustainability and Green Innovation
Self-employedSelf-employed
Aug 2024 - Present · 1 yr 5 mosAug 2024 to Present · 1 yr 5 mos
Antwerp, Flemish Region, Belgium · RemoteAntwerp, Flemish Region, Belgium · Remote
I have been purposefully working toward aligning my career with my passion for sustainability and green innovation. During this time, I have dedicated myself to acquiring knowledge and skills in green chemistry, circular economy principles, and life cycle assessment (LCA) through a combination of courses, workshops, and self-directed learning.

I am excited to integrate these new competencies with my technical expertise, continuously develop my abilities, and make a meaningful impact in improving the world around me.
I have been purposefully working toward aligning my career with my passion for sustainability and green innovation. During this time, I have dedicated myself to acquiring knowledge and skills in green chemistry, circular economy principles, and life cycle assessment (LCA) through a combination of courses, workshops, and self-directed learning. I am excited to integrate these new competencies with my technical expertise, continuously develop my abilities, and make a meaningful impact in improving the world around me.
Skills: Circular Economy · Project Management · Life Cycle Assessment
Skills: Circular Economy · Project Management · Life Cycle Assessment
Private Tutor
Private Tutor
Private Tutor
Dr Ship Tutoring & Consulting · Part-timeDr Ship Tutoring & Consulting · Part-time
Nov 2023 - Oct 2025 · 2 yrsNov 2023 to Oct 2025 · 2 yrs
Antwerp, Flemish Region, Belgium · HybridAntwerp, Flemish Region, Belgium · Hybrid
𝐏𝐚𝐬𝐬𝐢𝐨𝐧𝐚𝐭𝐞 𝐀𝐛𝐨𝐮𝐭 𝐂𝐡𝐞𝐦𝐢𝐬𝐭𝐫𝐲 | 𝐈𝐧𝐬𝐩𝐢𝐫𝐢𝐧𝐠 𝐅𝐮𝐭𝐮𝐫𝐞 𝐂𝐡𝐞𝐦𝐢𝐬𝐭𝐬
I share my decade-long love for chemistry by helping students explore their interests and see how chemistry shapes our world. In this part-time role, I do my best to :
- Simplify chemistry concepts to build a strong foundational understanding
- Guide students step-by-step to solve problems independently
- Develop problem-solving skills with tailored, progressively challenging questions
- Link theoretical knowledge to experimental applications
𝐏𝐚𝐬𝐬𝐢𝐨𝐧𝐚𝐭𝐞 𝐀𝐛𝐨𝐮𝐭 𝐂𝐡𝐞𝐦𝐢𝐬𝐭𝐫𝐲 | 𝐈𝐧𝐬𝐩𝐢𝐫𝐢𝐧𝐠 𝐅𝐮𝐭𝐮𝐫𝐞 𝐂𝐡𝐞𝐦𝐢𝐬𝐭𝐬 I share my decade-long love for chemistry by helping students explore their interests and see how chemistry shapes our world. In this part-time role, I do my best to : - Simplify chemistry concepts to build a strong foundational understanding - Guide students step-by-step to solve problems independently - Develop problem-solving skills with tailored, progressively challenging questions - Link theoretical knowledge to experimental applications
Skills: IB chemistry · Communication · Mentoring · teaching chemistry
Skills: IB chemistry · Communication · Mentoring · teaching chemistry
University of Antwerp logo
Early stage researcher 
Early stage researcher 
University of AntwerpUniversity of Antwerp
Apr 2023 - Feb 2024 · 11 mosApr 2023 to Feb 2024 · 11 mos
Skills: Thesis writing · Report Writing · Technical Presentations
Skills: Thesis writing · Report Writing · Technical Presentations
Cardiff University / Prifysgol Caerdydd logo
Cardiff University / Prifysgol Caerdydd
Cardiff University / Prifysgol Caerdydd
4 yrs 8 mos4 yrs 8 mos
On-siteOn-site
Early stage researcher
Early stage researcher
Full-timeFull-time
Jul 2019 - Feb 2024 · 4 yrs 8 mosJul 2019 to Feb 2024 · 4 yrs 8 mos
𝐑𝐞𝐬𝐞𝐚𝐫𝐜𝐡𝐞𝐫 𝐢𝐧 𝐚 𝐂𝐫𝐨𝐬𝐬-𝐃𝐢𝐬𝐜𝐢𝐩𝐥𝐢𝐧𝐚𝐫𝐲 𝐏𝐫𝐨𝐣𝐞𝐜𝐭 | 𝐃𝐞𝐯𝐞𝐥𝐨𝐩𝐢𝐧𝐠 𝐓𝐞𝐜𝐡𝐧𝐢𝐜𝐚𝐥 𝐚𝐧𝐝 𝐒𝐨𝐟𝐭 𝐒𝐤𝐢𝐥𝐥𝐬
In this role, I conducted research while gaining both technical expertise and essential soft skills, including:
-𝐀𝐝𝐯𝐚𝐧𝐜𝐞𝐝 𝐌𝐚𝐭𝐞𝐫𝐢𝐚𝐥 𝐂𝐡𝐚𝐫𝐚𝐜𝐭𝐞𝐫𝐢𝐳𝐚𝐭𝐢𝐨𝐧: Conducted in-depth investigations of inorganic/organic metal complexes using Electron Paramagnetic Resonance (EPR) Spectroscopy, including Pulsed and High-Frequency techniques, with additional expertise in IR, UV-Vis, DRS-UV, BET, and HPLC methods
-𝐁𝐫𝐨𝐚𝐝 𝐀𝐧𝐚𝐥𝐲𝐭𝐢𝐜𝐚𝐥 𝐏𝐫𝐨𝐟𝐢𝐜𝐢𝐞𝐧𝐜𝐲: Complementary experience with Raman, XRD, TEM, and SEM techniques to support comprehensive material analysis
-𝐂𝐚𝐭𝐚𝐥𝐲𝐬𝐭 𝐃𝐞𝐬𝐢𝐠𝐧: 3 years of experience in reaction design (DOE) and optimization
-𝐂𝐚𝐭𝐚𝐥𝐲𝐬𝐭 𝐒𝐲𝐧𝐭𝐡𝐞𝐬𝐢𝐬: Proficient in sol-gel, co-precipitation, and impregnation methods using porous materials
-𝐒𝐚𝐟𝐞𝐭𝐲 𝐄𝐱𝐩𝐞𝐫𝐭𝐢𝐬𝐞: Skilled in COSHH and Risk Assessments for handling metal salts and organic compounds
-𝐂𝐨𝐥𝐥𝐚𝐛𝐨𝐫𝐚𝐭𝐢𝐨𝐧: 4 years of cross-disciplinary work with international research teams
𝐑𝐞𝐬𝐞𝐚𝐫𝐜𝐡𝐞𝐫 𝐢𝐧 𝐚 𝐂𝐫𝐨𝐬𝐬-𝐃𝐢𝐬𝐜𝐢𝐩𝐥𝐢𝐧𝐚𝐫𝐲 𝐏𝐫𝐨𝐣𝐞𝐜𝐭 | 𝐃𝐞𝐯𝐞𝐥𝐨𝐩𝐢𝐧𝐠 𝐓𝐞𝐜𝐡𝐧𝐢𝐜𝐚𝐥 𝐚𝐧𝐝 𝐒𝐨𝐟𝐭 𝐒𝐤𝐢𝐥𝐥𝐬 In this role, I conducted research while gaining both technical expertise and essential soft skills, including: -𝐀𝐝𝐯𝐚𝐧𝐜𝐞𝐝 𝐌𝐚𝐭𝐞𝐫𝐢𝐚𝐥 𝐂𝐡𝐚𝐫𝐚𝐜𝐭𝐞𝐫𝐢𝐳𝐚𝐭𝐢𝐨𝐧: Conducted in-depth investigations of inorganic/organic metal complexes using Electron Paramagnetic Resonance (EPR) Spectroscopy, including Pulsed and High-Frequency techniques, with additional expertise in IR, UV-Vis, DRS-UV, BET, and HPLC methods -𝐁𝐫𝐨𝐚𝐝 𝐀𝐧𝐚𝐥𝐲𝐭𝐢𝐜𝐚𝐥 𝐏𝐫𝐨𝐟𝐢𝐜𝐢𝐞𝐧𝐜𝐲: Complementary experience with Raman, XRD, TEM, and SEM techniques to support comprehensive material analysis -𝐂𝐚𝐭𝐚𝐥𝐲𝐬𝐭 𝐃𝐞𝐬𝐢𝐠𝐧: 3 years of experience in reaction design (DOE) and optimization -𝐂𝐚𝐭𝐚𝐥𝐲𝐬𝐭 𝐒𝐲𝐧𝐭𝐡𝐞𝐬𝐢𝐬: Proficient in sol-gel, co-precipitation, and impregnation methods using porous materials -𝐒𝐚𝐟𝐞𝐭𝐲 𝐄𝐱𝐩𝐞𝐫𝐭𝐢𝐬𝐞: Skilled in COSHH and Risk Assessments for handling metal salts and organic compounds -𝐂𝐨𝐥𝐥𝐚𝐛𝐨𝐫𝐚𝐭𝐢𝐨𝐧: 4 years of cross-disciplinary work with international research teams
Skills: Public Speaking · Travel Arrangements · Data Analysis · Problem Solving · Poster Presentations · Design Thinking · Teamwork · Interdisciplinary Collaboration · Cross-cultural Teams · Crystallization · DFT · Homogeneous Catalysis · earth abundant metals chemistry · Zeolite characterization · Material Properties
Skills: Public Speaking · Travel Arrangements · Data Analysis · Problem Solving · Poster Presentations · Design Thinking · Teamwork · Interdisciplinary Collaboration · Cross-cultural Teams · Crystallization · DFT · Homogeneous Catalysis · earth abundant metals chemistry · Zeolite characterization · Material Properties
Graduate Teaching Assistant
Graduate Teaching Assistant
Nov 2021 - Apr 2022 · 6 mosNov 2021 to Apr 2022 · 6 mos
Skills: Teaching · Workplace Safety · Communication · Mentoring · practical · Laboratory Safety
Skills: Teaching · Workplace Safety · Communication · Mentoring · practical · Laboratory Safety
University of Antwerp logo
Early stage researcher
Early stage researcher
University of Antwerp · Full-timeUniversity of Antwerp · Full-time
Oct 2020 - Oct 2021 · 1 yr 1 moOct 2020 to Oct 2021 · 1 yr 1 mo
𝐄𝐱𝐩𝐚𝐧𝐝𝐞𝐝 𝐄𝐱𝐩𝐞𝐫𝐭𝐢𝐬𝐞 𝐓𝐡𝐫𝐨𝐮𝐠𝐡 𝐔𝐧𝐢𝐯𝐞𝐫𝐬𝐢𝐭𝐲 𝐒𝐞𝐜𝐨𝐧𝐝𝐦𝐞𝐧𝐭:
- Completed a one-year secondment, adapting to a new work environment and stepping out of my comfort zone
- Enhanced teamwork through collaboration with diverse research groups
- Gained expertise in advanced EPR techniques and cryogenic instruments
𝐄𝐱𝐩𝐚𝐧𝐝𝐞𝐝 𝐄𝐱𝐩𝐞𝐫𝐭𝐢𝐬𝐞 𝐓𝐡𝐫𝐨𝐮𝐠𝐡 𝐔𝐧𝐢𝐯𝐞𝐫𝐬𝐢𝐭𝐲 𝐒𝐞𝐜𝐨𝐧𝐝𝐦𝐞𝐧𝐭: - Completed a one-year secondment, adapting to a new work environment and stepping out of my comfort zone - Enhanced teamwork through collaboration with diverse research groups - Gained expertise in advanced EPR techniques and cryogenic instruments
Skills: Pulsed EPR · High Frequency EPR · Setting up cryogenic systems · Design Thinking · Cross-cultural Teams
Skills: Pulsed EPR · High Frequency EPR · Setting up cryogenic systems · Design Thinking · Cross-cultural Teams
ACG World logo
Sales Engineer
Sales Engineer
ACG World · Full-timeACG World · Full-time
Jul 2017 - Feb 2019 · 1 yr 8 mosJul 2017 to Feb 2019 · 1 yr 8 mos
Tehran Province, IranTehran Province, Iran
𝐒𝐚𝐥𝐞𝐬 𝐄𝐧𝐠𝐢𝐧𝐞𝐞𝐫 (𝐁2𝐁) – 𝐏𝐡𝐚𝐫𝐦𝐚 𝐄𝐪𝐮𝐢𝐩𝐦𝐞𝐧𝐭 𝐒𝐩𝐞𝐜𝐢𝐚𝐥𝐢𝐬𝐭
As a Sales Engineer, I gained interdisciplinary expertise in managing technical projects, production processes, and client relationships in pharmaceutical factories, building strong skills in planning, communication, and precision. In this role I :
- Presented advanced pharmaceutical machinery to 20+ companies
- Led installation of 4 packing and 2 inspection machines with ACG engineers
- Managed client accounts and delivered outstanding customer service.
- Streamlined installation processes, cutting time by 10%
𝐒𝐚𝐥𝐞𝐬 𝐄𝐧𝐠𝐢𝐧𝐞𝐞𝐫 (𝐁2𝐁) – 𝐏𝐡𝐚𝐫𝐦𝐚 𝐄𝐪𝐮𝐢𝐩𝐦𝐞𝐧𝐭 𝐒𝐩𝐞𝐜𝐢𝐚𝐥𝐢𝐬𝐭 As a Sales Engineer, I gained interdisciplinary expertise in managing technical projects, production processes, and client relationships in pharmaceutical factories, building strong skills in planning, communication, and precision. In this role I : - Presented advanced pharmaceutical machinery to 20+ companies - Led installation of 4 packing and 2 inspection machines with ACG engineers - Managed client accounts and delivered outstanding customer service. - Streamlined installation processes, cutting time by 10%
Skills: Pre-Sales Technical Consulting · Customer Support · Technical Documentation · Technical Support · Commercials · Cross-cultural Teams
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
