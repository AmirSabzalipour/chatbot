import streamlit as st
from together import Together
import chromadb
from sentence_transformers import SentenceTransformer
import time

# ---------------- UI CONFIG ----------------
BOT_NAME = "AmirBot"
BOT_ICON = "🤖"

st.set_page_config(page_title=BOT_NAME, page_icon=BOT_ICON, layout="centered")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 900px;}
.stChatMessage {border-radius: 14px; padding: 6px 10px;}
[data-testid="stChatInput"] textarea {border-radius: 14px;}
.model-pill {display:inline-block;padding:6px 10px;border-radius:999px;border:1px solid #ddd;font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

# ---------------- DOC ----------------
DOCUMENT = """
Here is the CV in plain text:

Ali ARAMESH Test Engineer (+32) 483 46 19 30 aliaramesh85@gmail.com Rue du moulin 105, 4684 Haccourt, Belgium linkedin.com/in/aliaramesh github.com/aramesh

CAREER SUMMARY/ACCOMPLISHED PROJECTS

Ongoing Test Rig Engineer | Stellantis, e-transmission, SINT-TRUIDEN, Belgium
March 2025
Responsibilities:

    Preparation and execution of different tests on Dual clutch transmissions including: (High Cycle Fatigue, Friction Plate Durability, Park Lock Rattle/Actiation Durability, Cold/Hot Efficiency, Gear Actuation, Electric Motor Durability, etc)
    Modifying the script to be adapted with the test request and the load collective
    Preparing test benches and CAN communication
    Adding a layer of safety to a sofisticated roll-pitch bench
    Building up the transmission into the bench
    Checking out a working copy, updating to the latest revision, make changes to the script adapted to teh TREQ and load collective and commit to the SVN repository browser

Skills: CANape, Control Desk, Validation, EPLAN, SVN tortoise, Test report, CAN communication, Jira, FEA, Siemens PLC & pilz relay troubleshooting, CFD & FEA analysis in dual-clutch transmission, TCU, PHEV integration

Senior Electrical Design & Project Engineer | TOYOTA, ZAVENTEM, Belgium
February 2024 - February 2025
Responsibilities:

    Spreadheaded the development of an advanced UWB-based anti-theft system
    Led the redesign of the wire-harness and connectors for C-HR
    Preparing removal and installation manuals for C-HR
    Managed cross-functional teams, ensuring project milestones were met on time and within budget
    Coordinated with stakeholders for seamless integration of design and manufacturing process

Skills: Project Management (PM), CATIA, Vehile Specification Documentation, System integration, Wire harness design, Electrical Wire Diagram (EWD), Manual & rework documentation

Powertrain Engineer | TOYOTA, ZAVENTEM, Belgium
April 2023 - January 2024
Responsibilities:

    Prepared RDDP (Resuest for design & development process)
    Redesign according to RDDP drawings
    NVH benchmarking on BEVs (Audi e-tron, Volvo XC40, bz4X)
    Order tracking on different powertrain components of the BEVs
    TPA analysis to identify powertrain's contribution to final comfort
    Design of interior permanent magnet (IPM) motor for BEVs

Skills: CATIA V5, Ansys Motor-CAD, Ansys Maxwell, Battery Management System (BMS), TPA (Transfer Path Analysis), Benchmarking, Order tracking, Body Sensitivity, Body Acoustic Sensitivity, FEA, CFD

Test Engineer | Capacitor lab/Hitachi Energy ABB via consultant, CHARLEROI, Belgium
June 2022 - March 2023

    Tear-down analysis on film capacitors
    Implementing different tests on film capacitors including ageing, humidity, thermal, DC leakage current measurement
    No-cause test on metalised film before winding including film thickness, resistivity, free margin and segmented, active region
    Report and documenting the analysis result
    Participation in daily meeting on RCA (Root Cause) analysis

Skills: Metallization slitting, Winding, Schoopage, Thermal treatment (analysis), Self-healing zinc projection, Vibration testing, Manufacturing via cleanroom, Short-circuit destruction testing on film capacitors (20KA), test report documentation

System Engineer | Power electronics lab - HIL testing, HITACHI ENERGY ABB VIA CONSULTANT, Charleroi, Belgium
April 2022 - March 2023

    Executed Hardware-in-the-loop (HIL) tests for active filters (PQactiF)
    Tested IGBT drives, AC/DC power modules, and control PCBs under real-world grid conditions, ensuring compliance with IEC 61000-3-2/IEEE 519 standards
    Evaluating various control algorithms under load conditions using Typhoon HIL device
    Assembled and debugged PQactiF systems (similar to e-transmission drive packages), including PCB rework (microscope-level soldering) and functional testing
    Scripted in python to automate multiple tests, with results generated and reported in the Allure framework

Skills: dSPACE, Harmonic filtering, Load balancing, Reactive power compensation, PCB rework functional testing, Typhoon-HIL, Block diagram & Schematic drawing

Master Thesis | Driveline/Machatronics lab, ANTWERPEN, Belgium
February 2021 - September 2021
Suppressing shock and vibration in a driveline by means of redesignig speed controller and utilizing flexible coupling:

    Building two identical drivelines on mobile chassis
    Commissioning the drivelines by servo drives
    Applying different tunning methods to design the speed controller

Skills: Siemens starter S120 servo drives, Matlab, PI-controller, Flexible coupling

System Engineering Intern | Model-Based Testing & Validation, Siemens, LEUVEN, Belgium
September 2019 - February 2020

    Contributed to the V-cycle development process for a hybrid hydraulic powertrain system combining internal combustion engine (ICE) and hydraulic actuation
    Derived and implemented test cases from system-level requirements, focusing on validating mode transitions and control logic behavior
    Executed Model-in-the-Loop (MiL) and Software-in-the-Loop (SiL) simulations in Simulink, ensuring model accuracy and behavioral integrity
    Automated test case execution using Jenkins, achieving 100% maturity for multiple test scenarios through decision and condition coverage
    Utilized Polarion for traceability between requirements, test cases, and validation outcomes in alignment with ALM processes
    Simulated operational scenarios including transitions such as Combined mode to Hydraulic mode, and Brake to Drive, ensuring compliance with expected functional performance
    Supported verification activities at the left-hand side of the V-model, contributing to system robustness before integration phases

Skills: Model-based design principles, V-cycle systems engineering, Automotive software quality standards, ALM polarion, Simulink, Matlab, Jenkins, SVN tortoise

HV MV Commissioning and Test Engineer, METANIR, Tehran, Iran
October 2011 - October 2015

    Commissioning and maintenance testing of HV & MV switchgear
    Assessment of ESB, manufacturer and contractor test results and documentation
    Maintenance and commissioning testing of protection relays
    Restoring power and channels of pilz relays in case of trip or fault (X3.10P & XV2P PNOZ Pilz relays)
    Verification of SCADA systems and associated instrumentation loops and transducers
    EPLAN redesign in case of minor modification into the control cabinet

Skills: Block diagrams, Schematic drawing, LV cabling, Data acquisition system, Protection relay wiring, EPLAN, JIRA

PROJECTS

DESIGN OF INTERIOR PERMANNET MAGNET (IPM) MOTOR FOR EV VEHICLES
NOV 2024 - FEB 2024
github.com/ArameshAli/IPM- Motor technology
In this project we are designing interior Permanent Magnet (IPM) technology for EVs. IPM motors offer superior power density and efficiency, delivering higher torque across a broad speed range. By enhancing reluctance torque, they achieve increased performance at high speeds, making them ideal for EV applications.
Tools: ANSYS Motor-CAD

VIBROACOUSTIC COMFORT IN ELECTRIC VEHICLES
APRIL 2019 - JUNE 2019
github.com/ArameshAli/Half-Vehicle-Model/blob/main/HalfVM.m - Half Vehicle model
The half vehicle modal is the most popular method of analysing comfort and perform mathematical modelling of a vehicle. Objectives are: 1. Obtain engine modal excitation data from Ansys. 2. To determine the transmissibility function of the HVM. 3. Validate obtained/observed acceleration function from the modal with real Twizy experimental values.
Tools: Ansys, simulink

SUSTAINABLE VEHICLE POWERTRAINS
JAN 2019 - MARCH 2019
github.com/ArameshAli/Battery - Battery pack specification
The aim of this project is to characterise the performance of commercial Lithium ion automotive battery cells and use experimental data to create the design specification for a battery electric vehicle.

POWERTRAIN CALIBRATION OPTIMISATION
JAN 2019 - MARCH 2019
github.com/ArameshAli/NOx-modeling/blob/main/Project1.m - NOx simulation
We will reduce NOx emissions in this project. The goal is to reduce NOx while increasing particulate matter (PM), which is unavoidable. There is a tradeoff between NOx and PM. While NOx levels have decreased, PM levels have increased.
Techniques: mixed-integer linear programming, quadratic programming, nonlinear optimisation

SUSTAINABLE VEHICLE POWERTRAIN
JAN 2019 - MARCH 2019
github.com/ArameshAli/NOx-modeling/blob/main/Project1.m - Hybrid Vehicle Modelling & Analysis
In the future, cars will be characterised by an increasing degree of hybridisation. They use increasing levels of hybridisation, and the expectation is that this improves the fuel economy. The purpose of this project is to look at the impact of vehicle architectures, to understand how and why fuel economy can be improved, and at finding the right control strategy for a hybrid powertrain.
Tools: WLTP, CADC, Elvio (ELectrified Vehicle library for sImulation and Optimisation) Toolbox

CORE SKILLS & LANGUAGES

System Engineerng Tools: ALM Pomarion, V-cycle
Simulation/Modeling/CAD: MATLAB/Simulink, Typhoon HIL, dSPACE, CANape, control desk, Ansys Motor-CAD, CATIA V5
Problem-Solving: 8D/QRQC, six sigma, FMEA, sisk assessment
Core skills: Automation, optimization, control design for industrial applications
Strengths: Goal-oriented, analytical mindset, industrious, cross-functional, team member, problem solver
Methodologies: Kaizen, Design review based on failure mode (DRBFM), Mizen boushi
Languages: French, English, Basic Japanease

EDUCATION

2018-2021
Joint master's degree in sustainable automotive engineering
Universiteit Antwerpen, Loughborough University, Universidad de Deusto, Université de Bordeaux
Thesis subject: A comparison of different tuning methods to design a speed controller for a driveline

2017-2018
MS.c in Electrical Engineering (Electric power and energy systems) (60 ECTS)
Université de Liège

2016-2017
Preparatory year for the master's degree in computer science
Université de Liège

2008-2012
BS.c in electrical and Electronics Engineeirng
Azad University - Central Tehran Branch (iauctb)
Control the speed and direction of a DC motor

REFERENCES
Available upon request
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
        kept.append(t)
        kept_sets.append(w)
    return [t for t in original if t in kept]

@st.cache_resource(show_spinner="🔄 Thinking…")
def build_rag():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = chunk_text_words(DOCUMENT, 80, 20)
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

# ---------------- HEADER ----------------
st.title(f"{BOT_ICON} {BOT_NAME}")
st.caption("Ask your question. If it’s not in the doc, I’ll say I don’t know.")
st.markdown(f"<div class='model-pill'>{MODEL_NAME}</div>", unsafe_allow_html=True)

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
llm, embedder, col = build_rag()

# ---------------- CHAT HISTORY (multi-session) ----------------
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
