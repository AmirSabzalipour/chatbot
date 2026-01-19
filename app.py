from pathlib import Path
import hashlib

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from together import Together

# =========================
# BASIC APP CONFIG
# =========================
st.set_page_config(page_title="Chatbot", layout="wide")

# =========================
# DEFAULTS
# =========================
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
TOP_K = 5
DEBUG = False

# =========================
# LAYOUT CONFIGURATION
# =========================
LEFT_PANEL_WIDTH_PX = 280
OUTER_LEFT_GAP_PX = 10
OUTER_RIGHT_GAP_PX = 20
OUTER_TOP_GAP_PX = 10
OUTER_BOTTOM_GAP_PX = 20
PANEL_GAP_PX = 20

RIGHT_PANEL_MAX_WIDTH_PX = 800
PANEL_PADDING_PX = 20
MAIN_PADDING_PX = 24

# ✅ Add extra top spacing ONLY for the right panel (your request)
RIGHT_PANEL_TOP_EXTRA_PX = 40

# Chat input
INPUT_BOTTOM_PX = 20  # ✅ give it a little breathing room
INPUT_WIDTH_PX = RIGHT_PANEL_MAX_WIDTH_PX - (MAIN_PADDING_PX * 2)

# Derived positions
RIGHT_PANEL_LEFT_PX = OUTER_LEFT_GAP_PX + LEFT_PANEL_WIDTH_PX + PANEL_GAP_PX
INPUT_LEFT_PX = RIGHT_PANEL_LEFT_PX + MAIN_PADDING_PX

# ✅ Height math (viewport-based, prevents outer scrollbar)
PANEL_HEIGHT_CSS = f"calc(100vh - {OUTER_TOP_GAP_PX}px - {OUTER_BOTTOM_GAP_PX}px)"
RIGHT_PANEL_HEIGHT_CSS = f"calc(100vh - {OUTER_TOP_GAP_PX + RIGHT_PANEL_TOP_EXTRA_PX}px - {OUTER_BOTTOM_GAP_PX}px)"

# ✅ Reserve space so the fixed input doesn’t cover last messages
CHAT_INPUT_RESERVED_PX = 120

# =========================
# GLOBAL CSS (CONSOLIDATED)
# =========================
st.markdown(
    f"""
<style>
/* =========================================================
   GLOBAL: hard stop outer scrolling + remove outer frame
========================================================= */

/* Box sizing */
*, *::before, *::after {{
  box-sizing: border-box;
}}

/* No outer scrolling anywhere */
html, body {{
  height: 100% !important;
  overflow: hidden !important;
  margin: 0 !important;
  padding: 0 !important;
  background: #ffffff !important;
}}

/* Streamlit app wrappers: remove padding/margins/borders/shadows */
.stApp,
div[data-testid="stAppViewContainer"],
div[data-testid="stAppViewBlockContainer"],
section.main,
.main,
section[data-testid="stMain"],
section.stMain {{
  height: 100vh !important;
  overflow: hidden !important;
  padding: 0 !important;
  margin: 0 !important;
  background: #ffffff !important;
  border: 0 !important;
  box-shadow: none !important;
  outline: none !important;
}}

/* Remove any “outer card” look from inner wrappers */
div[data-testid="stAppViewBlockContainer"] > div,
div[data-testid="stVerticalBlock"],
div[data-testid="stVerticalBlock"] > div {{
  padding: 0 !important;
  margin: 0 !important;
  border: 0 !important;
  box-shadow: none !important;
  outline: none !important;
  background: transparent !important;
}}

/* Hide Streamlit chrome */
#MainMenu,
header,
footer,
div[data-testid="stHeader"],
div[data-testid="stFooter"],
div[data-testid="stDecoration"],
div[data-testid="stToolbar"],
div[data-testid="stStatusWidget"],
div[data-testid="stToolbarActions"],
div[data-testid="stToolbarActionButton"] {{
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}}

/* Hide viewer badge / embed controls */
[class^="viewerBadge_"],
[class*=" viewerBadge_"],
.viewerBadge_container__1QSob,
.viewerBadge_link__1S137,
.viewerBadge_text__1JaDK,
div.container_lupux_1,
div[class^="container_"][class$="_1"] {{
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  min-height: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  border: 0 !important;
}}

/* Hide floating buttons */
button[title="View fullscreen"],
button[title="Open in new tab"],
button[title="Rerun"],
button[title="Settings"] {{
  display: none !important;
}}

/* Hide sidebar */
section[data-testid="stSidebar"] {{
  display: none !important;
}}

/* Remove bottom spacing containers Streamlit may add */
div[data-testid="stBottomBlockContainer"],
div[data-testid="stBottomBlockContainer"] > div {{
  background: transparent !important;
  padding: 0 !important;
  margin: 0 !important;
  border: 0 !important;
  height: 0 !important;
  min-height: 0 !important;
}}

/* =========================================================
   COLORS
   - Outer (3): white
   - Panels (1,2): light gray
========================================================= */
.block-container,
.left-panel {{
  background: #f3f4f6 !important; /* light gray */
}}

/* =========================================================
   LEFT PANEL
========================================================= */
.left-panel {{
  position: fixed;
  top: {OUTER_TOP_GAP_PX}px;
  left: {OUTER_LEFT_GAP_PX}px;
  width: {LEFT_PANEL_WIDTH_PX}px;
  height: {PANEL_HEIGHT_CSS};

  border: 0 !important;               /* no border */
  box-shadow: none !important;        /* no shadow */
  border-radius: 16px;

  padding: {PANEL_PADDING_PX}px;
  overflow-y: auto;
  z-index: 1000;
}}

.left-panel h3 {{
  margin: 0 0 1rem 0;
  font-size: 1.1rem;
  font-weight: 600;
  color: #1a1a1a;
}}

.left-panel ul {{
  list-style: none;
  padding: 0;
  margin: 0;
}}

.left-panel li {{
  padding: 0.75rem 1rem;
  margin: 0.25rem 0;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.2s;
  font-size: 0.9rem;
  color: #4a4a4a;
}}

.left-panel li:hover {{
  background: rgba(0,0,0,0.04);
}}

.left-panel li.active {{
  background: #e8f0fe;
  color: #1a73e8;
  font-weight: 500;
}}

/* =========================================================
   RIGHT PANEL (Main Chat)
   - Only this scrolls internally
   - No outer scroll
========================================================= */
.block-container {{
  max-width: {RIGHT_PANEL_MAX_WIDTH_PX}px !important;
  width: 100% !important;

  margin: {OUTER_TOP_GAP_PX}px {OUTER_RIGHT_GAP_PX}px {OUTER_BOTTOM_GAP_PX}px {RIGHT_PANEL_LEFT_PX}px !important;
  padding: {MAIN_PADDING_PX}px !important;

  border: 0 !important;               /* no border */
  box-shadow: none !important;        /* no shadow */
  border-radius: 16px !important;

  height: {RIGHT_PANEL_HEIGHT_CSS} !important;
  min-height: {RIGHT_PANEL_HEIGHT_CSS} !important;

  overflow-y: auto !important;

  /* Prevent fixed input overlapping last messages */
  padding-bottom: {CHAT_INPUT_RESERVED_PX}px !important;
}}

/* Chat message spacing */
div[data-testid="stChatMessage"] {{
  padding: 0.5rem 0 !important;
}}

/* =========================================================
   CHAT INPUT (fixed)
========================================================= */
div[data-testid="stChatInput"] {{
  position: fixed !important;
  bottom: {INPUT_BOTTOM_PX}px !important;
  left: {INPUT_LEFT_PX}px !important;
  width: {INPUT_WIDTH_PX}px !important;
  right: auto !important;
  max-width: none !important;
  padding: 0 !important;
  margin: 0 !important;
  z-index: 10000 !important;
}}

/* Make wrappers transparent */
div[data-testid="stChatInput"],
div[data-testid="stChatInput"] > div,
div[data-testid="stChatInput"] > div > div,
div[data-testid="stChatInput"] > div > div > div {{
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  outline: none !important;
  padding: 0 !important;
  margin: 0 !important;
}}

/* Style actual editable area */
div[data-testid="stChatInput"] textarea,
div[data-testid="stChatInput"] div[contenteditable="true"] {{
  background: #ffffff !important;
  border: 1px solid rgba(0,0,0,0.12) !important;
  border-radius: 24px !important;
  box-shadow: 0 2px 12px rgba(0,0,0,0.08) !important;
  padding: 0.7rem 1rem !important;
  font-size: 14px !important;
}}

/* If a faint top line remains anywhere */
div[data-testid="stAppViewBlockContainer"] * {{
  border-top: 0 !important;
}}
</style>
""",
    unsafe_allow_html=True,
)

# LEFT PANEL HTML
st.markdown(
    """
<div class="left-panel">
  <h3>Conversations</h3>
  <ul>
    <li class="active">Current Chat</li>
    <li>Previous Chat 1</li>
    <li>Previous Chat 2</li>
    <li>Previous Chat 3</li>
  </ul>
</div>
""",
    unsafe_allow_html=True,
)
