# Use of AI for code review

# streamlit_app.py
# -------------------------------------------------------------------
# Landfills Fire & Air Quality Monitor (Streamlit app)
# -------------------------------------------------------------------

import os
import math
from datetime import datetime
from pathlib import Path
import base64

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st

# Page config MUST be set before any other st.* calls
st.set_page_config(page_title="Landfills Fire & Air Quality Monitor", page_icon="🔥", layout="wide")

# Import your core pipelines AFTER page config
import landfill_pollution_detection_v2 as core


# -------------------------------------------------------------------
# THEME & STYLES
# -------------------------------------------------------------------
def apply_sane_theme(
    bg_image="assets/app_image.jpg",
    bg_dim=0.5,                    # dark overlay over background image
    main_card_bg="rgba(0,0,0,0.78)",   # dark translucent main card
    main_text="#f8fafc",           # near-white text in main card
    main_link="#cfe4ff",
    text_shadow="0 1px 2px rgba(0,0,0,.75)",  # subtle halo on dark bg
    sidebar_text="#0f172a",        # dark text (sidebar is light)
    button_bg="#2563eb",
    button_text="#ffffff",
    button_border="#1d4ed8",
    input_border="#cbd5e1",
    placeholder="#64748b",
    blur_px=0
):
    # --- Background image + overlay ---
    css_bg = ""
    img_file = Path(bg_image)
    if img_file.exists():
        b64 = base64.b64encode(img_file.read_bytes()).decode()
        overlay = f"linear-gradient(rgba(0,0,0,{bg_dim}), rgba(0,0,0,{bg_dim})), " if bg_dim > 0 else ""
        css_bg = f"""
        [data-testid="stAppViewContainer"] {{
          background: {overlay} url("data:image/{img_file.suffix[1:]};base64,{b64}") center/cover no-repeat fixed;
          {'backdrop-filter: blur(' + str(blur_px) + 'px);' if blur_px > 0 else ''}
        }}
        """

    st.markdown(f"""
    <style>
      /* Background (with optional overlay) */
      {css_bg}

      /* Transparent top header */
      [data-testid="stHeader"] {{
        background: rgba(0,0,0,0) !important;
      }}

      /* -------- Main content card -------- */
      /* Target the central container */
      [data-testid="stAppViewContainer"] .main .block-container {{
        background: {main_card_bg} !important;
        border-radius: 12px !important;
        padding: 1rem 1.25rem !important;
      }}

      /* Make virtually everything INSIDE the main card bright & readable */
      [data-testid="stAppViewContainer"] .main .block-container :is(
        p, li, span, div, code, kbd, strong, em, small, pre, table, th, td
      ) {{
        color: {main_text} !important;
        text-shadow: {text_shadow} !important;
      }}
      [data-testid="stAppViewContainer"] .main .block-container :is(h1,h2,h3,h4,h5,h6) {{
        color: {main_text} !important;
        text-shadow: {text_shadow} !important;
      }}

      /* Links in main card */
      [data-testid="stAppViewContainer"] .main .block-container a,
      [data-testid="stAppViewContainer"] .main .block-container a:visited {{
        color: {main_link} !important;
        text-shadow: none !important;
      }}

      /* Alerts in main card: keep light background with dark text for clarity */
      [data-testid="stAppViewContainer"] .main .block-container .stAlert > div {{
        background: rgba(255,255,255,0.98) !important;
        border-radius: 10px !important;
      }}
      [data-testid="stAppViewContainer"] .main .block-container .stAlert,
      [data-testid="stAppViewContainer"] .main .block-container .stAlert * {{
        color: #0f172a !important;
        text-shadow: none !important;
      }}

      /* Metrics & tables in main card */
      [data-testid="stAppViewContainer"] .main .block-container [data-testid="stMetricValue"],
      [data-testid="stAppViewContainer"] .main .block-container [data-testid="stMetricLabel"],
      [data-testid="stAppViewContainer"] .main .block-container [data-testid="stMetricDelta"] * {{
        color: {main_text} !important;
        text-shadow: {text_shadow} !important;
      }}
      [data-testid="stAppViewContainer"] .main .block-container .stTable,
      [data-testid="stAppViewContainer"] .main .block-container .stTable * {{
        color: {main_text} !important;
        text-shadow: {text_shadow} !important;
      }}

      /* -------- Sidebar stays light with dark text -------- */
      [data-testid="stSidebar"] > div:first-child {{
        background: rgba(255,255,255,0.9) !important;
      }}
      [data-testid="stSidebar"] * {{
        color: {sidebar_text} !important;
        text-shadow: none !important;
      }}

      /* -------- Inputs (main + sidebar): dark text on white widgets -------- */
      /* Scoped to Streamlit input wrappers instead of global <label> */
      .stNumberInput label,
      .stTextInput label,
      .stSelectbox label,
      .stSlider label,
      .stDateInput label {{ color: {sidebar_text} !important; text-shadow: none !important; }}

      input, textarea, select {{
        color: {sidebar_text} !important;
        border-color: {input_border} !important;
      }}
      [data-baseweb="select"] * {{ color: {sidebar_text} !important; }}
      [data-baseweb="menu"] * {{ color: {sidebar_text} !important; }}
      ::placeholder {{ color: {placeholder} !important; opacity: 1 !important; }}

      /* Buttons (both places) */
      div.stButton > button {{
        background-color: {button_bg} !important;
        color: {button_text} !important;
        border: 1px solid {button_border} !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.9rem !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 2px rgba(16,24,40,0.07) !important;
      }}
      div.stButton > button:hover {{ filter: brightness(1.05) !important; }}
      div.stButton > button:active {{ filter: brightness(0.95) !important; }}
    </style>
    """, unsafe_allow_html=True)

    # Align Matplotlib with bright-on-dark main card
    mpl.rcParams.update({
        "text.color": main_text,
        "axes.labelcolor": main_text,
        "xtick.color": main_text,
        "ytick.color": main_text,
        "axes.edgecolor": main_text,
        "grid.color": "#9aa4b2",
        "figure.facecolor": (0,0,0,0),
        "axes.facecolor": (0,0,0,0),
    })


# Call once, BEFORE rendering any content
apply_sane_theme(
    bg_image="assets/app_image.jpg",
    bg_dim=0.5,
    main_card_bg="rgba(0,0,0,0.78)",
    main_text="#f8fafc",
    main_link="#cfe4ff",
)


# -------------------------------------------------------------------
# APP TEXT
# -------------------------------------------------------------------
APP_TITLE = "Landfills Fire & Air Quality Monitor"
APP_PURPOSE = (
    "Watches for landfills' fires (NASA FIRMS). "
    "If a fire is detected, it fetches TEMPO satellite data and local ground measurements, "
    "computes a fused AQI from both satellite AQI and ground AQI. It then trains a short-term model "
    "to forecast AQI for the next 72 hours."
)

# ---------- Helpers ----------
def _fmt_value(val, unit=None, digits=2):
    """Return a friendly string or 'Could not get measurement' for missing values."""
    if val is None:
        return "Could not get measurement"
    try:
        v = float(val)
        if math.isnan(v):
            return "Could not get measurement"
        s = f"{v:.{digits}f}"
    except Exception:
        return "Could not get measurement"
    return f"{s} {unit}" if unit else s

def render_cards(title, items, cols=3):
    """
    items: list of tuples (label, value_str) already formatted
    Renders as compact 'cards' using columns.
    """
    st.markdown(f"#### {title}")
    rows = (len(items) + cols - 1) // cols
    for r in range(rows):
        cols_row = st.columns(cols)
        for c in range(cols):
            idx = r * cols + c
