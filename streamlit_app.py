# streamlit_app.py
import os
from datetime import datetime
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import math
import base64
from pathlib import Path
import matplotlib as mpl

# Import your core pipelines
import landfill_pollution_detection_v2 as core

def set_app_background(image_path: str, blur_px: int = 0, dim: float = 0.0):
    """
    Sets a full-page background image using base64 CSS injection.
    - image_path: path relative to the repo root (e.g., 'assets/app_bg.jpg')
    - blur_px: optional CSS blur for readability (e.g., 2–6)
    - dim: 0.0–1.0 overlay darkening for contrast (0 = none, 0.3 = subtle)
    """
    img_file = Path(image_path)
    if not img_file.exists():
        st.warning(f"Background image not found: {image_path}")
        return
    b64 = base64.b64encode(img_file.read_bytes()).decode()
    # Optional dark overlay using linear-gradient, then the image
    overlay = f"linear-gradient(rgba(0,0,0,{dim}), rgba(0,0,0,{dim})), " if dim > 0 else ""
    css = f"""
    <style>
      /* Page background */
      [data-testid="stAppViewContainer"] {{
        background: {overlay} url("data:image/{img_file.suffix[1:]};base64,{b64}") center/cover no-repeat fixed;
        {'backdrop-filter: blur(' + str(blur_px) + 'px);' if blur_px > 0 else ''}
      }}
      /* Transparent header */
      [data-testid="stHeader"] {{ background: rgba(0,0,0,0); }}
      /* Slightly translucent main block for readability */
      .main .block-container {{
        background: rgba(255,255,255,0.85);
        border-radius: 12px;
        padding: 1rem 1.25rem;
      }}
      /* Sidebar readability */
      [data-testid="stSidebar"] > div:first-child {{
        background: rgba(255,255,255,0.85);
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def enforce_text_color(color="#0f172a", link_color="#2563eb", sidebar_color=None):
    """
    Force app text colors via CSS.
    - color: main text color (e.g., #0f172a for dark slate, #f8fafc for near-white)
    - link_color: links
    - sidebar_color: sidebar text (defaults to color)
    """
    if sidebar_color is None:
        sidebar_color = color

    st.markdown(f"""
    <style>
      /* Most text in the app */
      [data-testid="stAppViewContainer"] * {{
        color: {color} !important;
      }}

      /* Sidebar */
      [data-testid="stSidebar"] * {{
        color: {sidebar_color} !important;
      }}

      /* Links */
      a, a:visited {{
        color: {link_color} !important;
      }}

      /* Metric widget numbers and labels */
      [data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] * {{
        color: {color} !important;
      }}

      /* Expander titles */
      details summary {{
        color: {color} !important;
      }}
    </style>
    """, unsafe_allow_html=True)

    # Match Matplotlib plot text to the chosen color
    mpl.rcParams.update({
        "text.color": color,
        "axes.labelcolor": color,
        "xtick.color": color,
        "ytick.color": color,
        "figure.facecolor": (1,1,1,0),  # transparent-ish
        "axes.facecolor": (1,1,1,0.85)  # readable panel for charts
    })

set_app_background("assets/app_image.jpg", blur_px=0, dim=0.45)
enforce_text_color("#f8fafc", link_color="#93c5fd")  # light text

# --- Make UI readable over a light/white sidebar & inputs ---
def apply_ui_readability(
    base_text="#0f172a",           # dark slate for text
    link_text="#2563eb",           # link blue
    button_bg="#2563eb",           # primary button background
    button_text="#ffffff",         # primary button text
    button_border="#1d4ed8",       # primary button border
    input_border="#cbd5e1",        # light slate border for inputs
    placeholder="#64748b"          # placeholder color
):
    st.markdown(f"""
    <style>
      /* ---------- Sidebar: force dark text on white bg ---------- */
      [data-testid="stSidebar"] * {{
        color: {base_text} !important;
      }}

      /* Sidebar headings/expanders */
      [data-testid="stSidebar"] h1, 
      [data-testid="stSidebar"] h2, 
      [data-testid="stSidebar"] h3, 
      [data-testid="stSidebar"] h4, 
      [data-testid="stSidebar"] h5, 
      [data-testid="stSidebar"] h6,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] span {{
        color: {base_text} !important;
      }}

      /* ---------- Inputs: selectbox / number_input / text inputs --- */
      /* Labels */
      label, .stSelectbox label, .stNumberInput label, .stTextInput label {{
        color: {base_text} !important;
      }}

      /* Native inputs */
      input, textarea, select {{
        color: {base_text} !important;
        border-color: {input_border} !important;
      }}

      /* Streamlit's selectbox uses BaseWeb select */
      [data-baseweb="select"] * {{
        color: {base_text} !important;
      }}
      /* Dropdown menu items for select */
      [data-baseweb="menu"] * {{
        color: {base_text} !important;
      }}
      /* Placeholder text */
      ::placeholder {{
        color: {placeholder} !important;
        opacity: 1 !important;
      }}

      /* ---------- Buttons: make them visible and consistent ---------- */
      div.stButton > button {{
        background-color: {button_bg} !important;
        color: {button_text} !important;
        border: 1px solid {button_border} !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.9rem !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 2px rgba(16,24,40,0.07) !important;
      }}
      div.stButton > button:hover {{
        filter: brightness(1.05) !important;
      }}
      div.stButton > button:active {{
        filter: brightness(0.95) !important;
      }}

      /* Link color globally */
      a, a:visited {{ color: {link_text} !important; }}

      /* Metrics (numbers and labels) */
      [data-testid="stMetricValue"],
      [data-testid="stMetricLabel"],
      [data-testid="stMetricDelta"] * {{
        color: {base_text} !important;
      }}

      /* Expander titles */
      details summary {{
        color: {base_text} !important;
      }}

      /* Table text */
      .stTable, .stTable * {{ color: {base_text} !important; }}

    </style>
    """, unsafe_allow_html=True)

apply_ui_readability()


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
            if idx >= len(items): 
                continue
            label, val = items[idx]
            with cols_row[c]:
                st.markdown(
                    f"""
                    <div style="border:1px solid rgba(200,200,200,.4); border-radius:10px; padding:.6rem .8rem; margin-bottom:.5rem;">
                        <div style="font-size:.85rem; color:#666;">{label}</div>
                        <div style="font-size:1.1rem; font-weight:600;">{val}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

def _get_query_page(default="home"):
    qp = st.query_params
    return (qp.get("page", default) or default).lower()

def _nav(page_name: str):
    st.query_params["page"] = page_name.lower()
    st.rerun()

def _format_latlon(lat, lon):
    try:
        return f"{float(lat):.4f}, {float(lon):.4f}"
    except Exception:
        return f"{lat}, {lon}"

# Keep last results in session state
if "last_run" not in st.session_state:
    st.session_state["last_run"] = None
if "notifications" not in st.session_state:
    st.session_state["notifications"] = True

# ---------- AQI visualization helpers ----------
AQI_CATS = [
    (0,   50,  "Good",                              "#00E400"),
    (51,  100, "Moderate",                          "#FFFF00"),
    (101, 150, "Unhealthy for Sensitive Groups",    "#FF7E00"),
    (151, 200, "Unhealthy",                         "#FF0000"),
    (201, 300, "Very Unhealthy",                    "#8F3F97"),
    (301, 500, "Hazardous",                         "#7E0023"),
]

def aqi_category_and_color(value: float | None):
    if value is None:
        return "—", "#999999"
    v = max(0, min(500, float(value)))
    for lo, hi, name, hexcol in AQI_CATS:
        if lo <= v <= hi:
            return name, hexcol
    return "Hazardous", "#7E0023"

def plot_aqi_bar(value: float | None, show_label=True, height=0.35):
    """
    Draw an EPA-style horizontal AQI bar [0..500] with a vertical marker at `value`.
    Returns a matplotlib figure for st.pyplot.
    """
    fig, ax = plt.subplots(figsize=(7.5, 1.0))
    left = 0
    for lo, hi, name, hexcol in AQI_CATS:
        width = hi - lo
        ax.barh(y=0, width=width, left=lo, height=height, color=hexcol, edgecolor="black", linewidth=0.5)
    ax.set_xlim(0, 500)
    ax.set_yticks([])
    ax.set_xlabel("AQI", fontsize=9)
    ax.tick_params(axis='x', labelsize=8)

    if value is not None:
        v = max(0, min(500, float(value)))
        ax.axvline(v, color="black", linewidth=2)
        if show_label:
            cat, _ = aqi_category_and_color(v)
            ax.text(v, height + 0.12, f"{int(round(v))} • {cat}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig

# ---------- Layout ----------
st.set_page_config(page_title=APP_TITLE, page_icon="🔥", layout="wide")
page_key = _get_query_page("home")
valid_keys = ["home", "monitor", "results"]
try:
    idx = valid_keys.index(page_key)
except ValueError:
    idx = 0
page = st.sidebar.radio("Pages", ["Home", "Monitor", "Results"], index=idx)

st.title(APP_TITLE)

# ---------- PAGE: HOME ----------
if page.lower() == "home":
    st.subheader("What this app does")
    st.write(APP_PURPOSE)

    st.markdown("### How it works")
    st.markdown(
        """
        1. **FIRMS check** near your landfill coordinates.  
        2. If fire detected → use **TEMPO** + **OpenAQ/PurpleAir** to fetch measurements.  
        3. Compute **satellite AQI**, **ground AQI**, and **fused AQI**.  
        4. Fetch **current weather** from OpenWeatherMap.  
        5. **Forecast AQI** for 72 hours.  
        """
    )
    st.info("Run a live check from the Monitor page.")

    # Centered call-to-action button
    cta1, cta2, cta3 = st.columns([1, 1, 1])
    with cta2:
        if st.button("➡️ Go to Monitor", key="cta_monitor", type="primary"):
            _nav("monitor")  # uses your existing nav helper


# ---------- PAGE: MONITOR ----------
elif page.lower() == "monitor":
    st.subheader("Monitor: run a check for fires and AQI")

    # Defaults to Calabasas landfill (demo)
    #demo_lat = core.LANDFILL.get("lat", 34.1439)
    #demo_lon = core.LANDFILL.get("lon", -118.6615)

    # --- Landfill selection (dropdown) ---
    from landfill_pollution_detection_v2 import available_landfills, set_landfill_by_name  # adjust import path if needed

    names = available_landfills()
    default_idx = names.index("Calabasas Landfill") if "Calabasas Landfill" in names else 0
    selected_name = st.selectbox("Choose landfill", options=names, index=default_idx)

    lf = set_landfill_by_name(selected_name)
    st.session_state["selected_landfill_name"] = lf.get("name", selected_name)
    lat, lon = float(lf["lat"]), float(lf["lon"])

    
   # Show coords + current radius (read-only on this row)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.metric("Latitude", f"{lat:.6f}")
    with c2:
        st.metric("Longitude", f"{lon:.6f}")
    with c3:
        st.metric("Radius (km)", st.session_state.get("radius_km", 5))

    # Editable inputs
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        lat = st.number_input("Latitude", value=float(lat), format="%.6f")
    with col2:
        lon = st.number_input("Longitude", value=float(lon), format="%.6f")
    with col3:
        radius_km = st.slider(
            "Detection radius (km)",
            min_value=1, max_value=20, value=st.session_state.get("radius_km", 5), step=1,
            key="radius_km_main"
        )
    st.session_state["radius_km"] = radius_km


    st.checkbox("Enable in-app notifications", value=st.session_state["notifications"], key="notifications")

    st.markdown("---")
    run_col, info_col = st.columns([1,2])
    with run_col:
        run_btn = st.button("🚀 Check now", type="primary")
    with info_col:
        st.caption(
            "On click, the app runs FIRMS check. If a fire is detected, it fetches data to get current AQI "
            "and forecast AQI for next 72 hours."
        )

    if run_btn:
        with st.spinner("Checking for fire at chosen landfill…"):
            out = core.run_workflow_if_fire(
                lat=float(lat),
                lon=float(lon),
                radius_km=float(radius_km),
                run_parallel=True,
                forecast_horizon_h=72
            )


            out["landfill_name"] = st.session_state.get("selected_landfill_name", selected_name)
            out["lat"] = float(lat)
            out["lon"] = float(lon)

            st.session_state["last_run"] = out

        if out.get("fire_detected"):
            if st.session_state["notifications"]:
                st.toast("🔥 Fire detected near the chosen landfill. Click **Results** to see AQI.", icon="🔥")
            landfill_name = st.session_state.get("selected_landfill_name", "selected site")
            st.warning(f"🔥 Fire detected near **{landfill_name}** — {_format_latlon(lat, lon)}.")
        else:
            st.info("No ongoing fire detected near the chosen landfill.")

    # ----------- Last run summary (visual) -----------
    if st.session_state["last_run"] is not None:
        st.markdown("---")
        st.markdown("### Last run summary")

        lr = st.session_state["last_run"]
        run_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
        fire = bool(lr.get("fire_detected"))
        fused = lr.get("fused_aqi")


        landfill_name = lr.get("landfill_name", st.session_state.get("selected_landfill_name", "—"))
        coords_str = _format_latlon(lr.get("lat", None) or lat, lr.get("lon", None) or lon)
        st.markdown(f"**Landfill checked:** {landfill_name}  \n**Coordinates:** {coords_str}")

        c1, c2, c3 = st.columns([1,1,3])
        with c1:
            st.markdown("**Run time (UTC)**")
            st.write(run_time)
        with c2:
            st.markdown("**Fire detected**")
            if fire:
                st.markdown('<span style="color:#D32F2F;font-weight:700;">YES</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#388E3C;font-weight:700;">NO</span>', unsafe_allow_html=True)
        with c3:
            st.markdown("**Fused AQI**")
            cat, col = aqi_category_and_color(fused)
            if fused is None:
                st.write("—")
            else:
                st.write(f"{int(round(fused))} • {cat}")
            fig = plot_aqi_bar(fused)
            st.pyplot(fig, clear_figure=True)

        # Navigation buttons (no dead hyperlinks)
        nav1, nav2 = st.columns([1,1])
        with nav1:
            if st.button("➡️ Go to Results"):
                _nav("results")
        with nav2:
            if st.button("🏠 Back to Home"):
                _nav("home")

# ---------- PAGE: RESULTS ----------
elif page.lower() == "results":
    st.subheader("Results")

    # --- Landfill name (if chosen on Monitor page) ---
    landfill_name = st.session_state.get("selected_landfill_name")
    if landfill_name:
        st.markdown(f"**Landfill checked:** {landfill_name}")

    # --- Last run results ---
    lr = st.session_state.get("last_run")

    if not lr:
        st.info("No results yet. Go to **Monitor** and click **Check now**.")
        col_a, col_b = st.columns([1, 3])
        with col_a:
            if st.button("Go to Monitor", key="go_to_monitor_results_empty"):
                new_qp = dict(st.query_params)
                new_qp["page"] = "Monitor"
                st.query_params = new_qp
                st.rerun()
        #with col_b:
        #    st.markdown("Or click this link: [Open Monitor](?page=Monitor)")

    else:
        if not lr.get("fire_detected"):
            st.success("✅ No ongoing fire.")
            st.caption("If you believe there should be one, try increasing the radius on the Monitor page and re-run.")

            colb1, colb2 = st.columns([1,1])
            with colb1:
                if st.button("🔁 Run another check"):
                    _nav("monitor")
            with colb2:
                if st.button("🏠 Home"):
                    _nav("home")

        else:
            # --- Current fused AQI + color bar ---
            fused = lr.get("fused_aqi")
            cA, cB = st.columns([1,3])
            with cA:
                st.markdown("**Current fused AQI**")
                cat, col = aqi_category_and_color(fused)
                st.markdown(
                    f'<div style="font-size:28px;font-weight:700;color:{col};">{int(round(fused)) if fused is not None else "—"}</div>'
                    f'<div style="color:#666;">{cat if fused is not None else ""}</div>',
                    unsafe_allow_html=True
                )
            with cB:
                fig = plot_aqi_bar(fused)
                st.pyplot(fig, clear_figure=True)

            # --- Predicted AQI + chart ---
            fdf = lr.get("forecast_df")
            if isinstance(fdf, pd.DataFrame) and not fdf.empty and {"datetime","pred_AQI"}.issubset(fdf.columns):
                next_hour = float(fdf["pred_AQI"].iloc[0])
                st.markdown("#### 72-hour forecast")
                plot_df = fdf.copy().set_index("datetime")[["pred_AQI"]]
                st.line_chart(plot_df)
                st.caption(f"Next-hour predicted AQI: **{int(round(next_hour))}**")
            else:
                st.warning("No forecast data available (model may have skipped training).")

            # --- More details: friendly cards instead of raw JSON ---
            with st.expander("More details", expanded=False):
                satellite_current = lr.get("satellite_current", {}) or {}
                ground_current    = lr.get("ground_current", {}) or {}
                sat_breakdown     = lr.get("sat_breakdown", {}) or {}
                ground_breakdown  = lr.get("ground_breakdown", {}) or {}

                # Satellite (TEMPO): show all keys even if missing
                sat_expected = [
                    ("NO2 (molec/cm²)",          satellite_current.get("NO2"),  None),
                    ("O3 trop (molec/cm²)",      satellite_current.get("O3"),   None),
                    ("HCHO (molec/cm²)",         satellite_current.get("HCHO"), None),
                    ("UV Aerosol Index (UVAI)",  satellite_current.get("AER"),  None),
                ]
                sat_items = [(label, _fmt_value(val, unit)) for (label, val, unit) in sat_expected]
                render_cards("Satellite (TEMPO)", sat_items, cols=2)

                # Ground (OpenAQ / PurpleAir fallback): always show all six
                ground_expected = [
                    ("PM2.5 (µg/m³)", ground_current.get("PM2.5"), "µg/m³"),
                    ("PM10 (µg/m³)",  ground_current.get("PM10"), "µg/m³"),
                    ("NO₂ (ppb)",     ground_current.get("NO2"),  "ppb"),
                    ("O₃ (ppb)",      ground_current.get("O3"),   "ppb"),
                    ("CO (ppm)",      ground_current.get("CO"),   "ppm"),
                    ("SO₂ (ppb)",     ground_current.get("SO2"),  "ppb"),
                ]
                ground_items = [(label, _fmt_value(val, unit)) for (label, val, unit) in ground_expected]
                render_cards("Ground (OpenAQ / PurpleAir)", ground_items, cols=3)

                # AQI sub-indices (optional)
                if isinstance(sat_breakdown, dict) and sat_breakdown:
                    s_items = [(k, _fmt_value(v, None, digits=0)) for k, v in sat_breakdown.items()]
                    render_cards("Satellite AQI sub-indices", s_items, cols=3)

                if isinstance(ground_breakdown, dict) and ground_breakdown:
                    display_order = ["PM2.5", "PM10", "NO2", "O3", "CO", "SO2"]
                    g_items = []
                    for pol in display_order:
                        sub = ground_breakdown.get(pol, {})
                        val = sub.get("subindex") if isinstance(sub, dict) else None
                        g_items.append((pol, _fmt_value(val, None, digits=0)))
                    render_cards("Ground AQI sub-indices", g_items, cols=3)

                # Training diagnostics (if available)
                diag_left, diag_right = st.columns([1,2])
                with diag_left:
                    conf = lr.get("confidence_score")
                    if conf is None:
                        st.metric("Forecast confidence", "—")
                    else:
                        # nice formatting: 0–100, no decimals
                        st.metric("Forecast confidence", f"{int(round(conf))} / 100")
                with diag_right:
                    reason = lr.get("training_fallback_reason") or "—"
                    st.caption(f"Training note: {reason}")

                st.caption("Values sourced from NASA TEMPO (via Harmony) and OpenAQ v3 (with PurpleAir only for PM2.5 when missing).")

            # --- Navigation ---
            colb1, colb2 = st.columns([1,1])
            with colb1:
                if st.button("🔁 Run another check"):
                    _nav("monitor")
            with colb2:
                if st.button("🏠 Home"):
                    _nav("home")

# Footer
st.markdown("---")
st.caption(
    "Demo app — uses public data sources (FIRMS / TEMPO via Harmony / OpenAQ / PurpleAir / OpenWeatherMap). "
    "If a service is unavailable or authentication fails, the demo may fall back to mock data."
)
