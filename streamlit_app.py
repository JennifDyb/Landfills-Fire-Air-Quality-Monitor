# streamlit_app.py
# -------------------------------------------------------------
# Landfills Fire & Air Quality Monitor (Streamlit app)
# Robust background + readable content card styling
# -------------------------------------------------------------

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
import base64

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st

# Page config MUST be first
st.set_page_config(page_title="Landfills Fire & Air Quality Monitor", page_icon="🔥", layout="wide")

# Try to import core pipelines; keep UI alive even if import fails
try:
    import landfill_pollution_detection_v2 as core
except Exception as e:
    core = None
    st.error("Could not import core module `landfill_pollution_detection_v2`. The UI will load, but actions may fail.")
    st.exception(e)


# ----------------------- Styling helpers -----------------------
def set_background_image(image_path: str, dim: float = 0.45, blur_px: int = 0) -> None:
    """
    Set a page-wide background image with optional dark overlay (dim) and blur.
    This never touches main content colors; it just sets the backdrop.
    """
    img = Path(image_path)
    bg_css = ""
    if img.exists():
        b64 = base64.b64encode(img.read_bytes()).decode()
        overlay = f"linear-gradient(rgba(0,0,0,{dim}), rgba(0,0,0,{dim})), " if dim > 0 else ""
        bg_css = f"""
        [data-testid="stAppViewContainer"] {{
          background: {overlay} url("data:image/{img.suffix[1:]};base64,{b64}") center/cover no-repeat fixed;
          {'backdrop-filter: blur(' + str(blur_px) + 'px);' if blur_px > 0 else ''}
        }}
        """
    st.markdown(f"""
    <style>
      {bg_css}
      /* Transparent header so the image shows through */
      [data-testid="stHeader"] {{ background: rgba(0,0,0,0) !important; }}

      /* Sidebar stays light with dark text */
      [data-testid="stSidebar"] > div:first-child {{
        background: rgba(255,255,255,0.92) !important;
      }}
      [data-testid="stSidebar"] * {{
        color: #0f172a !important;
        text-shadow: none !important;
      }}

      /* Inputs (both main & sidebar): dark text on white controls */
      .stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label, .stDateInput label {{
        color: #0f172a !important;
        text-shadow: none !important;
      }}
      input, textarea, select {{ color: #0f172a !important; }}
      [data-baseweb="select"] * {{ color: #0f172a !important; }}
      [data-baseweb="menu"] * {{ color: #0f172a !important; }}
      ::placeholder {{ color: #64748b !important; opacity: 1 !important; }}

      /* Buttons (everywhere) */
      div.stButton > button {{
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #1d4ed8 !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.9rem !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 2px rgba(16,24,40,0.07) !important;
      }}
      div.stButton > button:hover {{ filter: brightness(1.05) !important; }}
      div.stButton > button:active {{ filter: brightness(0.95) !important; }}

      /* ------- Our content card ------- */
      .content-card {{
        background: rgba(0,0,0,0.78);      /* dark translucent panel */
        color: #f8fafc;                    /* bright text */
        border-radius: 12px;
        padding: 16px 20px;
      }}
      .content-card :is(p, li, span, div, code, kbd, strong, em, small, pre, table, th, td) {{
        color: #f8fafc !important;
        text-shadow: 0 1px 2px rgba(0,0,0,.75);
      }}
      .content-card :is(h1,h2,h3,h4,h5,h6) {{
        color: #f8fafc !important;
        text-shadow: 0 1px 2px rgba(0,0,0,.75);
      }}
      .content-card a, .content-card a:visited {{
        color: #cfe4ff !important;
        text-shadow: none !important;
      }}
      .content-card .stAlert > div {{
        background: rgba(255,255,255,0.98) !important;
        border-radius: 10px !important;
      }}
      .content-card .stAlert, .content-card .stAlert * {{
        color: #0f172a !important;
        text-shadow: none !important;
      }}
      .content-card [data-testid="stMetricValue"],
      .content-card [data-testid="stMetricLabel"],
      .content-card [data-testid="stMetricDelta"] * {{
        color: #f8fafc !important;
        text-shadow: 0 1px 2px rgba(0,0,0,.75) !important;
      }}
      .content-card .stTable, .content-card .stTable * {{
        color: #f8fafc !important;
        text-shadow: 0 1px 2px rgba(0,0,0,.75) !important;
      }}
    </style>
    """, unsafe_allow_html=True)

    # Matplotlib axes/text to match bright-on-dark card
    mpl.rcParams.update({
        "text.color": "#f8fafc",
        "axes.labelcolor": "#f8fafc",
        "xtick.color": "#f8fafc",
        "ytick.color": "#f8fafc",
        "axes.edgecolor": "#f8fafc",
        "grid.color": "#9aa4b2",
        "figure.facecolor": (0,0,0,0),
        "axes.facecolor": (0,0,0,0),
    })


def start_card():
    st.markdown('<div class="content-card">', unsafe_allow_html=True)


def end_card():
    st.markdown('</div>', unsafe_allow_html=True)


# Apply background (does not affect text)
set_background_image("assets/app_image.jpg", dim=0.5, blur_px=0)


# ----------------------- App text / helpers -----------------------
APP_TITLE = "Landfills Fire & Air Quality Monitor"
APP_PURPOSE = (
    "Watches for landfills' fires (NASA FIRMS). "
    "If a fire is detected, it fetches TEMPO satellite data and local ground measurements, "
    "computes a fused AQI from both satellite AQI and ground AQI, then trains a short-term model "
    "to forecast AQI for the next 72 hours."
)

def _fmt_value(val, unit=None, digits=2):
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
                    """
                    <div style="border:1px solid rgba(200,200,200,.35); border-radius:10px; padding:.6rem .8rem; margin-bottom:.5rem;">
                        <div style="font-size:.85rem; opacity:.85;">{label}</div>
                        <div style="font-size:1.1rem; font-weight:600;">{val}</div>
                    </div>
                    """.format(label=label, val=val),
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

# Session state
if "last_run" not in st.session_state:
    st.session_state["last_run"] = None
if "notifications" not in st.session_state:
    st.session_state["notifications"] = True

# AQI helpers / plot
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
    fig, ax = plt.subplots(figsize=(7.5, 1.0))
    for lo, hi, name, hexcol in AQI_CATS:
        ax.barh(y=0, width=hi - lo, left=lo, height=height, color=hexcol, edgecolor="black", linewidth=0.5)
    ax.set_xlim(0, 500); ax.set_yticks([]); ax.set_xlabel("AQI", fontsize=9); ax.tick_params(axis='x', labelsize=8)
    if value is not None:
        v = max(0, min(500, float(value)))
        ax.axvline(v, color="black", linewidth=2)
        if show_label:
            cat, _ = aqi_category_and_color(v)
            ax.text(v, height + 0.12, f"{int(round(v))} • {cat}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    return fig


# ----------------------- Layout & Pages -----------------------
page_key = _get_query_page("home")
valid_keys = ["home", "monitor", "results"]
try:
    idx = valid_keys.index(page_key)
except ValueError:
    idx = 0
page = st.sidebar.radio("Pages", ["Home", "Monitor", "Results"], index=idx)

# Title OUTSIDE the card so it can sit on the background — then card content
st.title("Landfills Fire & Air Quality Monitor")

# ----------------------- HOME -----------------------
if page.lower() == "home":
    start_card()
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

    cta1, cta2, cta3 = st.columns([1, 1, 1])
    with cta2:
        if st.button("➡️ Go to Monitor", key="cta_monitor", type="primary"):
            _nav("monitor")
    end_card()


# ----------------------- MONITOR -----------------------
elif page.lower() == "monitor":
    start_card()
    st.subheader("Monitor: run a check for fires and AQI")

    # Landfill selection (drop-down)
    if core is not None:
        try:
            from landfill_pollution_detection_v2 import available_landfills, set_landfill_by_name
            names = available_landfills()
        except Exception:
            names = ["Calabasas Landfill"]
    else:
        names = ["Calabasas Landfill"]

    default_idx = names.index("Calabasas Landfill") if "Calabasas Landfill" in names else 0
    selected_name = st.selectbox("Choose landfill", options=names, index=default_idx)

    if core is not None:
        try:
            from landfill_pollution_detection_v2 import set_landfill_by_name
            lf = set_landfill_by_name(selected_name)
        except Exception:
            lf = {"name": selected_name, "lat": 34.1208, "lon": -118.6994}  # Calabasas official-ish
    else:
        lf = {"name": selected_name, "lat": 34.1208, "lon": -118.6994}

    st.session_state["selected_landfill_name"] = lf.get("name", selected_name)
    lat, lon = float(lf["lat"]), float(lf["lon"])

    # Show coords + current radius (read-only row)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1: st.metric("Latitude", f"{lat:.6f}")
    with c2: st.metric("Longitude", f"{lon:.6f}")
    with c3: st.metric("Radius (km)", st.session_state.get("radius_km", 5))

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
        st.caption("On click, the app runs FIRMS check. If a fire is detected, it fetches data to get current AQI and forecast AQI for next 72 hours.")
    end_card()

    # The button should be outside the card? It can be inside; logic below runs regardless.
    if run_btn and core is not None:
        with st.spinner("Checking for fire at chosen landfill…"):
            out = core.run_workflow_if_fire(
                lat=float(lat),
                lon=float(lon),
                radius_km=float(radius_km),
                run_parallel=True,
                forecast_horizon_h=72
            )
            out["landfill_name"] = st.session_state.get("selected_landfill_name", selected_name)
            out["lat"] = float(lat); out["lon"] = float(lon)
            st.session_state["last_run"] = out

        start_card()
        if out.get("fire_detected"):
            if st.session_state["notifications"]:
                st.toast("🔥 Fire detected near the chosen landfill. Click **Results** to see AQI.", icon="🔥")
            landfill_name = st.session_state.get("selected_landfill_name", "selected site")
            st.warning(f"🔥 Fire detected near **{landfill_name}** — {_format_latlon(lat, lon)}.")
        else:
            st.info("No ongoing fire detected near the chosen landfill.")
        end_card()

    # Last run summary (visual)
    if st.session_state["last_run"] is not None:
        start_card()
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
            st.markdown("**Run time (UTC)**"); st.write(run_time)
        with c2:
            st.markdown("**Fire detected**")
            if fire:
                st.markdown('<span style="color:#FF7E00;font-weight:700;">YES</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#00E400;font-weight:700;">NO</span>', unsafe_allow_html=True)
        with c3:
            st.markdown("**Fused AQI**")
            cat, col = aqi_category_and_color(fused)
            if fused is None:
                st.write("—")
            else:
                st.write(f"{int(round(fused))} • {cat}")
            fig = plot_aqi_bar(fused); st.pyplot(fig, clear_figure=True)

        nav1, nav2 = st.columns([1,1])
        with nav1:
            if st.button("➡️ Go to Results"): _nav("results")
        with nav2:
            if st.button("🏠 Back to Home"): _nav("home")
        end_card()


# ----------------------- RESULTS -----------------------
elif page.lower() == "results":
    start_card()
    st.subheader("Results")

    landfill_name = st.session_state.get("selected_landfill_name")
    if landfill_name:
        st.markdown(f"**Landfill checked:** {landfill_name}")

    lr = st.session_state.get("last_run")

    if not lr:
        st.info("No results yet. Go to **Monitor** and click **Check now**.")
        col_a, col_b = st.columns([1, 3])
        with col_a:
            if st.button("Go to Monitor", key="go_to_monitor_results_empty"):
                new_qp = dict(st.query_params); new_qp["page"] = "Monitor"; st.query_params = new_qp; st.rerun()
        end_card()
    else:
        if not lr.get("fire_detected"):
            st.success("✅ No ongoing fire.")
            st.caption("If you believe there should be one, try increasing the radius on the Monitor page and re-run.")
            colb1, colb2 = st.columns([1,1])
            with colb1:
                if st.button("🔁 Run another check"): _nav("monitor")
            with colb2:
                if st.button("🏠 Home"): _nav("home")
            end_card()
        else:
            fused = lr.get("fused_aqi")
            cA, cB = st.columns([1,3])
            with cA:
                st.markdown("**Current fused AQI**")
                cat, col = aqi_category_and_color(fused)
                st.markdown(
                    f'<div style="font-size:28px;font-weight:700;color:{col};">{int(round(fused)) if fused is not None else "—"}</div>'
                    f'<div style="opacity:.85;">{cat if fused is not None else ""}</div>',
                    unsafe_allow_html=True
                )
            with cB:
                fig = plot_aqi_bar(fused); st.pyplot(fig, clear_figure=True)

            fdf = lr.get("forecast_df")
            if isinstance(fdf, pd.DataFrame) and not fdf.empty and {"datetime","pred_AQI"}.issubset(fdf.columns):
                next_hour = float(fdf["pred_AQI"].iloc[0])
                st.markdown("#### 72-hour forecast")
                plot_df = fdf.copy().set_index("datetime")[["pred_AQI"]]
                st.line_chart(plot_df)
                st.caption(f"Next-hour predicted AQI: **{int(round(next_hour))}**")
            else:
                st.warning("No forecast data available (model may have skipped training).")

            with st.expander("More details", expanded=False):
                satellite_current = lr.get("satellite_current", {}) or {}
                ground_current    = lr.get("ground_current", {}) or {}
                sat_breakdown     = lr.get("sat_breakdown", {}) or {}
                ground_breakdown  = lr.get("ground_breakdown", {}) or {}

                sat_expected = [
                    ("NO2 (molec/cm²)",          satellite_current.get("NO2"),  None),
                    ("O3 trop (molec/cm²)",      satellite_current.get("O3"),   None),
                    ("HCHO (molec/cm²)",         satellite_current.get("HCHO"), None),
                    ("UV Aerosol Index (UVAI)",  satellite_current.get("AER"),  None),
                ]
                sat_items = [(label, _fmt_value(val, unit)) for (label, val, unit) in sat_expected]
                render_cards("Satellite (TEMPO)", sat_items, cols=2)

                ground_expected = [
                    ("PM2.5 (µg/m³)", ground_current.get("PM2.5") or ground_current.get("PM25"), "µg/m³"),
                    ("PM10 (µg/m³)",  ground_current.get("PM10"), "µg/m³"),
                    ("NO₂ (ppb)",     ground_current.get("NO2"),  "ppb"),
                    ("O₃ (ppb)",      ground_current.get("O3"),   "ppb"),
                    ("CO (ppm)",      ground_current.get("CO"),   "ppm"),
                    ("SO₂ (ppb)",     ground_current.get("SO2"),  "ppb"),
                ]
                ground_items = [(label, _fmt_value(val, unit)) for (label, val, unit) in ground_expected]
                render_cards("Ground (OpenAQ / PurpleAir)", ground_items, cols=3)

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

                diag_left, diag_right = st.columns([1,2])
                with diag_left:
                    conf = lr.get("confidence_score")
                    st.metric("Forecast confidence", "—" if conf is None else f"{int(round(conf))} / 100")
                with diag_right:
                    reason = lr.get("training_fallback_reason") or "—"
                    st.caption(f"Training note: {reason}")

                st.caption("Values sourced from NASA TEMPO (via Harmony) and OpenAQ v3 (with PurpleAir only for PM2.5 when missing).")

            colb1, colb2 = st.columns([1,1])
            with colb1:
                if st.button("🔁 Run another check"): _nav("monitor")
            with colb2:
                if st.button("🏠 Home"): _nav("home")
            end_card()

# Footer (outside card so it blends with background)
st.markdown("---")
st.caption(
    "Demo app — uses public data sources (FIRMS / TEMPO via Harmony / OpenAQ / PurpleAir / OpenWeatherMap). "
    "If a service is unavailable or authentication fails, the demo may fall back to mock data."
)
