# streamlit_app.py
import os
from datetime import datetime
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Import your core pipelines
import landfill_pollution_detection_v2 as core

APP_TITLE = "Landfills Fire & Air Quality Monitor"
APP_PURPOSE = (
    "Watches for nearby landfills fires (NASA FIRMS). "
    "If a fire is detected, it fetches TEMPO satellite data and local ground measurements, "
    "computes satellite AQI, ground AQI, and a fused AQI, then trains a short-term model "
    "to forecast AQI for the next 72 hours."
)

# ---------- Helpers ----------
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
    st.info("Go to the **Monitor** page to run a live check.")

# ---------- PAGE: MONITOR ----------
elif page.lower() == "monitor":
    st.subheader("Monitor: run a check for fires and AQI")

    # Defaults to Calabasas landfill (demo)
    demo_lat = core.LANDFILL.get("lat", 34.1439)
    demo_lon = core.LANDFILL.get("lon", -118.6615)

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        lat = st.number_input("Latitude", value=float(demo_lat), format="%.6f")
    with col2:
        lon = st.number_input("Longitude", value=float(demo_lon), format="%.6f")
    with col3:
        radius_km = st.slider("Detection radius (km)", min_value=1, max_value=20, value=5, step=1)

    st.checkbox("Enable in-app notifications", value=st.session_state["notifications"], key="notifications")

    st.markdown("---")
    run_col, info_col = st.columns([1,2])
    with run_col:
        run_btn = st.button("🚀 Check now", type="primary")
    with info_col:
        st.caption(
            "On click, the app runs FIRMS check. If a fire is detected, it downloads data to get current AQI "
            "and forecast AQI for next 72 hours (Pipelines 2–6)."
        )

    if run_btn:
        with st.spinner("Running pipelines…"):
            out = core.run_workflow_if_fire(
                lat=float(lat),
                lon=float(lon),
                radius_km=float(radius_km),
                run_parallel=True,
                forecast_horizon_h=72
            )
            st.session_state["last_run"] = out

        if out.get("fire_detected"):
            if st.session_state["notifications"]:
                st.toast("🔥 Fire detected near the chosen landfill. Click **Results** to see AQI.", icon="🔥")
            st.success(f"Fire detected near {_format_latlon(lat, lon)}.")
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

    lr = st.session_state.get("last_run")
    if not lr:
        st.info("No results yet. Go to **Monitor** and click **Check now**.")
    else:
        if not lr.get("fire_detected"):
            st.success("✅ No ongoing fire.")
            st.caption("If you believe there should be one, try increasing the radius on the Monitor page and re-run.")

            # Clear navigation (avoid dead links)
            colb1, colb2 = st.columns([1,1])
            with colb1:
                if st.button("🔁 Run another check"):
                    _nav("monitor")
            with colb2:
                if st.button("🏠 Home"):
                    _nav("home")
        else:
            # Current fused AQI + color bar
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

            # Predicted AQI + chart
            fdf = lr.get("forecast_df")
            if isinstance(fdf, pd.DataFrame) and not fdf.empty and {"datetime","pred_AQI"}.issubset(fdf.columns):
                next_hour = float(fdf["pred_AQI"].iloc[0])
                st.markdown("#### 72-hour forecast")
                plot_df = fdf.copy().set_index("datetime")[["pred_AQI"]]
                st.line_chart(plot_df)
                st.caption(f"Next-hour predicted AQI: **{int(round(next_hour))}**")
            else:
                st.warning("No forecast data available (model may have skipped training).")

            # Optional details
            with st.expander("More details"):
                st.write({
                    "satellite_current": lr.get("satellite_current"),
                    "ground_current": lr.get("ground_current"),
                    "sat_AQI": lr.get("sat_aqi_value"),
                    "ground_AQI": lr.get("ground_aqi_value"),
                    "valid_rmse": lr.get("valid_rmse"),
                    "training_fallback_reason": lr.get("training_fallback_reason"),
                })

            # Clean navigation (no empty hyperlinks)
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
    "If a service is unavailable or unauthorized, the underlying pipelines may fall back to mock data."
)
