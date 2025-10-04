# streamlit_app.py
import os
from datetime import datetime
import pandas as pd
import streamlit as st

# Import your core pipelines (same folder)
# The file should be: landfill_pollution_detection_v2.py
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
    qp = st.experimental_get_query_params()
    return (qp.get("page", [default])[0] or default).lower()

def _nav(page_name: str):
    st.experimental_set_query_params(page=page_name.lower())
    st.experimental_rerun()

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

# ---------- Layout ----------
st.set_page_config(page_title=APP_TITLE, page_icon="🔥", layout="wide")

# Sidebar navigation (simple 3-page app)
page = st.sidebar.radio("Pages", ["Home", "Monitor", "Results"], index=["home","monitor","results"].index(_get_query_page("home")))

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
            "On click, the app runs FIRMS check. If a fire is detected, it downloads data to get current AQI and forecast AQI for next 72 hours"           "Pipelines 2–6. "
            "This may take a short while depending on data access and network."
        )

    if run_btn:
        with st.spinner("Running pipelines…"):
            # Orchestrator: runs 1 → (2&3) → 4 → 5 → 6
            # Set run_parallel=True to fetch TEMPO & Ground concurrently
            out = core.run_workflow_if_fire(
                lat=float(lat),
                lon=float(lon),
                radius_km=float(radius_km),
                run_parallel=True,
                forecast_horizon_h=72
            )
            st.session_state["last_run"] = out

        # Notify user & offer quick navigation
        if out.get("fire_detected"):
            if st.session_state["notifications"]:
                st.toast("🔥 Fire detected near the chosen landfill. Click **View details** to see AQI.", icon="🔥")
            st.success(f"Fire detected near {_format_latlon(lat, lon)}.")
            st.link_button("➡️ View details (Results page)", on_click=_nav, args=("results",))
        else:
            st.info("No ongoing fire detected near the chosen landfill.")

    # Show a tiny preview of the last run (if any)
    if st.session_state["last_run"] is not None:
        lr = st.session_state["last_run"]
        st.markdown("---")
        st.markdown("#### Last run summary")
        st.write(
            {
                "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"),
                "fire_detected": bool(lr.get("fire_detected")),
                "fused_AQI": lr.get("fused_aqi"),
            }
        )

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
        else:
            # Show fused AQI (current)
            fused = lr.get("fused_aqi")
            colA, colB = st.columns([1,1])
            with colA:
                st.metric("Current fused AQI", value=(f"{int(fused)}" if fused is not None else "—"))

            # Show predicted AQI (next hour) + chart
            fdf = lr.get("forecast_df")
            if isinstance(fdf, pd.DataFrame) and not fdf.empty and {"datetime","pred_AQI"}.issubset(fdf.columns):
                next_hour = float(fdf["pred_AQI"].iloc[0])
                with colB:
                    st.metric("Predicted AQI (next hour)", value=f"{int(round(next_hour))}")
                st.markdown("#### 72-hour forecast")
                plot_df = fdf.copy()
                plot_df = plot_df.set_index("datetime")[["pred_AQI"]]
                st.line_chart(plot_df)
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
                })

# Footer
st.markdown("---")
st.caption(
    "Demo app — uses public data sources (FIRMS / TEMPO via Harmony / OpenAQ / PurpleAir / OpenWeatherMap). "
    "If a service is unavailable or unauthorized, the underlying pipelines may fall back to mock data."
)
