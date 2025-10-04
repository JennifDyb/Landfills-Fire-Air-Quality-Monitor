Landfills Fire & Air Quality Monitor (Streamlit App)
This Streamlit app watches for nearby fires (NASA FIRMS) for a landfill location, and—if a fire is detected—fetches satellite (TEMPO via Harmony) and ground (OpenAQ v3 / PurpleAir) measurements to compute Satellite AQI, Ground AQI, and a Fused AQI. 
It then pulls current weather (OpenWeatherMap) and trains a small LightGBM model to forecast the AQI for the next 72 hours.
The app is built on your pipelines in landfill_pollution_detection_v2.py and provides a minimal 3-page UI in streamlit_app.py.

What you get
Home — name & purpose of the app
Monitor — enter coordinates (pre-filled to Calabasas Landfill for demo), set detection radius, toggle in-app notifications, and Run Check
Results — if a fire is detected: show current fused AQI and next-72h prediction (chart). Otherwise: “no ongoing fire.”
Notifications are in-app toasts—Streamlit free tier doesn’t support OS push. The toast includes a “View details” button that jumps to Results.

Repository layout
.
├─ streamlit_app.py                 # Streamlit UI (entry point)
├─ landfill_pollution_detection_v2.py
├─ requirements.txt
├─ README.md                        # this file
└─ .env.example                     # optional (sample secrets for local dev)

Data sources / APIs
FIRMS — Fire Information for Resource Management System (NASA)
TEMPO — Tropospheric Emissions: Monitoring of Pollution (via Harmony APIs)
OpenAQ v3 — ground sensors (requires API key)
PurpleAir — PM2.5 fallback/fill (API key recommended)
OpenWeatherMap — current weather (temp, humidity, wind, gust, deg, pressure, rain)
If an upstream service is unavailable or unauthorized, the pipelines fallback to mock data so the app still renders.

Quick start (local)
1. Python & install
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

2. Create a .env (not committed) in the repo root:
OPENAQ_API_KEY=your-openaq-key
PURPLEAIR_KEY=your-purpleair-key
OWM_API_KEY=your-openweather-key
FIRMS_API_KEY=your-firms-key
EARTHDATA_USER=your-earthdata-login-username
EARTHDATA_PASS=your-earthdata-login-password

3. Run
streamlit run streamlit_app.py

Using the app

Go to Monitor:
Latitude/Longitude default to Calabasas Landfill (demo).
Adjust Radius (km) if needed.
Toggle Enable in-app notifications if you want a toast after the run.
Click Check now.

If a fire is detected (FIRMS inside the chosen radius):
Pipelines 2–6 run automatically.
You’ll see a toast; click View details or go to Results.

Results shows:
Current fused AQI
Next-hour predicted AQI
72-hour forecast chart

Expand More details for satellite/ground raw values and model RMSE (if available).

Notes & tips

TEMPO / Harmony auth
Requires Earthdata Login (EDL). The pipelines support either a ~/.netrc or passing credentials via env vars/secrets (what Streamlit Cloud uses). 
If you get auth errors, accept provider terms once in Earthdata Search using the same EDL account.

OpenAQ v3
Needs a valid API key. Timeouts/429s can happen; code retries then falls back to mock.

LightGBM dtype errors
We coerce all feature columns to numeric; if you load your own history, ensure columns like wind_gust aren’t strings.

Free tier considerations
Streamlit Cloud’s free tier may have limited CPU/time for heavy satellite jobs. The app handles this gracefully (mocks), so users still get a meaningful result.

Development commands

Run one full check locally (bypassing FIRMS):
python landfill_pollution_detection_v2.py --once --force-fire

Run once, FIRMS-gated:
python landfill_pollution_detection_v2.py --once

Scheduler (hourly FIRMS checks, stop after 1 cycle):
python landfill_pollution_detection_v2.py --runs 1

Requirements
All dependencies are listed in requirements.txt (includes Streamlit, requests, pandas, numpy, matplotlib, lightgbm, and the Harmony/xarray/NetCDF stack).

Privacy & security
Do not commit secrets. Use Streamlit Secrets or a local .env.
Rotate keys if you ever suspect they were exposed.

License
