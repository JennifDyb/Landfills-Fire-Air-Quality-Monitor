# -*- coding: utf-8 -*-

"""
# Calabasas Landfill — End-to-End AQ Monitoring Prototype (5 Pipelines)
Purpose: Detect fires near the Calabasas landfill and produce satellite AQI, ground AQI, fused AQI, and a 72-hour hourly AQI forecast using LightGBM.  
Structure:
- Pipeline 1: FIRMS fire detection (real + fallback)  
- Pipeline 2: TEMPO via HarmonyPy — multi-pollutant satellite subset (real + fallback)  
- Pipeline 3: Ground ingestion — OpenAQ & PurpleAir (real + fallback)  
- Pipeline 4: Compute sat-AQI, ground-AQI, fused-AQI + visualization & provenance  
- Pipeline 5: Meteorology
- Pipeline 6: LightGBM forecast using last 90 days (real if available, fallback synthetic)

"""

# Imports and basic config
import os
from dotenv import load_dotenv
import io
import json
import time
import random
import traceback
import math
import stat
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta,timezone
from shapely.geometry import Point
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
import streamlit as st

# Harmony
from harmony import BBox, Client, Collection, Request

# For xarray reading of NetCDF (TEMPO results)
import xarray as xr

# For LightGBM
import lightgbm as lgb

# User-configurable: landfill coordinates (Calabasas)
LANDFILL = {"name": "Calabasas Landfill", "lat": 34.1439, "lon": -118.6615}

# Environment variables
#load_dotenv()
#FIRMS_API_KEY = os.getenv("FIRMS_API_KEY")          #NASA FIRMS API
#EARTHDATA_USER = os.getenv("EARTHDATA_USER")        # Earthdata username for Harmony/Earthaccess
#EARTHDATA_PASS = os.getenv("EARTHDATA_PASS")        # Earthdata password
#PURPLEAIR_KEY = os.getenv("PURPLEAIR_KEY")          # PurpleAir  key
#OWM_API_KEY = os.getenv("OWM_API_KEY")              # OpenWeatherMap optional key
#OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")        # OpenAQ  key


FIRMS_API_KEY = st.secrets["FIRMS_API_KEY"]
EARTHDATA_USER = st.secrets["EARTHDATA_USER"]
EARTHDATA_PASS = st.secrets["EARTHDATA_PASS"]
PURPLEAIR_KEY = st.secrets["PURPLEAIR_KEY"]
OWM_API_KEY = st.secrets["OWM_API_KEY"]
OPENAQ_API_KEY = st.secrets["OPENAQ_API_KEY"]

# Defaults for the demo
SAT_HIST_DAYS = 90      # history days for satellite percentile mapping
GROUND_SEARCH_RADIUS_M = 20000  # search radius for ground sensors (m)
FUSED_W_GROUND = 0.7
FUSED_W_SAT = 0.3

print("Landfill:", LANDFILL)
print("FIRMS key set?", bool(FIRMS_API_KEY))
print("Earthdata credentials set?", bool(EARTHDATA_USER and EARTHDATA_PASS))
print("PurpleAir key set?", bool(PURPLEAIR_KEY))
print("OpenWeatherMap key set?", bool(OWM_API_KEY))


# Pipeline 1: FIRMS fire detection (VIIRS/MODIS) with fallback to mock data


FIRMS_AREA_CSV_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

def _bbox_str(lat_min, lat_max, lon_min, lon_max):
    # FIRMS expects west,south,east,north
    west, south, east, north = lon_min, lat_min, lon_max, lat_max
    return f"{west},{south},{east},{north}"

def query_firms_bbox(lat_min, lat_max, lon_min, lon_max,
                     api_key=None, source="VIIRS_SNPP_NRT", day_range=1, date=None):
    """
    Query FIRMS for active fires inside a bounding box.
    Returns a DataFrame with columns: latitude, longitude, acq_date, acq_time, confidence, instrument, satellite.
    Uses CSV endpoint per FIRMS spec; returns empty DataFrame on any failure.
    """
    api_key = api_key or FIRMS_API_KEY
    if not api_key:
        return pd.DataFrame()

    try:
        bbox = _bbox_str(lat_min, lat_max, lon_min, lon_max)
        url = f"{FIRMS_AREA_CSV_BASE}/{api_key}/{source}/{bbox}/{int(day_range)}"
        if date:
            url = f"{url}/{date}"  # YYYY-MM-DD

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        txt = resp.text.strip()

        # Some responses may be messages like "No data found" or HTML error pages
        if not txt or txt.lower().startswith("no data") or txt.startswith("<"):
            return pd.DataFrame()

        df = pd.read_csv(io.StringIO(txt))

        # Normalize/keep only columns we need if present
        colmap = {
            "latitude": "latitude",
            "longitude": "longitude",
            "acq_date": "acq_date",
            "acq_time": "acq_time",
            "confidence": "confidence",
            "instrument": "instrument",
            "satellite": "satellite"
        }
        keep = [c for c in colmap if c in df.columns]
        if not keep:
            return pd.DataFrame()

        out = df[keep].copy()

        # Ensure numeric coords
        for c in ("latitude", "longitude"):
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        return out.dropna(subset=["latitude", "longitude"])
    except Exception as e:
        print("FIRMS query error:", e)
        return pd.DataFrame()

def detect_fire_near_landfill(lat, lon, radius_km=5):
    # Define bbox around landfill to query FIRMS efficiently
    deg_buffer = 0.05  # ~5–6 km
    lat_min, lat_max = lat - deg_buffer, lat + deg_buffer
    lon_min, lon_max = lon - deg_buffer, lon + deg_buffer

    df = query_firms_bbox(lat_min, lat_max, lon_min, lon_max,
                          api_key=FIRMS_API_KEY, source="VIIRS_SNPP_NRT", day_range=1)

    if df.empty:
        # Fallback mock
        print("No FIRMS data found (or key invalid). Using mock fire event.")
        mock = pd.DataFrame([{
            "longitude": lon + 0.01,
            "latitude": lat - 0.01,
            "acq_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "acq_time": datetime.utcnow().strftime("%H%M"),
            "confidence": 95,
            "instrument": "VIIRS",
            "satellite": "MOCK"
        }])
        df = mock

    # Haversine distance
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*(math.sin(dlambda/2.0)**2)
        return 2*R*math.asin(min(1, math.sqrt(a)))

    df["dist_km"] = df.apply(lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]), axis=1)
    near = df[df["dist_km"] <= radius_km].copy()

    if near.empty:
        print(f"No FIRMS detections within {radius_km} km; returning nearest detection (if any).")
        if not df.empty:
            return df.iloc[[0]]
        return df

    return near

# Run detection
fires_near = detect_fire_near_landfill(LANDFILL["lat"], LANDFILL["lon"], radius_km=5)
print("FIRMS (or mock) detections near landfill:")
print(fires_near.head().to_string(index=False))



# Pipeline 2: TEMPO via Harmony (robust + mock fallback)

# === Config / constants ===

TEMPO_COLLECTIONS = {
    "NO2":  "TEMPO_NO2_L2",
    "O3":   "TEMPO_O3PROF_L2",
    "HCHO": "TEMPO_HCHO_L2",
    "AER":  "TEMPO_O3TOT_L2",
}

# Prefer env overrides if you want to pin specific versions
TEMPO_CMR_IDS = {
    "TEMPO_NO2_L2":    os.environ.get("TEMPO_NO2_CMR")    or "C3685896872-LARC_CLOUD",   # V04
    "TEMPO_O3PROF_L2": os.environ.get("TEMPO_O3PROF_CMR") or "C3685896287-LARC_CLOUD",   # V04
    "TEMPO_HCHO_L2":   os.environ.get("TEMPO_HCHO_CMR")   or "C3685668884-LARC_CLOUD",   # V02
    "TEMPO_O3TOT_L2":  os.environ.get("TEMPO_O3TOT_CMR")  or "C3685912131-LARC_CLOUD",   # V04
}

# Mock fallback (roughly realistic magnitudes)
MOCK_TEMPO = {
    "NO2":  5.0e14, # molec/cm^2
    "O3":   6.0e17, # molec/cm^2 (trop)
    "HCHO": 1.5e15, # molec/cm^2
    "AER":  0.3, # UVAI (unitless)
}

# === Helpers ===
def _ensure_netrc():
    """Create ~/.netrc for EDL if EARTHDATA_USER/PASS set and no valid netrc exists."""
    if not (EARTHDATA_USER and EARTHDATA_PASS):
        return
    path = os.path.expanduser("~/.netrc")
    needs_write = True
    if os.path.exists(path):
        try:
            st = os.stat(path)
            # If file exists with proper perms and has EDL entry, keep it
            if stat.S_IMODE(st.st_mode) & 0o077 == 0:
                txt = open(path, "r", encoding="utf-8").read()
                if "urs.earthdata.nasa.gov" in txt:
                    needs_write = False
        except Exception:
            pass
    if needs_write:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(
                "machine urs.earthdata.nasa.gov login {u} password {p}\n".format(
                    u=EARTHDATA_USER, p=EARTHDATA_PASS
                )
            )
        os.chmod(path, 0o600)

def _open_dataset_any(path):
    """Try opening a file with multiple xarray backends; return Dataset or raise last error."""
    last = None
    for engine in (None, "netcdf4", "h5netcdf"):
        try:
            return xr.open_dataset(path, engine=engine)
        except Exception as e:
            last = e
            continue
    raise last or RuntimeError(f"Failed to open dataset: {path}")

def _choose_var_for_pol(ds: xr.Dataset, pol: str) -> str | None:
    names = list(ds.data_vars.keys())
    lo = [n.lower() for n in names]

    def pick(tokens_any, must_all=()):
        for n, nl in zip(names, lo):
            if any(t in nl for t in tokens_any) and all(m in nl for m in must_all):
                return n
        return None

    if pol == "AER":
        v = pick(["uvai", "uv_ai", "uv-aerosol", "uv_aerosol", "aerosol_index"])
        if v: return v
    if pol == "O3":
        v = pick(["o3", "ozone"], ["trop"])
        if v: return v
        v = pick(["o3", "ozone"], ["partial", "0_2", "0-2", "0to2"])
        if v: return v
        v = pick(["o3", "ozone", "column"])
        if v: return v
    if pol == "NO2":
        v = pick(["no2"], ["trop"])
        if v: return v
        v = pick(["no2", "column"])
        if v: return v
    if pol == "HCHO":
        v = pick(["hcho", "formaldehyde", "column"])
        if v: return v
        v = pick(["hcho", "formaldehyde"])
        if v: return v

    # Generic: first lat/lon-like var
    for n in names:
        dims = set(ds[n].dims)
        if {"lat", "lon"}.issubset(dims) or {"latitude", "longitude"}.issubset(dims):
            return n
    return names[0] if names else None

def _safe_sel(ds: xr.Dataset, var: str, lat: float, lon: float) -> float:
    lat_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
    lon_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    if lat_name is None or lon_name is None:
        rename = {}
        for cand in ("y", "Y"):
            if cand in ds.coords: rename[cand] = "lat"
        for cand in ("x", "X"):
            if cand in ds.coords: rename[cand] = "lon"
        if rename:
            ds = ds.rename(rename)
            lat_name, lon_name = "lat", "lon"

    if lon_name and lon < 0:
        try:
            lvals = ds[lon_name].values
            if np.nanmin(lvals) >= 0 and np.nanmax(lvals) <= 360:
                lon = (lon + 360.0) % 360.0
        except Exception:
            pass

    try:
        if np.any(np.diff(ds[lat_name].values) < 0): ds = ds.sortby(lat_name)
        if np.any(np.diff(ds[lon_name].values) < 0): ds = ds.sortby(lon_name)
    except Exception:
        pass

    # nearest selection
    try:
        sel = ds[var].sel({lat_name: lat, lon_name: lon}, method="nearest").values
    except Exception:
        for la, lo in (("latitude","longitude"), ("lat","lon")):
            if la in ds.coords and lo in ds.coords:
                sel = ds[var].sel({la: lat, lo: lon}, method="nearest").values
                break
        else:
            raise
    arr = np.asarray(sel).squeeze()
    if arr.size == 0 or (np.issubdtype(arr.dtype, np.number) and np.isnan(arr).all()):
        raise ValueError("Nearest value is empty/NaN")
    return float(arr.ravel()[0])

def _concept_id(short_name: str) -> str:
    cid = TEMPO_CMR_IDS.get(short_name)
    if not cid:
        raise RuntimeError(f"No concept ID configured for {short_name}")
    return cid

def request_tempo_subset(lat: float, lon: float, days_back: int = 1,
                         collections: dict = TEMPO_COLLECTIONS) -> dict:
    """
    Try to fetch {NO2, O3, HCHO, AER} from TEMPO via Harmony.
    If anything fails (auth/network/parsing), returns realistic MOCK_TEMPO for missing keys.
    """
    # Prepare auth (netrc) if creds provided
    _ensure_netrc()

    # bbox ~5–6 km
    bbox = BBox(lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05)
    t_end = datetime.utcnow()
    t_start = t_end - timedelta(days=int(days_back))
    temporal = {"start": t_start, "stop": t_end}  # Harmony expects datetimes

    # If creds are missing, skip straight to mock
    if not (EARTHDATA_USER and EARTHDATA_PASS):
        print("Earthdata creds not set -> skipping Harmony calls, using mock satellite data.")
        return dict(MOCK_TEMPO)

    results = {}
    try:
        os.makedirs("./tempo_data", exist_ok=True)
        client = Client()  # reads ~/.netrc

        for pol, short_name in collections.items():
            try:
                concept_id = _concept_id(short_name)
                print(f"Submitting Harmony job for {pol} ({short_name} / {concept_id}) ...")

                # Build request with explicit Collection(id=...)
                req = Request(
                    collection=Collection(id=concept_id),
                    spatial=bbox,
                    temporal=temporal,
                    format="application/x-netcdf4",
                )

                job_id = client.submit(req)  # returns job id / job object depending on version
                files = client.download(job_id, directory="./tempo_data", show_progress=False)
                if not files:
                    print(f"  No files returned for {pol}, using mock.")
                    results[pol] = MOCK_TEMPO.get(pol)
                    continue

                # Try the returned files until one opens and yields a value
                value = None
                for path in files:
                    try:
                        with _open_dataset_any(path) as ds:
                            if "time" in ds.dims:
                                ds = ds.isel(time=-1, drop=True)
                            varname = _choose_var_for_pol(ds, pol)
                            if not varname:
                                continue
                            value = _safe_sel(ds, varname, lat, lon)
                            break
                    except Exception as e_file:
                        # File may be an error artifact or non-netcdf; try next
                        # print(f"  {pol}: file {os.path.basename(path)} error: {e_file}")
                        continue

                if value is None or (isinstance(value, float) and np.isnan(value)):
                    print(f"  {pol}: no usable value in outputs, using mock.")
                    results[pol] = MOCK_TEMPO.get(pol)
                else:
                    results[pol] = float(value)
                    print(f"  Retrieved {pol} -> {results[pol]}")
            except Exception as e_coll:
                print(f"  Warning: error retrieving {pol}: {e_coll} — using mock.")
                results[pol] = MOCK_TEMPO.get(pol)
    except Exception as e:
        print("Harmony client error:", e)
        print("Using mock satellite values for all pollutants.")
        # Fill all with mocks
        return {k: MOCK_TEMPO.get(k) for k in collections.keys()}

    # Ensure all keys present
    for pol in collections.keys():
        if pol not in results or results[pol] is None:
            results[pol] = MOCK_TEMPO.get(pol)
    return results

# ---- Run satellite ingestion (example) ----
satellite_current = request_tempo_subset(LANDFILL["lat"], LANDFILL["lon"], days_back=1)
print("Satellite values (TEMPO or mock):")
print(satellite_current)


# Pipeline 3 - Ground pipeline: OpenAQ v3 (for O3, NO2, PM2.5, PM10, CO, SO2) + PurpleAir (PM2.5) with mock fallback

# ---- Config ----
GROUND_SEARCH_RADIUS_M = 20_000          # OpenAQ radius in meters
OPENAQ_LOOKBACK_HOURS  = 6               # how far back to accept measurements
PURPLEAIR_RADIUS_KM = 10                 # bbox "radius" in km for PurpleAir
PURPLEAIR_MAX_AGE_MIN = 60               # ignore sensors older than this (minutes)

# Mock fallback if nothing is found
MOCK_GROUND = {"PM25": 40.0, "PM10": 60.0, "NO2": 30.0, "O3": 55.0, "CO": 0.9, "SO2": 2.0}

# Normalize to our canonical pollutant symbols
CANONICALS = {
    "pm25": "PM25", "pm2_5": "PM25", "pm2p5": "PM25",
    "pm10": "PM10",
    "no2": "NO2",
    "o3":  "O3",
    "co":  "CO",
    "so2": "SO2",
}
def _canon(name: str) -> str:
    return CANONICALS.get(name.lower(), name.upper())

# ---- Target units (what we output) ----
TARGET_UNITS = {
    "PM25": "ug/m3",   # µg/m³
    "PM10": "ug/m3",
    "NO2":  "ppb",
    "O3":   "ppb",
    "CO":   "ppm",
    "SO2":  "ppb",
}

# ---- Simple unit conversion helpers ----
# Assumes 25°C and 1 atm (24.45 L/mol) for ppb<->µg/m³ gas conversions.
MW = {"NO2": 46.0055, "O3": 48.0, "CO": 28.01, "SO2": 64.066}
MOLAR_VOLUME = 24.45  # L/mol at 25°C & 1 atm

def _norm_unit(u: str | None) -> str:
    if not u:
        return ""
    s = u.strip().lower()
    s = s.replace("µ", "u")  # normalize micro symbol
    s = s.replace("μ", "u")
    s = s.replace("μg/m3", "ug/m3")
    s = s.replace("µg/m3", "ug/m3")
    s = s.replace("ug/m^3", "ug/m3")
    s = s.replace("ug/m³", "ug/m3")
    s = s.replace("mg/m³", "mg/m3").replace("ng/m³", "ng/m3")
    return s

def _to_target(pollutant: str, value: float, unit: str | None) -> float | None:
    """
    Convert a single measurement (value, unit) to TARGET_UNITS[pollutant].
    Returns converted float or None if conversion is impossible.
    """
    try:
        v = float(value)
    except Exception:
        return None

    pol = pollutant.upper()
    if pol not in TARGET_UNITS:
        return None
    target = TARGET_UNITS[pol]
    u = _norm_unit(unit)

    # Particulates (mass concentration)
    if pol in ("PM25", "PM10"):
        if u in ("ug/m3", ""):  # assume ug/m3 if missing
            return v
        if u == "mg/m3":
            return v * 1000.0
        if u == "ng/m3":
            return v / 1000.0
        # If accidentally reported in (p)pp, we can't safely convert without density; skip
        if u in ("ppb", "ppm"):
            return None
        return None

    # Gases
    mw = MW.get(pol if pol != "O3" else "O3", None)
    if mw is None:
        return None

    # Desired is ppb for NO2/O3/SO2; ppm for CO
    if pol == "CO":
        # target = ppm
        if u == "ppm" or u == "":
            return v
        if u == "ppb":
            return v / 1000.0
        if u == "ug/m3":
            # ppm = (ug/m3) * 24.45 / MW / 1000
            return (v * MOLAR_VOLUME / mw) / 1000.0
        if u == "mg/m3":
            return ((v * 1000.0) * MOLAR_VOLUME / mw) / 1000.0
        if u == "ng/m3":
            return ((v / 1000.0) * MOLAR_VOLUME / mw) / 1000.0
        return None
    else:
        # target = ppb (NO2, O3, SO2)
        if u == "ppb" or u == "":
            return v
        if u == "ppm":
            return v * 1000.0
        if u == "ug/m3":
            # ppb = (ug/m3) * 24.45 / MW
            return (v * MOLAR_VOLUME) / mw
        if u == "mg/m3":
            return ((v * 1000.0) * MOLAR_VOLUME) / mw
        if u == "ng/m3":
            return ((v / 1000.0) * MOLAR_VOLUME) / mw
        return None

# -------------------------------
# OpenAQ v3
# -------------------------------
# --- Tunables (or set via env) ---
OPENAQ_RETRIES_PER_CALL      = int(os.environ.get("OPENAQ_RETRIES_PER_CALL", 2))   # per HTTP GET
OPENAQ_TOTAL_DEADLINE_SEC    = float(os.environ.get("OPENAQ_TOTAL_DEADLINE_SEC", 30))  # whole function budget
OPENAQ_PER_PAGE_LIMIT        = int(os.environ.get("OPENAQ_PER_PAGE_LIMIT", 200))
OPENAQ_MAX_PAGES_PER_SENSOR  = int(os.environ.get("OPENAQ_MAX_PAGES_PER_SENSOR", 6))
OPENAQ_MAX_SENSORS           = int(os.environ.get("OPENAQ_MAX_SENSORS", 40))
OPENAQ_MAX_SENSOR_ERRORS     = int(os.environ.get("OPENAQ_MAX_SENSOR_ERRORS", 5))
OPENAQ_INTER_PAGE_SLEEP_S    = float(os.environ.get("OPENAQ_INTER_PAGE_SLEEP_S", 0.15))

def _to_int(x, default=None):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default

def _request_json(session, url, params, headers, timeout=15,
                  retries=OPENAQ_RETRIES_PER_CALL, backoff=1.6,
                  retry_statuses=(408, 429, 500, 502, 503, 504)):
    """GET with bounded retry/backoff; raises on final failure."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = session.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code in retry_statuses:
                # honor Retry-After if present
                ra = r.headers.get("Retry-After")
                if ra:
                    try:
                        sleep_s = float(ra)
                    except:
                        sleep_s = (backoff ** attempt) + random.uniform(0, 0.25)
                else:
                    sleep_s = (backoff ** attempt) + random.uniform(0, 0.25)
                time.sleep(sleep_s)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_exc = e
            # network error: backoff if we still have attempts
            if attempt < retries:
                time.sleep((backoff ** attempt) + random.uniform(0, 0.25))
            else:
                break
    # out of attempts
    if isinstance(last_exc, requests.HTTPError) and last_exc.response is not None:
        raise last_exc
    raise requests.RequestException(last_exc)

def get_openaq_latest_v3(lat: float, lon: float,
                         radius_m: int = GROUND_SEARCH_RADIUS_M,
                         lookback_hours: int = OPENAQ_LOOKBACK_HOURS,
                         parameters=("pm25","pm10","no2","o3","co","so2"),
                         max_sensors: int = OPENAQ_MAX_SENSORS,
                         per_page_limit: int = OPENAQ_PER_PAGE_LIMIT,
                         max_pages_per_sensor: int = OPENAQ_MAX_PAGES_PER_SENSOR,
                         inter_page_sleep_s: float = OPENAQ_INTER_PAGE_SLEEP_S) -> dict:
    """
    v3 flow (bounded and resilient):
      1) /v3/locations?coordinates=...&radius=...
      2) /v3/locations/{id}/sensors
      3) /v3/sensors/{sid}/measurements (paged)
    Returns pollutant averages in TARGET_UNITS or {} on failure (caller falls back to mock).
    """
    start_time = time.time()
    def over_deadline():
        return (time.time() - start_time) > OPENAQ_TOTAL_DEADLINE_SEC

    try:
        headers = {"X-API-Key": OPENAQ_API_KEY} if OPENAQ_API_KEY else {}
        if not headers:
            print("Warning: OPENAQ_API_KEY not set. Requests will fail with 401.")

        base = "https://api.openaq.org/v3"
        t_end = datetime.now(timezone.utc)
        t_start = t_end - timedelta(hours=int(lookback_hours))

        with requests.Session() as sess:
            # 1) Nearby locations (cap radius at 25 km per docs)
            if over_deadline():
                return {}
            loc_params = {
                "coordinates": f"{lat},{lon}",
                "radius": int(min(radius_m, 25000)),
                "limit": 1000
            }
            try:
                js = _request_json(sess, f"{base}/locations", loc_params, headers)
            except requests.HTTPError as e:
                if getattr(e.response, "status_code", None) == 401:
                    print("OpenAQ v3 query error: 401 Unauthorized — check OPENAQ_API_KEY.")
                else:
                    print(f"OpenAQ v3 locations error: {e}")
                return {}
            except requests.RequestException as e:
                print(f"OpenAQ v3 locations network error: {e}")
                return {}

            loc_results = js.get("results", [])
            if not loc_results:
                return {}

            wanted = set(p.lower() for p in parameters)

            # 2) Sensors per location
            sensor_infos = []  # (sensor_id, parameter_name_lower)
            sensor_errors = 0
            for loc in loc_results:
                if over_deadline():
                    return {}
                loc_id = loc.get("id") or loc.get("locationsId")
                if not loc_id:
                    continue
                try:
                    sens_js = _request_json(sess, f"{base}/locations/{loc_id}/sensors",
                                            {"limit": 1000}, headers)
                except requests.HTTPError as e:
                    code = getattr(e.response, "status_code", None)
                    if code == 404:
                        continue
                    sensor_errors += 1
                    if sensor_errors >= OPENAQ_MAX_SENSOR_ERRORS or over_deadline():
                        return {}
                    continue
                except requests.RequestException:
                    sensor_errors += 1
                    if sensor_errors >= OPENAQ_MAX_SENSOR_ERRORS or over_deadline():
                        return {}
                    continue

                for s in sens_js.get("results", []):
                    pblock = s.get("parameter") or {}
                    pname = (pblock.get("name") or "").lower()
                    sid = s.get("id") or s.get("sensorsId")
                    if sid and pname in wanted:
                        sensor_infos.append((sid, pname))
                        if len(sensor_infos) >= max_sensors:
                            break
                if len(sensor_infos) >= max_sensors:
                    break

            if not sensor_infos:
                return {}

            # 3) Measurements per sensor, paged
            sums, counts = {}, {}
            for sid, pname in sensor_infos:
                if over_deadline():
                    return {}
                m_params = {
                    "date_from": t_start.isoformat(timespec="seconds").replace("+00:00","Z"),
                    "date_to":   t_end.isoformat(timespec="seconds").replace("+00:00","Z"),
                    "limit": int(per_page_limit),
                    "page": 1
                }
                pages_done = 0
                while True:
                    if over_deadline():
                        return {}
                    try:
                        js = _request_json(sess, f"{base}/sensors/{sid}/measurements",
                                           m_params, headers, timeout=15)
                    except requests.HTTPError as e:
                        code = getattr(e.response, "status_code", None)
                        if code == 404:
                            break  # sensor offline/retired
                        # log and give up on this sensor
                        # print(f"OpenAQ sensor {sid} page {m_params['page']} error: {e}")
                        break
                    except requests.RequestException:
                        # network failure: give up on this sensor page
                        break

                    results = js.get("results", []) or []
                    if not results:
                        break

                    for m in results:
                        v = m.get("value"); u = m.get("unit")
                        pol = _canon(pname)
                        conv = _to_target(pol, v, u)
                        if conv is None:
                            continue
                        sums[pol] = sums.get(pol, 0.0) + float(conv)
                        counts[pol] = counts.get(pol, 0) + 1

                    meta = js.get("meta") or {}
                    page_i  = _to_int(meta.get("page"),  m_params["page"])
                    limit_i = _to_int(meta.get("limit"), m_params["limit"])
                    found_i = _to_int(meta.get("found"), None)

                    pages_done += 1
                    if pages_done >= max_pages_per_sensor:
                        break
                    if found_i is not None:
                        if (page_i * limit_i) >= found_i:
                            break
                    else:
                        if len(results) < limit_i:
                            break

                    m_params["page"] = page_i + 1
                    if inter_page_sleep_s:
                        time.sleep(inter_page_sleep_s)

        return {k: sums[k] / counts[k] for k in sums} if sums else {}
    except Exception as e:
        print("OpenAQ v3 query error:", e)
        return {}


# -------------------------------
# PurpleAir (PM2.5 only, µg/m³)
# -------------------------------
def _bbox_from_radius(lat: float, lon: float, radius_km: float):
    """Approximate degrees from km for a small bbox."""
    dlat = radius_km / 111.0
    dlon = radius_km / (111.320 * math.cos(math.radians(lat)) or 1e-6)
    return (lat + dlat, lon - dlon, lat - dlat, lon + dlon)  # (nwlat, nwlng, selat, selng)

def get_purpleair_pm25(lat: float, lon: float,
                       key: str = PURPLEAIR_KEY,
                       radius_km: float = PURPLEAIR_RADIUS_KM,
                       max_age_min: int = PURPLEAIR_MAX_AGE_MIN) -> dict:
    """
    Query PurpleAir v1 sensors in a bbox. Returns {"PM25": mean} if available.
    Uses pm2.5_atm and filters to outdoor sensors. Units are already µg/m³.
    """
    if not key:
        return {}
    try:
        url = "https://api.purpleair.com/v1/sensors"
        nwlat, nwlng, selat, selng = _bbox_from_radius(lat, lon, radius_km)

        fields = "pm2.5_atm,latitude,longitude,last_seen"
        params = {
            "fields": fields,
            "location_type": 0,      # 0 = outdoors
            "max_age": int(max_age_min),
            "nwlat": nwlat, "nwlng": nwlng,
            "selat": selat, "selng": selng,
        }
        headers = {"X-API-Key": key}
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        js = resp.json()

        data = js.get("data", [])
        fields_list = js.get("fields", [])
        if not data or not fields_list:
            return {}

        try:
            idx_pm = fields_list.index("pm2.5_atm")
        except ValueError:
            idx_pm = fields_list.index("pm2.5_alt")

        idx_seen = fields_list.index("last_seen") if "last_seen" in fields_list else None

        pm_vals = []
        now_epoch = int(time.time())
        for row in data:
            try:
                pm = row[idx_pm]
                if pm is None:
                    continue
                if idx_seen is not None:
                    last_seen = row[idx_seen]
                    if isinstance(last_seen, (int, float)) and (now_epoch - last_seen) > max_age_min * 60:
                        continue
                pm_vals.append(float(pm))
            except Exception:
                continue

        if pm_vals:
            # Already in ug/m3 which matches TARGET_UNITS for PM25
            return {"PM25": float(np.mean(pm_vals))}
        return {}
    except Exception as e:
        print("PurpleAir query error:", e)
        return {}

# -------------------------------
# Orchestrator with mock fallback
# -------------------------------
def get_ground_measurements(lat: float, lon: float) -> dict:
    """
    - Try OpenAQ v3 for all six pollutants; convert each to TARGET_UNITS before averaging.
    - If PM2.5 missing, try PurpleAir to fill just PM25 (already ug/m3).
    - If still empty, return conservative mock values (already in TARGET_UNITS).
    """
    # 1) OpenAQ v3 (converted to TARGET_UNITS)
    vals = get_openaq_latest_v3(lat, lon)

    # 2) Fill PM25 from PurpleAir if missing
    if ("PM25" not in vals or vals.get("PM25") is None) and PURPLEAIR_KEY:
        pa = get_purpleair_pm25(lat, lon)
        if "PM25" in pa and pa["PM25"] is not None:
            vals["PM25"] = pa["PM25"]

    # 3) Fallback to mock if nothing collected
    if not vals:
        print("No ground data available from OpenAQ/PurpleAir — using mock ground values.")
        return dict(MOCK_GROUND)

    return vals

# -------------------------------
# Example usage (expects LANDFILL dict with 'lat'/'lon')
# -------------------------------
ground_current = get_ground_measurements(LANDFILL["lat"], LANDFILL["lon"])
print("Ground measurements (OpenAQ v3 / PurpleAir or mock):")
print(ground_current)


# Pipeline 4: Compute satellite AQI (relative percentile mapping), ground AQI (EPA formulas), fused AQI; plotting & provenance

#EPA breakpoints
AQI_BREAKPOINTS = {
    "PM25": [(0.0,12.0,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),(55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)],
    "PM10": [(0,54,0,50),(55,154,51,100),(155,254,101,150),(255,354,151,200),(355,424,201,300),(425,504,301,400),(505,604,401,500)],
    "NO2":  [(0,53,0,50),(54,100,51,100),(101,360,101,150),(361,649,151,200),(650,1249,201,300),(1250,2049,301,400),(2050,3049,401,500)],
    "O3":   [(0,54,0,50),(55,70,51,100),(71,85,101,150),(86,105,151,200),(106,200,201,300)],
    "CO":   [(0.0,4.4,0,50),(4.5,9.4,51,100),(9.5,12.4,101,150),(12.5,15.4,151,200)],
    "SO2":  [(0,35,0,50),(36,75,51,100),(76,185,101,150),(186,304,151,200),(305,604,201,300)]
}

def conc_to_aqi(conc, pollutant_key):
    """Return integer AQI subindex for a pollutant; pollutant_key in AQI_BREAKPOINTS keys.
       Assumes conc units match table (PM: µg/m3, gases: ppb or ppm as suited)."""
    if conc is None or (isinstance(conc, float) and np.isnan(conc)):
        return None
    if pollutant_key not in AQI_BREAKPOINTS:
        return None
    for Clow, Chigh, Ilow, Ihigh in AQI_BREAKPOINTS[pollutant_key]:
        if Clow <= conc <= Chigh:
            aqi = ((Ihigh - Ilow) / (Chigh - Clow)) * (conc - Clow) + Ilow
            return int(round(aqi))
    # Outside table -> clamp to last category max
    return int(round(AQI_BREAKPOINTS[pollutant_key][-1][-1]))

# --- Stabilized satellite percentile mapping controls ---
SAT_AQI_EXPONENT = 1.2   # gentler than 1.5 to avoid tail inflation
SAT_AQI_MAX      = 200   # cap satellite-only AQI so it doesn't dominate
AER_AQI_MAX      = 150   # UVAI tends to be narrow; keep a modest cap

def satellite_percentile_to_aqi(sat_value, hist_values, exponent=SAT_AQI_EXPONENT,
                                max_aqi=SAT_AQI_MAX, use_log=True):
    """
    Map a satellite measurement to a relative AQI via percentile of its own history.
    If use_log=True and values are positive, compute percentile in log10-space (stabilizes wide ranges).
    """
    if sat_value is None or hist_values is None or len(hist_values) == 0:
        return 50  # neutral when no history or value
    arr = np.asarray(hist_values, dtype=float)
    x = float(sat_value)

    # Compute percentile in log space when appropriate (columns vary over orders of magnitude)
    if use_log and np.all(arr > 0) and x > 0:
        arr = np.log10(arr)
        x = np.log10(x)

    perc = float(np.mean(arr <= x))  # 0..1
    scaled = (perc ** exponent) * max_aqi
    return int(round(min(max(0, scaled), max_aqi)))

def build_satellite_aqi(sat_current_dict, sat_hist_dict, exponent=SAT_AQI_EXPONENT):
    """
    Columns (NO2/O3/HCHO) -> log-space percentile; UVAI (AER) -> linear percentile.
    Returns (overall_sat_aqi, dominant_pollutant, subindex_dict)
    """
    sub = {}
    for pol, val in sat_current_dict.items():
        hist = sat_hist_dict.get(pol, [])
        if pol == "AER":
            sub[pol] = satellite_percentile_to_aqi(val, hist, exponent=exponent,
                                                   max_aqi=AER_AQI_MAX, use_log=False)
        else:
            sub[pol] = satellite_percentile_to_aqi(val, hist, exponent=exponent,
                                                   max_aqi=SAT_AQI_MAX, use_log=True)
    if len(sub) == 0:
        return None, None, {}
    dominant = max(sub, key=lambda k: sub[k] if sub[k] is not None else -1)
    return sub[dominant], dominant, sub

def build_ground_aqi(ground_vals):
    """ground_vals: dict with pollutant keys similar to AQI_BREAKPOINTS (PM25, NO2, O3, CO, SO2, PM10)"""
    sub = {}
    mapping = {"PM25":"PM25","PM10":"PM10","NO2":"NO2","O3":"O3","CO":"CO","SO2":"SO2"}
    for k, v in ground_vals.items():
        key = k.upper()
        if key in mapping:
            sub[key] = conc_to_aqi(v, mapping[key])
    if len(sub)==0:
        return None, None, {}
    dominant = max(sub, key=lambda k: sub[k] if sub[k] is not None else -1)
    return sub[dominant], dominant, sub

def fused_aqi_from_components(sat_aqi, ground_aqi, w_ground=FUSED_W_GROUND, w_sat=FUSED_W_SAT):
    """Weighted average; if one missing returns the other"""
    if sat_aqi is None and ground_aqi is None:
        return None
    if ground_aqi is None:
        return sat_aqi
    if sat_aqi is None:
        return ground_aqi
    s = w_ground + w_sat
    return int(round((w_ground/s)*ground_aqi + (w_sat/s)*sat_aqi))

# --- Mock satellite history for percentile mapping ---
# Columns are in molecules/cm^2 (NO2/O3/HCHO). UVAI (AER) is unitless (~ -1..5).
def build_mock_sat_hist(satellite_keys, days=SAT_HIST_DAYS):
    # hourly history -> days*24 samples
    n = days * 24
    medians = {
        "NO2":  2e15,  # tropospheric column
        "O3":   1e18,  # tropospheric column
        "HCHO": 5e15,  # total column
        "AER":  1.0,   # UVAI typical range
    }


    hist = {}
    for k in satellite_keys:
        if k == "AER":
            # UVAI can be slightly negative; keep a realistic band
            arr = np.random.normal(loc=medians.get(k, 1.0), scale=0.6, size=n)
            arr = np.clip(arr, -2.0, 5.0)
        else:
            med = medians.get(k, 1.0)
            # lognormal centered on desired median: mean=ln(median)
            arr = np.random.lognormal(mean=np.log(med), sigma=0.3, size=n)
        hist[k] = arr.tolist()
    return hist

# Demonstrate Pipeline 4 using current sat & ground values
sat_hist = build_mock_sat_hist(list(satellite_current.keys()), days=SAT_HIST_DAYS)

sat_aqi_value, sat_dom, sat_breakdown = build_satellite_aqi(satellite_current, sat_hist)
ground_aqi_value, ground_dom, ground_breakdown = build_ground_aqi(ground_current)
fused_value = fused_aqi_from_components(sat_aqi_value, ground_aqi_value)

print("Satellite-AQI:", sat_aqi_value, "dominant:", sat_dom)
print("Ground-AQI:", ground_aqi_value, "dominant:", ground_dom)
print("Fused-AQI (weights ground {:.2f}, sat {:.2f}):".format(FUSED_W_GROUND, FUSED_W_SAT), fused_value)
print("\nSatellite breakdown (subindices):", sat_breakdown)
print("Ground breakdown (subindices):", ground_breakdown)

# Simple visualization of the current indices and breakdown & provenance

def _as_subindex(entry):
    """Return a numeric subindex from either a dict({'subindex': x}) 
    or a plain number; else None."""

    if isinstance(entry, dict):
        return entry.get("subindex")
    if isinstance(entry, (int, float, np.number)):
        return float(entry)
    return None

def _get_subindex(mapping, key):
    """Safe getter that returns a subindex (or None) from a mapping that may store dicts or numbers."""
    if mapping is None:
        return None
    return _as_subindex(mapping.get(key))

# 1) Show numeric cards
print("=== SUMMARY ===")
print(f"Satellite-AQI: {sat_aqi_value} (dominant: {sat_dom})")
print(f"Ground-AQI   : {ground_aqi_value} (dominant: {ground_dom})")
print(f"Fused-AQI    : {fused_value}\n")

# 2) Bar chart of subindices (sat vs ground) for pollutants present
#pols = sorted(set(list(sat_breakdown.keys()) + list(ground_breakdown.keys())))
#sat_vals_plot = [_get_subindex(sat_breakdown, p) for p in pols]
#ground_vals_plot = [_get_subindex(ground_breakdown, p) for p in pols]

#x = np.arange(len(pols))
#width = 0.35
#fig, ax = plt.subplots(figsize=(10,5))
#ax.bar(x - width/2, [v if v is not None else 0 for v in sat_vals_plot], width, label='Satellite subindex')
#ax.bar(x + width/2, [v if v is not None else 0 for v in ground_vals_plot], width, label='Ground subindex')
#ax.set_xticks(x)
#ax.set_xticklabels(pols)
#ax.set_ylabel("AQI Sub-index")
#ax.set_title("Satellite vs Ground AQI sub-indices (current)")
#ax.legend()
#plt.show()

# 3) Provenance: list sources that contributed to fused AQI
print("\n=== PROVENANCE ===")
print("Satellite sources (TEMPO via Harmony or mock):")
for k, v in satellite_current.items():
    sat_si = _get_subindex(sat_breakdown, k)
    print(f" - {k}: {v} (sat-AQI subindex: {sat_si})")

print("\nGround sources (OpenAQ/PurpleAir or mock):")
for k, v in ground_current.items():
    gnd_si = _get_subindex(ground_breakdown, k)
    print(f" - {k}: {v} (ground-AQI subindex: {gnd_si})")

print(f"\nFusion weights: ground={FUSED_W_GROUND}, satellite={FUSED_W_SAT}")



# Pipeline 5: Meteorology ingestion (OpenWeatherMap Current Weather or mock)

def get_current_meteorology(lat, lon, api_key=OWM_API_KEY, units="metric", timeout=15):
    """
    Fetch current weather from OpenWeatherMap:
      - temp (°C if units='metric')
      - humidity (%)
      - wind_speed (m/s if units='metric')
      - wind_gust (m/s if available)
      - wind_deg (degrees)
      - pressure (hPa)
      - rain (mm in last hour if available, else 3h, else 0)
    """
    if not api_key:
        print("No OWM API key set; using mock meteorology.")
        return {
            "temp": 25.0, "humidity": 40.0,
            "wind_speed": 3.0, "wind_gust": None, "wind_deg": 200.0,
            "pressure": 1012.0, "rain": 0.0
        }

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lon, "appid": api_key, "units": units}
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        js = resp.json()

        main = js.get("main", {}) or {}
        wind = js.get("wind", {}) or {}
        rain = js.get("rain", {}) or {}

        def _f(x, default=None):
            try:
                return float(x)
            except Exception:
                return default

        out = {
            "temp": _f(main.get("temp"), 25.0),
            "humidity": _f(main.get("humidity"), 40.0),
            "wind_speed": _f(wind.get("speed"), 0.0),
            "wind_gust": _f(wind.get("gust"), np.nan),
            "wind_deg": _f(wind.get("deg"), np.nan),
            "pressure": _f(main.get("pressure"), 1012.0),
        }

        # Rain (mm). Prefer 1h; if only 3h, return that; else 0.
        rain_1h = _f(rain.get("1h"), None)
        rain_3h = _f(rain.get("3h"), None)
        if rain_1h is not None:
            out["rain"] = rain_1h
            out["rain_1h"] = rain_1h
        elif rain_3h is not None:
            out["rain"] = rain_3h  # total over last 3 hours
            out["rain_3h"] = rain_3h
        else:
            out["rain"] = 0.0

        return out

    except Exception as e:
        print("OpenWeatherMap error:", e)
        return {
            "temp": 25.0, "humidity": 40.0,
            "wind_speed": 3.0, "wind_gust": None, "wind_deg": 200.0,
            "pressure": 1012.0, "rain": 0.0
        }

# Example usage
current_meteo = get_current_meteorology(LANDFILL["lat"], LANDFILL["lon"])
print("Current meteo (live or mock):", current_meteo)


# Pipeline 6 - LightGBM forecast using 90 days real data (or fallback synthetic training data)
# =========================

# Helper: safe float
def _to_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def try_build_real_history(days=90):
    """
    Placeholder: in production, fetch 90 days of hourly:
      - Ground AQI (OpenAQ + your fusion)
      - Satellite proxies (TEMPO)
      - Meteorology (OWM history)
    Build an hourly dataframe with at least:
      datetime, fused_AQI, temp, humidity, wind_speed, [wind_gust], wind_deg, pressure, [rain]
    """
    return None

def build_synthetic_history(days=90):
    n_hours = days * 24
    rng = pd.date_range(end=datetime.utcnow(), periods=n_hours, freq="H")
    np.random.seed(42)

    # Base meteorology
    temp = 20 + 5*np.sin(np.linspace(0, 5, n_hours)) + np.random.randn(n_hours)
    humidity = np.clip(40 + 8*np.random.randn(n_hours), 0, 100)
    wind_speed = np.abs(3 + np.random.randn(n_hours))              # m/s, non-negative
    wind_deg = np.random.randint(0, 360, n_hours)                  # degrees
    pressure = 1010 + np.random.randn(n_hours)                     # hPa

    # Wind gusts: typically >= wind_speed
    wind_gust = np.maximum(wind_speed, wind_speed + np.abs(np.random.randn(n_hours) * 1.5))

    # Rain (mm): sparse with occasional bursts
    wet_mask = np.random.rand(n_hours) < 0.2                       # ~20% wet hours
    rain = np.zeros(n_hours, dtype=float)
    rain[wet_mask] = np.random.gamma(shape=1.3, scale=1.2, size=wet_mask.sum())  # ~0–10 mm/hr typical

    # Synthetic AQI series (sat + ground)
    sat_AQI = np.clip(60 + 30*np.sin(np.linspace(0, 20, n_hours)) + np.random.randn(n_hours)*15, 0, 500)
    ground_AQI = np.clip(55 + 25*np.sin(np.linspace(0, 18, n_hours)) + np.random.randn(n_hours)*10, 0, 500)

    df = pd.DataFrame({
        "datetime": rng,
        "sat_AQI": sat_AQI,
        "ground_AQI": ground_AQI,
        "fused_AQI": np.nan,  # filled below
        "temp": temp,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "wind_gust": wind_gust,
        "wind_deg": wind_deg.astype(float),
        "pressure": pressure,
        "rain": rain,  # mm in the last hour
    })

    # Compute fused AQI from your weights (use globals if defined, else sane defaults)
    _w_g = globals().get("FUSED_W_GROUND", 0.7)
    _w_s = globals().get("FUSED_W_SAT", 0.3)
    df["fused_AQI"] = (_w_g * df["ground_AQI"] + _w_s * df["sat_AQI"]).astype(float)
    return df

# 1) Build history (real or synthetic)
history_df = try_build_real_history(days=90)
if history_df is None:
    print("Real history not available — using synthetic 90-day history for model training.")
    history_df = build_synthetic_history(days=90)

# (Optional) one-time numeric cleanup if you loaded from CSV/Parquet
num_cols = ["fused_AQI","temp","humidity","wind_speed","wind_gust","wind_deg","pressure","rain"]
for c in num_cols:
    if c in history_df.columns:
        history_df[c] = pd.to_numeric(history_df[c], errors="coerce")

# 2) Dynamic features (include gust/rain if present in history)
base_features = ["fused_AQI", "temp", "humidity", "wind_speed", "wind_deg", "pressure"]
extra_features = []
if "wind_gust" in history_df.columns: extra_features.append("wind_gust")
if "rain" in history_df.columns:      extra_features.append("rain")
features = base_features + extra_features

# 3) Train LightGBM (autoregressive: predict fused_AQI at t+1)
try:
    import lightgbm as lgb

    df = history_df.copy().reset_index(drop=True)
    if "fused_AQI" not in df.columns:
        raise ValueError("history_df must contain 'fused_AQI'.")

    # Create next-hour target
    df["target"] = df["fused_AQI"].shift(-1)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    # Ensure feature/target dtypes are numeric floats
    for col in features + ["target"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build matrices
    X = df[features].astype(np.float32)
    y = df["target"].astype(np.float32)

    # Simple time-based split
    split = max(1, int(len(X) * 0.8))
    X_train, y_train = X.iloc[:split], y.iloc[:split]
    X_val,   y_val   = X.iloc[split:], y.iloc[split:]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {"objective": "regression", "metric": "rmse", "verbosity": -1}
    callbacks = [
        lgb.early_stopping(stopping_rounds=20, verbose=False),
        # lgb.log_evaluation(period=50),  # uncomment for periodic logs
    ]

    model = lgb.train(params, train_data, valid_sets=[val_data], num_boost_round=200, callbacks=callbacks)
    valid_rmse = model.best_score.get("valid_0", {}).get("rmse")
    print("Model trained. Valid RMSE:", valid_rmse)

    # 4) Forecast next 72 hours (autoregressive)
    horizon = 72

    # Pull fused_value/current_meteo from globals if available; otherwise fall back
    fused_value = globals().get("fused_value", float(df["fused_AQI"].iloc[-1]))
    current_meteo = globals().get("current_meteo", {})

    # Build current feature vector (numeric); keep meteo constant over horizon for this simple demo
    def _fv_get(name, default=np.nan):
        if name == "fused_AQI":
            return _to_float(fused_value, default)
        # prefer current meteo, else last historical value
        if name in current_meteo:
            return _to_float(current_meteo.get(name), default)
        if name in df.columns:
            return _to_float(df[name].iloc[-1], default)
        return default

    cur_feats = {f: _fv_get(f) for f in features}

    # Helper to build numeric single-row DF with exact column order
    def _num_row_dict(d, cols):
        row = {c: _to_float(d.get(c), np.nan) for c in cols}
        return pd.DataFrame([row], columns=cols).astype(np.float32)

    now = datetime.utcnow()
    rows = []
    for h in range(horizon):
        x_df = _num_row_dict(cur_feats, features)
        pred = float(model.predict(x_df, num_iteration=model.best_iteration)[0])
        rows.append({"datetime": now + timedelta(hours=h + 1), "pred_AQI": pred})
        cur_feats["fused_AQI"] = pred  # autoregressive feedback

    forecast_df = pd.DataFrame(rows)

    # 5) Plot forecast
    plt.figure(figsize=(12, 5))
    plt.plot(forecast_df["datetime"], forecast_df["pred_AQI"], label="Predicted fused AQI")
    plt.axhline(50,  color="green",  linestyle="--", label="Good threshold")
    plt.axhline(100, color="orange", linestyle="--", label="Moderate threshold")
    plt.axhline(150, color="red",    linestyle="--", label="Unhealthy threshold")
    plt.xlabel("Datetime")
    plt.ylabel("AQI")
    plt.title("72-hour AQI forecast (fused index)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print("Training error:", e)
    forecast_df = pd.DataFrame()


# ===== Orchestrator / Trigger =====

# Sensible fallbacks if not defined earlier in the file
SAT_HIST_DAYS   = globals().get("SAT_HIST_DAYS", 7)
FUSED_W_GROUND  = globals().get("FUSED_W_GROUND", 0.7)
FUSED_W_SAT     = globals().get("FUSED_W_SAT", 0.3)

def _fire_detected(fires_df: pd.DataFrame, radius_km: float) -> bool:
    """Positive if any detection within radius_km; if dist is absent, any row counts."""
    if fires_df is None or fires_df.empty:
        return False
    if "dist_km" in fires_df.columns:
        return bool((fires_df["dist_km"] <= radius_km).any())
    return True

def _dynamic_feature_list(df: pd.DataFrame) -> list:
    """Train/forecast features; includes wind_gust/rain if present in history."""
    base = ["fused_AQI", "temp", "humidity", "wind_speed", "wind_deg", "pressure"]
    extra = []
    if isinstance(df, pd.DataFrame):
        if "wind_gust" in df.columns: extra.append("wind_gust")
        if "rain" in df.columns: extra.append("rain")
    return base + extra

def train_and_forecast_safe(
    history_df,
    fused_value,
    current_meteo: dict,
    horizon: int = 72,
    features=("fused_AQI", "temp", "humidity", "wind_speed", "wind_deg", "pressure"),
    min_rows: int = 24,
):
    """
    Train a LightGBM regressor to predict next-hour fused_AQI and produce a horizon-hour forecast.
    Robust to missing/dirty data. Falls back to a naive persistence forecast on any failure.

    Returns:
        forecast_df (pd.DataFrame with ['datetime','pred_AQI']),
        valid_rmse (float | None),
        fallback_reason (str | None)  # None if LightGBM succeeded
    """

    def _naive_forecast(last_val, H):
        v = float(last_val) if (last_val is not None and np.isfinite(last_val)) else 75.0
        t0 = pd.Timestamp.utcnow()
        return pd.DataFrame(
            {"datetime": [t0 + pd.Timedelta(hours=h + 1) for h in range(H)],
             "pred_AQI": [v] * H}
        )

    # 0) Guard: history present?
    if history_df is None or len(history_df) == 0:
        return _naive_forecast(fused_value, horizon), None, "history_df is empty"

    df = history_df.copy()

    # 1) Ensure required feature columns exist
    for col in features:
        if col not in df.columns:
            df[col] = np.nan

    # 2) Coerce to numeric and clean NaNs/inf
    for col in set(features) | {"fused_AQI"}:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 3) Build next-hour target and drop rows without complete data
    df["target"] = df["fused_AQI"].shift(-1)
    df = df.dropna(subset=list(features) + ["target"]).reset_index(drop=True)

    if len(df) < min_rows:
        return _naive_forecast(fused_value, horizon), None, f"too few training rows ({len(df)}) after cleaning"

    # 4) Time-based split
    split = int(len(df) * 0.8)
    X_train, y_train = df.loc[:split - 1, list(features)], df.loc[:split - 1, "target"]
    X_val,   y_val   = df.loc[split:,    list(features)], df.loc[split:,    "target"]

    # Final sanity
    if X_train.isna().any().any() or X_val.isna().any().any():
        return _naive_forecast(fused_value, horizon), None, "NaNs remain after cleaning"

    try:
        import lightgbm as lgb

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data   = lgb.Dataset(X_val,   label=y_val, reference=train_data)

        params = {"objective": "regression", "metric": "rmse", "verbosity": -1}
        callbacks = [
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=50),
        ]

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=200,
            callbacks=callbacks,
        )

        # Extract RMSE safely
        valid_rmse = None
        try:
            valid_rmse = model.best_score.get("valid_0", {}).get("rmse")
        except Exception:
            pass

        # 5) Build current feature vector
        cur_feats = {k: float(current_meteo.get(k, 0.0)) for k in features if k != "fused_AQI"}
        cur_feats["fused_AQI"] = float(fused_value) if fused_value is not None else float(df["fused_AQI"].iloc[-1])

        # 6) Forecast iteratively (autoregressive on fused_AQI)
        out = []
        t0 = pd.Timestamp.utcnow()
        for h in range(horizon):
            x_df = pd.DataFrame([cur_feats])[list(features)]
            try:
                pred = float(model.predict(x_df, num_iteration=getattr(model, "best_iteration", None))[0])
            except Exception:
                pred = float(model.predict(x_df)[0])
            out.append({"datetime": t0 + pd.Timedelta(hours=h + 1), "pred_AQI": pred})
            cur_feats["fused_AQI"] = pred  # autoregressive update

        forecast_df = pd.DataFrame(out)
        return forecast_df, valid_rmse, None

    except Exception as e:
        # LightGBM not installed, version issues, or runtime error
        return _naive_forecast(fused_value, horizon), None, f"LightGBM error: {e.__class__.__name__}"


def run_workflow_if_fire(lat: float, lon: float,
                         radius_km: float = 5.0,
                         run_parallel: bool = True,
                         tempo_days_back: int = 1,
                         tempo_timeout_s: int = 600,
                         ground_timeout_s: int = 180,
                         forecast_horizon_h: int = 72):
    """
    Orchestrator:
      1) Pipeline 1 (FIRMS). If fire detected:
      2) Pipeline 2 (TEMPO) + 3 (Ground) in parallel (or sequential).
      3) Pipeline 4: build sat/ground/fused AQI.
      4) Pipeline 5: get current meteorology.
      5) Pipeline 6: train and forecast.
    Returns a dict with all outputs.
    """
    # --- Pipeline 1: FIRMS ---
    fires = detect_fire_near_landfill(lat, lon, radius_km=radius_km)
    fire_yes = _fire_detected(fires, radius_km)
    print(f"FIRMS fire detected: {fire_yes}")

    if not fire_yes:
        return {"fire_detected": False, "fires": fires}

    # --- Pipelines 2 & 3: in parallel (or sequential) ---
    sat = {}
    ground = {}
    if run_parallel:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_sat = pool.submit(request_tempo_subset, lat, lon, tempo_days_back)
            fut_gnd = pool.submit(get_ground_measurements, lat, lon)
            try:
                sat = fut_sat.result(timeout=tempo_timeout_s)
            except Exception as e:
                print("Pipeline 2 (TEMPO) error:", e)
            try:
                ground = fut_gnd.result(timeout=ground_timeout_s)
            except Exception as e:
                print("Pipeline 3 (Ground) error:", e)
    else:
        try:
            sat = request_tempo_subset(lat, lon, tempo_days_back)
        except Exception as e:
            print("Pipeline 2 (TEMPO) error:", e)
        try:
            ground = get_ground_measurements(lat, lon)
        except Exception as e:
            print("Pipeline 3 (Ground) error:", e)

    # --- Pipeline 4: AQIs ---
    try:
        sat_hist = build_mock_sat_hist(list(sat.keys()), days=SAT_HIST_DAYS)
        sat_aqi_value, sat_dom, sat_breakdown = build_satellite_aqi(sat, sat_hist)
        ground_aqi_value, ground_dom, ground_breakdown = build_ground_aqi(ground)
        fused_value = fused_aqi_from_components(sat_aqi_value, ground_aqi_value,
                                                w_ground=FUSED_W_GROUND, w_sat=FUSED_W_SAT)
    except Exception as e:
        print("Pipeline 4 error:", e)
        sat_aqi_value = sat_dom = ground_aqi_value = ground_dom = fused_value = None
        sat_breakdown, ground_breakdown = {}, {}

    # --- Pipeline 5: current meteorology (OpenWeatherMap) ---
    try:
        current_meteo = get_current_meteorology(lat, lon)
    except Exception as e:
        print("Pipeline 5 (meteo) error:", e)
        current_meteo = {"temp": np.nan, "humidity": np.nan, "wind_speed": np.nan,
                         "wind_gust": None, "wind_deg": np.nan, "pressure": np.nan, "rain": 0.0}
        
    # Ensure the keys trainer expects exist (safe defaults)
    for k in ("temp", "humidity", "wind_speed", "wind_deg", "pressure"):
        current_meteo.setdefault(k, 0.0)

    # --- Pipeline 6: training/forecast (requires history_df) ---
    try:
        history_df = try_build_real_history(days=90)
        if history_df is None or not isinstance(history_df, pd.DataFrame) or history_df.empty:
            history_df = build_synthetic_history(days=90)
    except Exception as e:
        print("History build error; using synthetic:", e)
        history_df = build_synthetic_history(days=90)

    try:
        forecast_df, valid_rmse, fallback_reason = train_and_forecast_safe(
            history_df=history_df,
            fused_value=fused_value,
            current_meteo=current_meteo,
            horizon=int(forecast_horizon_h),
        )
    except Exception as e:
        print("Training error (unexpected); falling back to persistence:", e)
        # Absolute safety fallback: persistence
        t0 = pd.Timestamp.utcnow()
        v = float(fused_value) if fused_value is not None else 75.0
        forecast_df = pd.DataFrame({
            "datetime": [t0 + pd.Timedelta(hours=h+1) for h in range(int(forecast_horizon_h))],
            "pred_AQI": [v] * int(forecast_horizon_h)
        })
        valid_rmse, fallback_reason = None, "hard fallback: exception during training"

    return {
        "fire_detected": True,
        "fires": fires,
        "satellite_current": sat,
        "ground_current": ground,
        "sat_aqi_value": sat_aqi_value,
        "sat_dominant": sat_dom,
        "sat_breakdown": sat_breakdown,
        "ground_aqi_value": ground_aqi_value,
        "ground_dominant": ground_dom,
        "ground_breakdown": ground_breakdown,
        "fused_aqi": fused_value,
        "current_meteo": current_meteo,
        "forecast_df": forecast_df,
        "valid_rmse": valid_rmse,
        "training_fallback_reason": fallback_reason,
    }

# ===== Hourly scheduler: check FIRMS and trigger pipelines =====

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    _LOCAL_TZ = ZoneInfo(os.environ.get("LOCAL_TZ", "Europe/Rome"))
except Exception:
    _LOCAL_TZ = None

def _fmt_dt(dt_utc: datetime) -> str:
    """Pretty-print a UTC datetime and (optionally) local time."""
    if _LOCAL_TZ is not None:
        loc = dt_utc.replace(tzinfo=timezone.utc).astimezone(_LOCAL_TZ)
        return f"{loc.strftime('%Y-%m-%d %H:%M:%S %Z')} ({dt_utc.strftime('%Y-%m-%d %H:%M:%SZ')} UTC)"
    return dt_utc.strftime('%Y-%m-%d %H:%M:%SZ')

def hourly_firms_scheduler(lat: float, lon: float,
                           radius_km: float = 5.0,
                           run_parallel: bool = True,
                           tempo_days_back: int = 1,
                           once_immediately: bool = True,
                           jitter_seconds: float = 10.0,
                           stop_after_runs: int | None = None):
    """
    Every hour (on the hour), run:
      - Pipeline 1 (FIRMS); if fire detected ->
      - Pipelines 2 & 3 (TEMPO + Ground), then 4 (AQIs), 5 (meteo), 6 (train+forecast).
    Prints a short summary each run. Use Ctrl+C to stop.

    Args:
      once_immediately: if True, runs a check right away, then hourly.
      jitter_seconds: add small random delay to avoid thundering herd.
      stop_after_runs: if set, stop after N runs (useful for testing).
    """
    runs = 0

    def _do_one_run():
        nonlocal runs
        try:
            print(f"\n[{_fmt_dt(datetime.utcnow())}] Running FIRMS check...")
            results = run_workflow_if_fire(lat, lon,
                                           radius_km=radius_km,
                                           run_parallel=run_parallel,
                                           tempo_days_back=tempo_days_back)
            if results.get("fire_detected"):
                print(f"🔥 Fire detected. Fused AQI: {results.get('fused_aqi')}")
                print(f"   Satellite AQI: {results.get('sat_aqi_value')} (dom: {results.get('sat_dominant')})")
                print(f"   Ground AQI   : {results.get('ground_aqi_value')} (dom: {results.get('ground_dominant')})")
                if "current_meteo" in results:
                    print(f"   Meteo: {results['current_meteo']}")
                fc = results.get("forecast_df")
                if fc is not None and not fc.empty:
                    print("   Forecast (first 3 rows):")
                    print(fc.head(3).to_string(index=False))
            else:
                print("No fire near the site. Pipelines 2–6 not triggered.")
        except Exception as e:
            print("Scheduler run error:", e)
            traceback.print_exc()
        runs += 1

    try:
        if once_immediately:
            _do_one_run()
            if stop_after_runs and runs >= stop_after_runs:
                return

        while True:
            if stop_after_runs and runs >= stop_after_runs:
                break

            now = datetime.utcnow()
            # sleep until top of next hour, plus small jitter
            next_top = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
            sleep_s = max(1.0, (next_top - now).total_seconds() + random.uniform(0, jitter_seconds))
            print(f"Next FIRMS check at {_fmt_dt(next_top)} (+jitter ≤ {int(jitter_seconds)}s). Sleeping ~{int(sleep_s)}s.")
            time.sleep(sleep_s)

            _do_one_run()

    except KeyboardInterrupt:
        print("\nScheduler stopped by user.")

# ===== Optional: run the hourly scheduler from CLI =====
if __name__ == "__main__":
    try:
        lat, lon = LANDFILL["lat"], LANDFILL["lon"]
    except Exception:
        lat = float(os.environ.get("LANDFILL_LAT", 34.1439))
        lon = float(os.environ.get("LANDFILL_LON", -118.6615))

    # Run forever; for a quick test, set stop_after_runs=1
    hourly_firms_scheduler(lat, lon,
                           radius_km=5.0,
                           run_parallel=True,
                           tempo_days_back=1,
                           once_immediately=True,
                           jitter_seconds=8.0,
                           stop_after_runs=1) # set to 1 for testing, else None


# ===== Optional: simple CLI entrypoint =====
if __name__ == "__main__":
    # Example: run the full workflow for your landfill point
    try:
        lat, lon = LANDFILL["lat"], LANDFILL["lon"]
    except Exception:
        # Fallback if LANDFILL not defined
        lat = float(os.environ.get("LANDFILL_LAT", 34.1439))
        lon = float(os.environ.get("LANDFILL_LON", -118.6615))

    results = run_workflow_if_fire(lat, lon, radius_km=5.0, run_parallel=True)
    print("\n=== SUMMARY ===")
    print("Fire detected:", results.get("fire_detected"))
    if results.get("fire_detected"):
        print("Fused AQI:", results.get("fused_aqi"))
        print("Satellite AQI:", results.get("sat_aqi_value"), "dominant:", results.get("sat_dominant"))
        print("Ground AQI:", results.get("ground_aqi_value"), "dominant:", results.get("ground_dominant"))
        print("Current meteo:", results.get("current_meteo"))
        if not results.get("forecast_df", pd.DataFrame()).empty:
            print("Forecast sample:\n", results["forecast_df"].head().to_string(index=False))
