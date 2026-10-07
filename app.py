import streamlit as st
import pandas as pd
import time
from sklearn.ensemble import IsolationForest

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("simulation.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = load_data()
timestamps = sorted(df["timestamp"].unique())

# ------------------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------------------
def train_model(df):
    features = df[[
        "pressure_psi",
        "flow_rate_ls",
        "temperature_c",
        "vibration_mms",
        "power_kw"
    ]]
    model = IsolationForest(
        n_estimators=200,
        contamination=0.20,
        random_state=42
    )
    model.fit(features)
    return model

model = train_model(df)

# ------------------------------------------------------------
# TREND COMMENTARY ENGINE
# ------------------------------------------------------------
TREND_HINTS = {
    "vibration_mms": "Vibration rising — monitor bearings",
    "temperature_c": "Temperature rising — check cooling flow",
    "flow_rate_ls": "Flow dropping — suction strainer may be clogging",
    "pressure_psi": "Pressure falling — possible impeller wear",
    "power_kw": "Power draw increasing — check motor load"
}

def trend_comment(row, prev_row):
    if prev_row is None:
        return "Stable readings — continue standard inspection"

    deltas = {
        "vibration_mms": row["vibration_mms"] - prev_row["vibration_mms"],
        "temperature_c": row["temperature_c"] - prev_row["temperature_c"],
        "flow_rate_ls": prev_row["flow_rate_ls"] - row["flow_rate_ls"],
        "pressure_psi": prev_row["pressure_psi"] - row["pressure_psi"],
        "power_kw": row["power_kw"] - prev_row["power_kw"]
    }

    signal = max(deltas, key=deltas.get)

    if deltas[signal] > 0.5:
        return TREND_HINTS[signal]
    else:
        return "Stable readings — continue standard inspection"

# ------------------------------------------------------------
# RISK → COLOUR SCALE
# ------------------------------------------------------------
def risk_color(score):
    if score < 20:
        return "#00CC00"
    elif score < 40:
        return "#99CC00"
    elif score < 60:
        return "#FFCC00"
    elif score < 80:
        return "#FF6600"
    else:
        return "#CC0000"

# ------------------------------------------------------------
# RISK TREND ARROW
# ------------------------------------------------------------
def trend_arrow(current, previous):
    if previous is None:
        return "➖"
    if current > previous:
        return "▲"
    if current < previous:
        return "▼"
    return "➖"

# ------------------------------------------------------------
# UI SETUP
# ------------------------------------------------------------
st.set_page_config(page_title="Warehouse Pump Dashboard", layout="wide")
st.title("⚡ Real-Time Pump Failure Risk Monitor")

speed = st.sidebar.slider("Playback Speed (seconds per tick)", 0.1, 5.0, 1.0)
skip_ahead = st.sidebar.slider("Skip Ahead (ticks)", 0, 500, 0)

mission_box = st.empty()
time_box = st.empty()
tiles_box = st.empty()

# ------------------------------------------------------------
# STATE
# ------------------------------------------------------------
if "cumulative_risk" not in st.session_state:
    st.session_state.cumulative_risk = {}

if "trend_risk" not in st.session_state:
    st.session_state.trend_risk = {}

if "prev_scores" not in st.session_state:
    st.session_state.prev_scores = {}

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
index = 0
while True:
    current_time = timestamps[index]
    current_df = df[df["timestamp"] == current_time]

    # ML scoring
    features = current_df[[
        "pressure_psi",
        "flow_rate_ls",
        "temperature_c",
        "vibration_mms",
        "power_kw"
    ]]
    raw_scores = model.decision_function(features)

    risk_raw = -raw_scores
    risk_norm = (risk_raw - risk_raw.min()) / (risk_raw.max() - risk_raw.min())
    current_df["risk_instant"] = (risk_norm * 100).clip(0, 100)

    # Cumulative smoothing
    cumulative_values = []
    for _, row in current_df.iterrows():
        pid = row["pump_id"]
        instant = row["risk_instant"]
        prev = st.session_state.cumulative_risk.get(pid, instant)
        cumulative = prev * 0.9 + instant * 0.1
        st.session_state.cumulative_risk[pid] = cumulative
        cumulative_values.append(cumulative)
    current_df["risk_cumulative"] = cumulative_values

    # Trend smoothing
    trend_values = []
    for _, row in current_df.iterrows():
        pid = row["pump_id"]
        cumulative = row["risk_cumulative"]
        prev = st.session_state.trend_risk.get(pid, cumulative)
        trend = prev * 0.7 + cumulative * 0.3
        st.session_state.trend_risk[pid] = trend
        trend_values.append(trend)
    current_df["risk_score"] = trend_values

    # Trend-up detection
    current_df["trend_up"] = current_df.apply(
        lambda r: (r["risk_score"] - st.session_state.prev_scores.get(r["pump_id"], 0)) > 0.2,
        axis=1
    )

    # ------------------------------------------------------------
    # TILES FILTER (FIXED):
    # Show ALL pumps with risk >= 60 (always updating)
    # ------------------------------------------------------------
    tiles_df = current_df[current_df["risk_score"] >= 60].copy()

    # ORDER BY TOP RISK FIRST
    tiles_df = tiles_df.sort_values("risk_score", ascending=False)

    # ------------------------------------------------------------
    # FIX: Remove stale prev_scores for pumps no longer visible
    # ------------------------------------------------------------
    visible_ids = set(tiles_df["pump_id"])
    stored_ids = set(st.session_state.prev_scores.keys())

    for pid in stored_ids:
        if pid not in visible_ids:
            del st.session_state.prev_scores[pid]

    # ------------------------------------------------------------
    # HOURLY MISSION FILTER:
    # risk >= 60 AND trending upward
    # ------------------------------------------------------------
    mission_df = current_df[
        (current_df["risk_score"] >= 60) &
        (current_df["trend_up"] == True)
    ].copy()

    # ------------------------------------------------------------
    # HOURLY MISSION RENDER
    # ------------------------------------------------------------
    actions = []
    for _, row in mission_df.iterrows():
        pid = row["pump_id"]

        prev_row_df = df[(df["pump_id"] == pid) & (df["timestamp"] < current_time)].sort_values("timestamp").tail(1)
        prev_row = prev_row_df.iloc[0] if not prev_row_df.empty else None

        comment = trend_comment(row, prev_row)
        actions.append(f"- **{pid}** — {comment}")

    if len(actions) == 0:
        mission_box.markdown(
            """
            <div style="background-color:#222;padding:20px;border-radius:10px;color:white;">
                <h2>🔧 Your Hourly Mission</h2>
                <p>No pumps currently exceed risk level 60 with upward trend.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        mission_box.markdown(
            f"""
            <div style="background-color:#222;padding:20px;border-radius:10px;color:white;">
                <h2>🔧 Your Hourly Mission</h2>
                { "<br>".join(actions) }
            </div>
            """,
            unsafe_allow_html=True
        )

    # ------------------------------------------------------------
    # TIME DISPLAY
    # ------------------------------------------------------------
    time_box.markdown(f"### ⏱️ Current Time: **{current_time.strftime('%H:%M:%S')}**")

    # ------------------------------------------------------------
    # TILES (always updating for risk >= 60)
    # ------------------------------------------------------------
    cols = tiles_box.columns(5)

    for i, (_, row) in enumerate(tiles_df.iterrows()):
        pid = row["pump_id"]
        current_risk = row["risk_score"]
        prev_risk = st.session_state.prev_scores.get(pid)
        arrow = trend_arrow(current_risk, prev_risk)
        st.session_state.prev_scores[pid] = current_risk

        tile_color = risk_color(current_risk)

        with cols[i % 5]:
            st.markdown(
                f"""
                <div style="
                    background-color:{tile_color};
                    padding:25px;
                    border-radius:12px;
                    color:white;
                    text-align:center;
                    font-size:20px;
                    margin-bottom:20px;
                ">
                    <h2 style="font-size:32px;">{pid}</h2>
                    <h3>Risk: {current_risk:.0f} {arrow}</h3>
                    <p>Temp: {row['temperature_c']} °C</p>
                    <p>Vibration: {row['vibration_mms']} mm/s</p>
                    <p>Flow: {row['flow_rate_ls']} L/s</p>
                    <p>Pressure: {row['pressure_psi']} psi</p>
                    <p>Power: {row['power_kw']} kW</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ------------------------------------------------------------
    # ADVANCE TIME
    # ------------------------------------------------------------
    index = (index + 1 + skip_ahead) % len(timestamps)
    time.sleep(speed)
