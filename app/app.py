import sys
from pathlib import Path

import folium
import streamlit as st
import plotly.graph_objects as go
from folium.plugins import AntPath
from streamlit_folium import st_folium

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
from src.live_prediction import predict_live_train

st.set_page_config(
    page_title="RailSense | Train Intelligence",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- DESIGN SYSTEM ----------
st.markdown("""

<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --bg:#07111f; --panel:#0d1d30; --panel2:#10263d; --line:rgba(111,178,220,.17); --muted:#91a9bf; --text:#eaf4ff; --cyan:#35d8ef; --blue:#398bff; }
.stApp { background: radial-gradient(ellipse at 12% 0%, #12365a 0%, transparent 36%), linear-gradient(145deg,#06101d,#091a2b 55%,#071421); color:var(--text); font-family:'DM Sans',sans-serif; }
.block-container { max-width:1500px; padding-top:1.5rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#081827,#0b1d30); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--text); }
#MainMenu, footer { visibility:hidden; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:var(--text) !important; }
p, label, .stCaption { color:#b4c7d9; }
.stTextInput input { background:#0d2136!important; color:#fff!important; border:1px solid #285477!important; border-radius:11px!important; min-height:46px; }
.stButton>button, .stFormSubmitButton>button { background:linear-gradient(100deg,#1678db,#11b8c9)!important; color:white!important; border:0!important; border-radius:11px!important; min-height:45px; font-weight:800!important; box-shadow:0 8px 25px rgba(17,184,201,.16); transition:.2s ease; }
.stButton>button:hover,.stFormSubmitButton>button:hover { transform:translateY(-2px); box-shadow:0 10px 30px rgba(17,184,201,.28); }
[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(17,43,67,.96),rgba(10,27,44,.96)); border:1px solid var(--line); border-radius:16px; padding:17px 18px; min-height:120px; box-shadow:0 12px 32px rgba(0,0,0,.13); }
[data-testid="stMetricLabel"] { color:#91abc2!important; font-size:12px!important; font-weight:700!important; letter-spacing:.08em; text-transform:uppercase; }
[data-testid="stMetricValue"] { color:#eaf8ff!important; font-family:'Space Grotesk',sans-serif; font-weight:700!important; }
[data-testid="stMetricDelta"] { font-size:12px; }
.stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { background:#0b1c2e; border:1px solid var(--line); border-radius:10px 10px 0 0; color:#a9bfd2; padding:10px 18px; }
.stTabs [aria-selected="true"] { background:#123453!important; color:#42d9ef!important; border-color:#23658b!important; }
div[data-testid="stVerticalBlockBorderWrapper"] { background:rgba(12,30,48,.65); border:1px solid var(--line); border-radius:16px; }
hr { border-color:var(--line); }
[data-testid="stAlert"] { border-radius:12px; }
.hero { position:relative; overflow:hidden; padding:28px 30px; border:1px solid rgba(72,194,232,.25); border-radius:22px; background:linear-gradient(112deg,rgba(12,39,64,.98),rgba(10,83,122,.75) 58%,rgba(7,35,56,.95)); margin-bottom:22px; }
.hero:after { content:'🚄'; position:absolute; right:6%; top:4px; font-size:94px; opacity:.16; transform:rotate(-5deg); }
.eyebrow { color:#5de2f1; font-size:11px; font-weight:800; letter-spacing:.19em; text-transform:uppercase; }
.hero h1 { font-size:clamp(30px,4vw,48px); margin:7px 0 3px; }
.hero p { margin:0; color:#c2d9eb; font-size:14px; }
.section-head { display:flex; align-items:center; gap:10px; margin:18px 0 12px; color:#dff6ff; font-family:'Space Grotesk',sans-serif; font-size:19px; font-weight:700; }
.section-head:before { content:''; width:4px; height:22px; border-radius:5px; background:linear-gradient(#39d8ef,#398bff); }
.panel { background:linear-gradient(145deg,rgba(16,39,61,.92),rgba(9,25,41,.96)); border:1px solid var(--line); border-radius:17px; padding:18px; height:100%; }
.panel-label { color:#8faac1; font-size:11px; font-weight:800; letter-spacing:.1em; text-transform:uppercase; }
.panel-value { color:#eaf8ff; font-family:'Space Grotesk',sans-serif; font-size:23px; font-weight:700; margin-top:8px; }
.panel-sub { color:#8faac1; font-size:12px; margin-top:5px; }
.pill { display:inline-block; border:1px solid rgba(53,216,239,.35); color:#55ddec; background:rgba(53,216,239,.08); border-radius:30px; padding:5px 10px; font-size:11px; font-weight:800; letter-spacing:.06em; }
.muted { color:#8faac1; font-size:12px; }
.footer { text-align:center; color:#718da5; font-size:12px; padding:25px 0 5px; }

/* ---------- PREMIUM LAYOUT UPGRADE ---------- */
header[data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="stToolbar"] {
    background: transparent !important;
}
[data-testid="stDecoration"] {
    display: none !important;
}
[data-testid="stSidebar"] {
    min-width: 245px !important;
    max-width: 245px !important;
}
.block-container {
    padding-top: 1rem !important;
    max-width: 1600px !important;
}
.hero {
    padding: 34px 38px;
    min-height: 175px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 15px 45px rgba(0,0,0,.18);
}
.hero h1 {
    letter-spacing: -1.5px;
}
.hero p {
    max-width: 650px;
    line-height: 1.8;
}
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 15px;
    margin: 12px 0 25px;
}
.kpi-card {
    background: linear-gradient(145deg,#122e49,#0b1d30);
    border: 1px solid rgba(75,180,225,.22);
    border-radius: 17px;
    padding: 20px;
    min-height: 135px;
    transition: .25s ease;
}
.kpi-card:hover {
    transform: translateY(-4px);
    border-color: #35d8ef;
    box-shadow: 0 12px 30px rgba(0,180,220,.12);
}
.kpi-label {
    color: #91abc2;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.2px;
}
.kpi-value {
    color: #eaf8ff;
    font-size: clamp(17px,1.5vw,23px);
    font-weight: 800;
    line-height: 1.4;
    overflow-wrap: anywhere;
    margin-top: 13px;
}
.kpi-sub {
    color: #829eb7;
    font-size: 11px;
    margin-top: 7px;
}
@media (max-width: 900px) {
    .kpi-grid { grid-template-columns: repeat(2,minmax(0,1fr)); }
}
@media (max-width: 520px) {
    .kpi-grid { grid-template-columns: 1fr; }
}
/* Hide Streamlit platform controls */
header[data-testid="stHeader"] {
    visibility: hidden !important;
}

[data-testid="stToolbar"] {
    visibility: hidden !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

#MainMenu,
footer {
    visibility: hidden !important;
}
</style>

""", unsafe_allow_html=True)


# ---------- HELPERS ----------
def section(title):
    st.markdown(f'<div class="section-head">{title}</div>', unsafe_allow_html=True)

def safe_delay(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None

def delay_label(value):
    if value is None:
        return "UNAVAILABLE", "#91a9bf", "No estimate available"
    if value > 5:
        return "EXPECTED LATE", "#ff8195", "Estimated arrival after schedule"
    if value < -5:
        return "EXPECTED EARLY", "#56d69a", "Estimated arrival before schedule"
    return "NEAR ON TIME", "#42c9e8", "Estimated close to scheduled time"

def make_route_map(result):
    route = result.get("route_stations") or []

    stations = [
        s for s in route
        if s.get("latitude") is not None
        and s.get("longitude") is not None
    ]

    stations.sort(key=lambda s: s.get("sequence") or 0)

    if not stations:
        return None, 0

    coords = [
        (float(s["latitude"]), float(s["longitude"]))
        for s in stations
    ]

    fmap = folium.Map(
        location=coords[0],
        zoom_start=5,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # Animated complete railway route
    AntPath(
        locations=coords,
        color="#19bfe0",
        weight=5,
        opacity=.9,
        delay=750,
        pulse_color="#ffffff"
    ).add_to(fmap)

    # Selected intermediate station markers
    step = max(1, len(stations) // 18)

    for station in stations[::step]:
        folium.CircleMarker(
            location=[
                float(station["latitude"]),
                float(station["longitude"])
            ],
            radius=3,
            color="#35d8ef",
            fill=True,
            fill_color="#1688c9",
            fill_opacity=.85,
            tooltip=station.get("station_name", "Station")
        ).add_to(fmap)

    # Premium station markers
    current_name = str(result.get("current_station") or "").strip().lower()

    current_position = next(
        (
            station for station in stations
            if str(station.get("station_name") or "").strip().lower()
            == current_name
        ),
        None
    )

    # Show train icon at the current station
    if current_position:
        folium.Marker(
            location=[
                float(current_position["latitude"]),
                float(current_position["longitude"])
            ],
            tooltip="🚆 CURRENT TRAIN LOCATION",
            popup=f"Train currently at {result.get('current_station')}",
            icon=folium.DivIcon(
                html="""
                <div style="
                    font-size:30px;
                    width:48px;
                    height:48px;
                    text-align:center;
                    line-height:44px;
                    background:#102b48;
                    border:2px solid #35d8ef;
                    border-radius:50%;
                    box-shadow:0 3px 14px rgba(0,0,0,.5);
                ">🚆</div>
                """
            )
        ).add_to(fmap)

    start_station = stations[0]
    folium.Marker(
        location=coords[0],
        tooltip="Journey start",
        popup=f"Start: {start_station.get('station_name', 'Starting station')}",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(fmap)

    destination_station = next(
        (
            station for station in reversed(stations)
            if station.get("stop_type") == "DESTINATION"
        ),
        stations[-1]
    )
    folium.Marker(
        location=[
            float(destination_station["latitude"]),
            float(destination_station["longitude"])
        ],
        tooltip="Final destination",
        popup=f"Destination: {result.get('destination') or destination_station.get('station_name', 'Destination')}",
        icon=folium.Icon(color="red", icon="flag")
    ).add_to(fmap)

    fmap.fit_bounds(coords)

    return fmap, len(stations)
# ---------- AI PREDICTION GAUGE ----------

def prediction_gauge(title, value):
    label, color, explanation = delay_label(value)
    shown = value if value is not None else 0

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=shown,
        number={
            "suffix": " min",
            "font": {"size": 27, "color": "#eaf8ff"}
        },
        title={
            "text": title,
            "font": {"size": 13, "color": "#a9c0d4"}
        },
        gauge={
            "axis": {
                "range": [-60, 120],
                "tickcolor": "#8faac1",
                "tickfont": {"color": "#8faac1", "size": 9}
            },
            "bar": {"color": color, "thickness": .22},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [-60, -5], "color": "rgba(56,190,135,.13)"},
                {"range": [-5, 5], "color": "rgba(53,216,239,.12)"},
                {"range": [5, 120], "color": "rgba(255,110,135,.12)"}
            ]
        }
    ))

    fig.update_layout(
        height=245,
        margin=dict(l=18, r=18, t=45, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#eaf8ff"
    )

    return fig, label, color, explanation
# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("<div class='eyebrow'>RAILWAY INTELLIGENCE</div><h2 style='margin:4px 0 0'>🚄 RailSense</h2><p class='muted'>Dynamic Train ETA Platform</p>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("NAVIGATION", ["Overview", "Live Route", "AI Predictions", "Data Quality"], label_visibility="visible")
    st.divider()
    st.markdown("<div class='panel-label'>SYSTEM STATUS</div><div style='margin-top:9px'><span class='pill'>● READY</span></div><p class='muted' style='margin-top:10px'>Live information depends on the connected train-status provider.</p>", unsafe_allow_html=True)
    st.markdown("<div class='muted' style='margin-top:28px'>RAILSENSE · ACADEMIC ML PROJECT</div>", unsafe_allow_html=True)

# ---------- HEADER + SEARCH ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">LIVE OPERATIONS · ETA INTELLIGENCE · MACHINE LEARNING</div>
  <h1>Railway Command Centre</h1>
  <p>Track a train, understand its running condition and explore machine-learning delay estimates.</p>
</div>
""", unsafe_allow_html=True)

with st.container(border=True):
    c1, c2 = st.columns([3, 1])
    with c1:
        with st.form("railsense_search", clear_on_submit=False):
            train_number = st.text_input("TRAIN NUMBER", placeholder="Enter a train number, e.g. 12628", label_visibility="visible")
            submitted = st.form_submit_button("↗  FETCH LIVE INTELLIGENCE", use_container_width=True)
    with c2:
        st.markdown("<div class='panel-label' style='margin-top:10px'>SUPPORTED INPUT</div><div class='panel-value'>Train ID</div><div class='panel-sub'>Indian Railways train number</div>", unsafe_allow_html=True)

if submitted:
    cleaned = train_number.strip()
    if not cleaned:
        st.warning("Enter a train number to continue.")
    elif not cleaned.isdigit():
        st.warning("Train number must contain digits only.")
    else:
        try:
            with st.spinner("Connecting to live train-status provider and calculating ETA intelligence..."):
                st.session_state["railsense_result"] = predict_live_train(cleaned)
                st.session_state["railsense_train_number"] = cleaned
            st.success("Train intelligence refreshed successfully.")
        except Exception as exc:
            st.error(f"Unable to retrieve train information: {exc}")

result = st.session_state.get("railsense_result")
if not result:
    st.info("Enter a train number above and select **Fetch Live Intelligence** to load the command centre.")
    st.markdown("<div class='footer'>RAILSENSE · Dynamic Train ETA Prediction & Delay Intelligence System</div>", unsafe_allow_html=True)
    st.stop()

# ---------- SHARED DATA ----------
train_name = result.get("train_name") or "Train name unavailable"
number = result.get("train_number") or st.session_state.get("railsense_train_number", "—")
current_station = result.get("current_station") or "Unavailable"
next_station = result.get("next_station") or "Unavailable"
destination = result.get("destination") or "Unavailable"
reported = safe_delay(result.get("reported_delay"))
next_prediction = safe_delay(result.get("predicted_next_delay"))
dest_prediction = safe_delay(result.get("predicted_destination_delay"))
data_state = str(result.get("data_state") or "Unknown").upper()

# ---------- PAGE: OVERVIEW ----------
if page == "Overview":
    st.markdown(f"<span class='pill'>● TRAIN {number}</span> &nbsp; <span class='muted'>{train_name}</span>", unsafe_allow_html=True)
    section("Live operations snapshot")

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">CURRENT STATION</div>
        <div class="kpi-value">📍 {current_station}</div>
        <div class="kpi-sub">Provider-reported location</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">NEXT STATION</div>
        <div class="kpi-value">➡️ {next_station}</div>
        <div class="kpi-sub">Upcoming station</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">DESTINATION</div>
        <div class="kpi-value">🏁 {destination}</div>
        <div class="kpi-sub">Final journey destination</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">REPORTED DELAY</div>
        <div class="kpi-value">⏱️ {f"{reported:g} min" if reported is not None else "Unavailable"}</div>
        <div class="kpi-sub">Live provider-reported delay</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    section("Journey intelligence")
    p1,p2,p3 = st.columns(3)
    for col, label, value, sub in [
        (p1,"TRAIN ID",str(number),train_name),
        (p2,"RUNNING CONDITION",delay_label(reported)[0],"Provider-reported running delay"),
        (p3,"DATA FRESHNESS",data_state,"Status supplied by data provider"),
    ]:
        with col:
            st.markdown(f"<div class='panel'><div class='panel-label'>{label}</div><div class='panel-value'>{value}</div><div class='panel-sub'>{sub}</div></div>", unsafe_allow_html=True)

    section("Route at a glance")
    fmap, station_count = make_route_map(result)
    if fmap:
        st_folium(fmap, use_container_width=True, height=430, key="railsense_overview_map", returned_objects=[])
        st.caption(f"{station_count} route stations with coordinates. The current marker reflects the provider-reported location, not continuous GPS tracking.")
    else:
        st.warning("Route coordinates are unavailable for this train.")

    section("AI delay estimates")
    x,y = st.columns(2)
    with x:
        fig, label, color, explanation = prediction_gauge("NEXT STATION · ESTIMATED DELAY", next_prediction)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"<span class='pill'>{label}</span> <span class='muted'>{explanation}</span>", unsafe_allow_html=True)
    with y:
        fig, label, color, explanation = prediction_gauge("FINAL DESTINATION · ESTIMATED DELAY", dest_prediction)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"<span class='pill'>{label}</span> <span class='muted'>{explanation}</span>", unsafe_allow_html=True)

# ---------- PAGE: LIVE ROUTE ----------
elif page == "Live Route":
    section("Live railway route")
    fmap, station_count = make_route_map(result)
    if fmap:
        st_folium(fmap, use_container_width=True, height=570, key="railsense_route_map", returned_objects=[])
        st.caption(f"Route map includes {station_count} stations. Locations and route coordinates are supplied by the connected provider; the current marker is not continuous GPS tracking.")
    else:
        st.warning("Geographical coordinates are unavailable for this journey.")

    section("Station-by-station journey progress")
    stations = result.get("route_stations") or []
    if stations:
        stations = sorted(stations, key=lambda s: s.get("sequence") or 0)
        current_index = next((i for i,s in enumerate(stations) if s.get("station_name") == current_station), 0)
        total = len(stations)
        st.progress(min((current_index + 1) / total, 1.0), text=f"Journey progress · station {current_index + 1} of {total}")
        start, end = max(0,current_index-3), min(total,current_index+6)
        for i in range(start,end):
            station = stations[i]
            if i < current_index:
                marker, state = "🟢", "COMPLETED"
            elif i == current_index:
                marker, state = "🚄", "CURRENT REPORTED STATION"
            else:
                marker, state = "⚪", "UPCOMING"
            left,right = st.columns([1,8])
            left.markdown(f"<div style='font-size:25px;text-align:center'>{marker}</div>", unsafe_allow_html=True)
            right.markdown(f"<div class='panel' style='padding:12px 16px'><b>{station.get('station_name','Unknown station')}</b><div class='muted'>{state} · {station.get('station_code','')}</div></div>", unsafe_allow_html=True)
    else:
        st.info("Station-wise route information is unavailable.")

# ---------- PAGE: AI PREDICTIONS ----------
elif page == "AI Predictions":
    section("Machine-learning delay intelligence")
    st.markdown("<p class='muted'>Predictions estimate arrival delay relative to scheduled time. Positive values indicate lateness; negative values indicate early arrival.</p>", unsafe_allow_html=True)
    left,right = st.columns(2)
    for col,title,value,subtitle in [
        (left,"NEXT STATION",next_prediction,f"Estimated delay at {next_station}"),
        (right,"FINAL DESTINATION",dest_prediction,f"Estimated delay at {destination}"),
    ]:
        with col:
            fig,label,color,explanation = prediction_gauge(title,value)
            st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
            st.markdown(f"<div class='panel'><span class='pill'>{label}</span><div class='panel-sub' style='margin-top:10px'>{subtitle}</div><div class='panel-value' style='font-size:17px'>{explanation}</div></div>",unsafe_allow_html=True)
    section("Prediction interpretation")
    chart = go.Figure()
    chart.add_trace(go.Bar(x=["Next station", "Final destination"], y=[next_prediction or 0, dest_prediction or 0], marker_color=["#35d8ef", "#398bff"], text=[f"{next_prediction:+.1f} min" if next_prediction is not None else "N/A", f"{dest_prediction:+.1f} min" if dest_prediction is not None else "N/A"], textposition="outside"))
    chart.add_hline(y=0,line_dash="dash",line_color="#91a9bf")
    chart.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dcecff", yaxis_title="Estimated delay (minutes)", margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(chart,use_container_width=True,config={"displayModeBar":False})
    st.warning("These are model estimates, not guaranteed arrival times. Actual train movement and delays may change after the latest provider update.")

# ---------- PAGE: DATA QUALITY ----------
elif page == "Data Quality":
    section("Live data quality monitor")
    age = result.get("data_age_seconds")
    updated = result.get("updated_at") or "Unavailable"
    q1,q2,q3 = st.columns(3)
    q1.metric("DATA STATE",data_state)
    q2.metric("DATA AGE",f"{age} sec" if age is not None else "Unavailable")
    q3.metric("LAST UPDATED",updated.replace("T"," ")[:19] if updated != "Unavailable" else updated)
    section("Data source and limitations")
    st.markdown("""
    <div class='panel'>
      <div class='panel-label'>LIVE STATUS SOURCE</div>
      <div class='panel-value' style='font-size:19px'>Connected train-status provider</div>
      <div class='panel-sub'>Live train details, reported station and route coordinates are supplied by the configured third-party provider.</div>
      <hr>
      <div class='panel-label'>MODEL OUTPUT</div>
      <div class='panel-value' style='font-size:19px'>Trained machine-learning models</div>
      <div class='panel-sub'>RailSense uses its trained prediction pipeline to estimate delay at the next station and final destination.</div>
      <hr>
      <div class='panel-label'>IMPORTANT LIMITATION</div>
      <div class='panel-sub'>The map marker represents the provider-reported location. It should not be interpreted as continuous GPS tracking. Predictions are estimates and may differ from actual arrival times.</div>
    </div>
    """,unsafe_allow_html=True)

st.markdown("<div class='footer'>RAILSENSE · DYNAMIC TRAIN ETA PREDICTION & DELAY INTELLIGENCE SYSTEM · ACADEMIC MACHINE-LEARNING PROJECT</div>",unsafe_allow_html=True)
