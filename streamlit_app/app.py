import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Make src/ importable when run as `streamlit run streamlit_app/app.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.schemas import VehicleClass, ClearanceVerdict, RestrictionType, ParkingRule  # noqa: E402
from src.rules import parse_text_to_rules, evaluate_parking_permission  # noqa: E402
from src.pipeline import (  # noqa: E402
    process_frame,
    scan_lookback_buffer,
    get_detector,
    get_reader,
    MIN_OCR_CONF,
)

# ---------------------------------------------------------
# Page Config & Minimalist Aesthetic Theme (Orange & White)
# ---------------------------------------------------------
st.set_page_config(
    page_title="OFFSIDER",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #F4F3EE;
        --card: #FFFFFF;
        --card-muted: #EAE8DF;
        --ink: #161616;
        --ink-secondary: #706F6A;
        --ink-tertiary: #A6A59E;
        --orange: #FF4400;
        --orange-soft: #FFF0EB;
        --green: #1E8E3E;
        --green-soft: #EBF7EE;
        --red: #D93025;
        --red-soft: #FDECEA;
        --amber: #D97706;
        --amber-soft: #FFFBEB;
    }

    html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: var(--bg) !important;
        color: var(--ink) !important;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 960px !important;
    }

    .mono {
        font-family: 'Space Mono', monospace;
    }

    /* Minimalist Top Bar */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 1rem;
        background-color: #FFFFFF;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.02);
        margin-bottom: 1rem;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 9px;
    }
    .brand-dot {
        width: 9px;
        height: 9px;
        background: var(--orange);
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px rgba(255, 68, 0, 0.45);
    }
    .brand-name {
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: -0.03em;
        color: var(--ink);
    }
    .brand-badge {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: var(--orange);
        background: var(--orange-soft);
        padding: 2px 8px;
        border-radius: 12px;
    }
    .top-meta {
        font-family: 'Space Mono', monospace;
        font-size: 0.74rem;
        color: var(--ink-secondary);
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Big Hero Video Player */
    .stVideo > video {
        width: 100% !important;
        max-height: 68vh !important;
        border: 1px solid rgba(0, 0, 0, 0.08) !important;
        border-radius: 18px !important;
        background-color: #000000 !important;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.07), 0 2px 6px rgba(0, 0, 0, 0.02) !important;
    }

    /* Primary Action Button (Brand Orange) */
    div.stButton > button[kind="primary"] {
        background-color: var(--orange) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Space Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.04em !important;
        padding: 0.65rem 1.1rem !important;
        box-shadow: 0 4px 14px rgba(255, 68, 0, 0.28) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #E63D00 !important;
        box-shadow: 0 6px 18px rgba(255, 68, 0, 0.36) !important;
        transform: translateY(-1px) !important;
    }

    /* Segmented Control in Brand Orange */
    [data-testid="stSegmentedControl"] {
        background: #FFFFFF !important;
        border: 1px solid rgba(0, 0, 0, 0.07) !important;
        border-radius: 10px !important;
        padding: 3px !important;
    }
    [data-testid="stSegmentedControl"] button {
        border-radius: 8px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        font-family: 'Space Mono', monospace !important;
        color: var(--ink-secondary) !important;
        border: none !important;
    }
    [data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background-color: var(--orange) !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 6px rgba(255, 68, 0, 0.28) !important;
    }

    /* Slider Accents */
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[role="slider"] {
        background-color: var(--orange) !important;
        border: 2px solid #FFFFFF !important;
        box-shadow: 0 2px 6px rgba(255, 68, 0, 0.4) !important;
    }
    div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:first-child {
        background: var(--orange) !important;
    }

    /* Minimalist Output Display Banner Below Video */
    .verdict-banner {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #FFFFFF;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-left: 5px solid var(--orange);
        border-radius: 16px;
        padding: 1.15rem 1.4rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.03);
        margin-top: 1rem;
        margin-bottom: 0.9rem;
        gap: 1.25rem;
    }
    .verdict-banner.go { border-left-color: var(--green); }
    .verdict-banner.stop { border-left-color: var(--red); }
    .verdict-banner.warn { border-left-color: var(--amber); }
    .verdict-banner.standby { border-left-color: var(--orange); }

    .verdict-left {
        display: flex;
        flex-direction: column;
        gap: 3px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        width: fit-content;
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        padding: 3px 9px;
        border-radius: 14px;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .pill-go { background: var(--green-soft); color: var(--green); border: 1px solid rgba(30, 142, 62, 0.2); }
    .pill-stop { background: var(--red-soft); color: var(--red); border: 1px solid rgba(217, 48, 37, 0.2); }
    .pill-warn { background: var(--amber-soft); color: var(--amber); border: 1px solid rgba(217, 119, 6, 0.2); }
    .pill-standby { background: var(--orange-soft); color: var(--orange); border: 1px solid rgba(255, 68, 0, 0.2); }

    .verdict-headline {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--ink);
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .verdict-subtext {
        font-size: 0.86rem;
        color: var(--ink-secondary);
        line-height: 1.35;
    }

    .verdict-metrics {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-shrink: 0;
    }
    .metric-chip {
        background: var(--bg);
        border: 1px solid rgba(0, 0, 0, 0.04);
        border-radius: 10px;
        padding: 0.5rem 0.8rem;
        text-align: right;
        min-width: 85px;
    }
    .metric-label {
        display: block;
        font-family: 'Space Mono', monospace;
        font-size: 0.62rem;
        font-weight: 600;
        color: var(--ink-secondary);
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-val {
        display: block;
        font-family: 'Space Mono', monospace;
        font-size: 0.92rem;
        font-weight: 700;
        color: var(--ink);
        margin-top: 2px;
    }

    /* Minimalist Sign Crop Strip */
    .crop-card {
        background: #FFFFFF;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.02);
        margin-bottom: 0.9rem;
    }
    .crop-meta-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 3px;
    }
    .crop-tag {
        font-family: 'Space Mono', monospace;
        font-size: 0.66rem;
        font-weight: 700;
        color: var(--orange);
        background: var(--orange-soft);
        padding: 2px 7px;
        border-radius: 6px;
    }
    .crop-tokens {
        font-family: 'Space Mono', monospace;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--ink);
    }

    /* Minimal Filmstrip Below */
    .film-meta {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        padding: 0.4rem 0.5rem;
        background-color: #FFFFFF;
        border: 1px solid rgba(0, 0, 0, 0.07);
        border-radius: 8px;
        margin-top: -6px;
        text-align: center;
    }
    .film-locked { background-color: var(--green-soft); border-color: rgba(30, 142, 62, 0.25); color: var(--green); }
    .film-rejected { background-color: var(--amber-soft); border-color: rgba(217, 119, 6, 0.25); color: var(--amber); }
    .film-empty { background-color: #FFFFFF; border-color: rgba(0, 0, 0, 0.07); color: var(--ink-secondary); }

    /* Clean Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid rgba(0, 0, 0, 0.06) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }
    [data-testid="stSidebar"] .stMarkdown h4 {
        color: var(--orange) !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.06em !important;
        font-family: 'Space Mono', monospace !important;
        margin-top: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Sidebar: Technical Controls (Hidden / Folded by Default)
# ---------------------------------------------------------
VEHICLE_OPTIONS = {
    "Commercial Heavy Truck (GVM > 4.5t)": VehicleClass.TRUCK,
    "Delivery Van / Light Commercial": VehicleClass.VAN,
    "Standard Passenger Vehicle (Sedan)": VehicleClass.SEDAN,
    "Motorcycle": VehicleClass.MOTORCYCLE,
}

with st.sidebar:
    st.markdown("#### EDGE PIPELINE PARAMETERS")
    selected_vehicle_label = st.selectbox(
        "Vehicle Class",
        list(VEHICLE_OPTIONS.keys()),
        index=0,
    )
    selected_vehicle = VEHICLE_OPTIONS[selected_vehicle_label]

    conf_thresh = st.slider("YOLO Confidence Gate", 0.25, 0.90, 0.68, 0.02)
    lookback_sec = st.slider("Buffer Window (sec)", 2.0, 6.0, 3.0, 0.5)
    ocr_gate = st.toggle("OCR Verification Gate", value=True)
    min_ocr_conf = st.slider("Min OCR Confidence", 0.20, 0.80, float(MIN_OCR_CONF), 0.05)

    st.markdown("#### EVALUATION CONTEXT")
    use_system_clock = st.checkbox("Live Clock & Current Day", value=True)
    if use_system_clock:
        now = datetime.now()
        eval_time = now.time()
        eval_day = now.strftime("%a").upper()
    else:
        col_d, col_t = st.columns(2)
        with col_d:
            eval_day = st.selectbox("Day", ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"], index=0)
        with col_t:
            eval_time = st.time_input("Time", value=datetime.strptime("10:00", "%H:%M").time())


# ---------------------------------------------------------
# Top Navigation Bar
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="top-nav">
        <div class="brand">
            <span class="brand-name">OFFSIDER DEMO</span>
        </div>
        <div class="top-meta">
            <span>v0.3.0</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Selector Row: Scenario Clip & Vehicle Quick Toggle
# ---------------------------------------------------------
demo_dir = ROOT / "data" / "demo_clips"
demo_dir.mkdir(parents=True, exist_ok=True)
preset_files = sorted(
    [p for p in demo_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".m4v"}]
)

ctrl_left, ctrl_right = st.columns([1.6, 1.4], vertical_alignment="center")

with ctrl_left:
    options = [p.name for p in preset_files]
    selected_source = st.segmented_control(
        "Scenario",
        options=options,
        selection_mode="single",
        key="video_selector",
        default=options[0] if options else None,
        label_visibility="collapsed",
    )

with ctrl_right:
    quick_map = {
        "Commercial Truck": VehicleClass.TRUCK,
        "Delivery Van": VehicleClass.VAN,
        "Passenger Car": VehicleClass.SEDAN,
    }
    default_idx = 0 if selected_vehicle == VehicleClass.TRUCK else (1 if selected_vehicle == VehicleClass.VAN else 2)
    quick_choice = st.radio(
        "Vehicle",
        list(quick_map.keys()),
        index=default_idx,
        horizontal=True,
        label_visibility="collapsed",
    )
    selected_vehicle = quick_map[quick_choice]

# Resolve Active Video Path
active_video_path = None
if selected_source:
    chosen_path = str(demo_dir / selected_source)
    if st.session_state.get("active_preset") != chosen_path:
        st.session_state.active_preset = chosen_path
        st.session_state.scan_output = None
    active_video_path = chosen_path
elif preset_files:
    active_video_path = str(preset_files[0])


# ---------------------------------------------------------
# Big Hero Video Screen & Rewind Trigger
# ---------------------------------------------------------
if not active_video_path or not Path(active_video_path).exists():
    st.info("No video clip found in `data/demo_clips/`.")
else:
    cap_meta = cv2.VideoCapture(active_video_path)
    v_fps = cap_meta.get(cv2.CAP_PROP_FPS) or 30.0
    v_frames = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
    v_duration = max(1.0, round(v_frames / v_fps, 1))
    cap_meta.release()

    # Big hero video display
    st.video(active_video_path, autoplay=True, muted=True, loop=True)

    # Scrubber slider and primary action button directly beneath video
    col_scrub, col_btn = st.columns([3.4, 1.2], vertical_alignment="center")
    with col_scrub:
        park_ts = st.slider(
            "Timestamp",
            min_value=0.5,
            max_value=float(v_duration),
            value=float(v_duration),
            step=0.2,
            format="%.1fs",
            label_visibility="collapsed",
        )
    with col_btn:
        trigger_scan = st.button(
            "PARK",
            type="primary",
            use_container_width=True,
        )

    if trigger_scan:
        st.session_state.scan_output = scan_lookback_buffer(
            video_path=active_video_path,
            park_timestamp_sec=park_ts,
            lookback_sec=lookback_sec,
            num_steps=5,
            conf_thresh=conf_thresh,
            ocr_gate=ocr_gate,
            min_ocr_conf=min_ocr_conf,
            vehicle=selected_vehicle,
            arrival_time=eval_time,
            arrival_day=eval_day,
        )

    out = st.session_state.get("scan_output")

    # ---------------------------------------------------------
    # Output Display Banner Below Video
    # ---------------------------------------------------------
    if out is None:
        st.markdown(
            """
            <div class="verdict-banner standby">
                <div class="verdict-left">
                    <div class="verdict-headline">Parking in progress</div>
                    <div class="verdict-subtext">Click <strong>PARK</strong> to verify parking clearance.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif not out["found"]:
        st.markdown(
            f"""
            <div class="verdict-banner warn">
                <div class="verdict-left">
                    <div class="verdict-headline">CHECK MANUALLY</div>
                    <div class="verdict-subtext">Human verification required.</div>
                </div>
                <div class="verdict-metrics">
                    <div class="metric-chip">
                        <span class="metric-label">LATENCY</span>
                        <span class="metric-val">{out['latency_ms']:.0f}ms</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        verdict: ClearanceVerdict = out["verdict"]
        locked = out["locked"]
        crops = locked["crops"]
        results = locked["results"]

        if verdict.warning:
            banner_class = "warn"
            pill_class = "pill-warn"
            pill_text = "CHECK MANUALLY"
            headline = "Human Verification Required"
        elif not verdict.allowed:
            banner_class = "stop"
            pill_class = "pill-stop"
            pill_text = "PROHIBITED"
            headline = "Parking Prohibited"
        else:
            banner_class = "go"
            pill_class = "pill-go"
            pill_text = "PERMITTED"
            headline = "Parking Permitted"

        if verdict.max_stay_minutes is not None:
            if verdict.max_stay_minutes == 0:
                dwell_str = "0 MIN"
            elif verdict.max_stay_minutes < 60:
                dwell_str = f"{verdict.max_stay_minutes} MIN"
            else:
                dwell_str = f"{verdict.max_stay_minutes // 60}H"
        elif not verdict.allowed:
            dwell_str = "0 MIN"
        else:
            dwell_str = "FREE"

        st.markdown(
            f"""
            <div class="verdict-banner {banner_class}">
                <div class="verdict-left">
                    <span class="status-pill {pill_class}">{pill_text}</span>
                    <div class="verdict-headline">{headline}</div>
                    <div class="verdict-subtext">{verdict.reason}</div>
                </div>
                <div class="verdict-metrics">
                    <div class="metric-chip">
                        <span class="metric-label">MAX STAY</span>
                        <span class="metric-val">{dwell_str}</span>
                    </div>
                    <div class="metric-chip">
                        <span class="metric-label">LOCKED</span>
                        <span class="metric-val">T{locked['offset_sec']:+.1f}s</span>
                    </div>
                    <div class="metric-chip">
                        <span class="metric-label">LATENCY</span>
                        <span class="metric-val">{out['latency_ms']:.0f}ms</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Compact Sign Panel Strip
        if crops and results:
            for idx, ((crop_bgr, meta), res) in enumerate(zip(crops, results), start=1):
                det_c = meta.get("confidence", 0.0)
                ocr_c = res.get("mean_conf", 0.0)
                tokens_text = " &bull; ".join(res.get("lines", [])) or "No text"

                c_img, c_info = st.columns([0.18, 0.82], vertical_alignment="center")
                with c_img:
                    st.image(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                with c_info:
                    st.markdown(
                        f"""
                        <div style="display:flex; flex-direction:column; gap:2px;">
                            <div class="crop-meta-row">
                                <span class="crop-tag">PANEL 0{idx}</span>
                                <span style="font-family:'Space Mono'; font-size:0.72rem; color:#706F6A;">YOLO: <strong style="color:#161616;">{det_c:.0%}</strong> &bull; OCR: <strong style="color:#161616;">{ocr_c:.0%}</strong></span>
                            </div>
                            <div class="crop-tokens">{tokens_text}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # ---------------------------------------------------------
    # 5-Frame Backward Lookback Filmstrip
    # ---------------------------------------------------------
    if out is not None and out.get("trace"):
        st.markdown(
            "<div class='mono' style='font-size:0.72rem; font-weight:700; color:#706F6A; margin: 1.2rem 0 0.5rem 0; letter-spacing:0.04em; text-transform:uppercase;'>"
            "ROLLING LOOKBACK TRACE"
            "</div>",
            unsafe_allow_html=True,
        )

        cols = st.columns(len(out["trace"]), gap="small")
        for col, item in zip(cols, out["trace"]):
            with col:
                st.image(item["thumb_rgb"], use_container_width=True)
                status_class = {
                    "LOCKED": "film-locked",
                    "REJECTED": "film-rejected",
                    "NO_SIGN": "film-empty",
                    "SKIPPED": "film-empty",
                }[item["status"]]
                st.markdown(
                    f"""
                    <div class="film-meta {status_class}">
                        <strong>T{item['offset']:+.1f}s</strong> &bull; {item['status']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
