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
# Page Config & Industrial Utilitarian Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="OFFSIDER // Kerbside Compliance",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #0d0d0d;
        color: #e5e5e5;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    .mono {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Utilitarian Top Bar */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.65rem 1rem;
        background-color: #141414;
        border: 1px solid #262626;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .top-bar-left {
        display: flex;
        align-items: center;
        gap: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        color: #fafafa;
    }
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #22c55e;
        border-radius: 1px;
    }
    .top-bar-right {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #8c8c8c;
    }

    /* Industrial Verdict Banner */
    .verdict-box {
        padding: 1.25rem;
        background-color: #141414;
        border: 1px solid #262626;
        border-left-width: 5px;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .verdict-go { border-left-color: #22c55e; }
    .verdict-stop { border-left-color: #ef4444; }
    .verdict-warn { border-left-color: #eab308; }

    .verdict-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 2px 6px;
        border-radius: 2px;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .tag-go { background-color: rgba(34, 197, 94, 0.15); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
    .tag-stop { background-color: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .tag-warn { background-color: rgba(234, 179, 8, 0.15); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.3); }

    .verdict-heading {
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin: 0 0 0.35rem 0;
    }
    .verdict-detail {
        font-size: 0.9rem;
        color: #a3a3a3;
        margin: 0;
        line-height: 1.4;
    }

    /* Spec Table */
    .spec-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        margin-top: 1rem;
        padding-top: 0.85rem;
        border-top: 1px solid #262626;
    }
    .spec-cell {
        background-color: #0d0d0d;
        border: 1px solid #1f1f1f;
        padding: 0.5rem 0.65rem;
        border-radius: 3px;
    }
    .spec-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #737373;
        text-transform: uppercase;
    }
    .spec-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.92rem;
        font-weight: 600;
        color: #e5e5e5;
        margin-top: 2px;
    }

    /* Sharp Primary Action Button */
    div.stButton > button[kind="primary"] {
        background-color: #f5f5f5 !important;
        color: #0d0d0d !important;
        border: 1px solid #ffffff !important;
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.04em !important;
        padding: 0.65rem 1.2rem !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #d4d4d4 !important;
    }

    /* Video & Image Borders */
    .stVideo > video {
        border: 1px solid #262626;
        border-radius: 4px !important;
        background-color: #000;
        max-height: 56vh;
    }

    /* Filmstrip Caption Cards */
    .film-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        padding: 0.4rem 0.5rem;
        background-color: #141414;
        border: 1px solid #262626;
        border-radius: 3px;
        margin-top: -8px;
    }
    .film-locked { border-color: #22c55e; color: #4ade80; }
    .film-rejected { border-color: #eab308; color: #facc15; }
    .film-empty { border-color: #262626; color: #737373; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Sidebar: SDK & Pipeline Parameters
# ---------------------------------------------------------
VEHICLE_OPTIONS = {
    "Commercial Heavy Truck (GVM > 4.5t)": VehicleClass.TRUCK,
    "Delivery Van / Light Commercial": VehicleClass.VAN,
    "Standard Passenger Vehicle (Sedan)": VehicleClass.SEDAN,
    "Motorcycle": VehicleClass.MOTORCYCLE,
}

with st.sidebar:
    st.markdown("#### DASHCAM SDK CONTROLS")
    selected_vehicle_label = st.selectbox(
        "Fleet Vehicle Class (VehicleClass)",
        list(VEHICLE_OPTIONS.keys()),
        index=0,
    )
    selected_vehicle = VEHICLE_OPTIONS[selected_vehicle_label]

    conf_thresh = st.slider(
        "YOLOv8n Confidence Gate",
        min_value=0.25,
        max_value=0.90,
        value=0.68,
        step=0.02,
        help="Detection threshold for SignDetector.",
    )
    lookback_sec = st.slider("Rolling Buffer Window (sec)", 2.0, 6.0, 3.0, 0.5)
    ocr_gate = st.toggle("Two-Stage OCR Verification Gate", value=True)
    min_ocr_conf = st.slider("Min OCR Confidence", 0.20, 0.80, float(MIN_OCR_CONF), 0.05)

    st.markdown("#### NSW LAW VERIFICATION CONTEXT")
    use_system_clock = st.checkbox("Use Live Clock & Current Day", value=True)
    if use_system_clock:
        now = datetime.now()
        eval_time = now.time()
        eval_day = now.strftime("%a").upper()
    else:
        col_day, col_time = st.columns(2)
        with col_day:
            eval_day = st.selectbox("Arrival Day", ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"], index=0)
        with col_time:
            time_val = st.time_input("Arrival Time", value=datetime.strptime("10:00", "%H:%M").time())
            eval_time = time_val


# ---------------------------------------------------------
# Utilitarian Header Bar
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="top-bar">
        <div class="top-bar-left">
            <span class="status-indicator"></span>
            OFFSIDER // KERBSIDE SIGN TRIAGE &amp; COMPLIANCE SDK
        </div>
        <div class="top-bar-right">
            MODEL: YOLOv8n v0.3.0 &nbsp;|&nbsp;
            OCR_GATE: {"ENABLED" if ocr_gate else "BYPASSED"} &nbsp;|&nbsp;
            PROFILE: {selected_vehicle.value.upper()} &nbsp;|&nbsp;
            EVAL: {eval_day} {eval_time.strftime('%H:%M')}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Scenario Switcher & Approach Clip Selector
# ---------------------------------------------------------
demo_dir = ROOT / "data" / "demo_clips"
demo_dir.mkdir(parents=True, exist_ok=True)
preset_files = sorted(
    [p for p in demo_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".m4v"}]
)

ctrl_left, ctrl_right = st.columns([2.2, 1], vertical_alignment="bottom")

with ctrl_left:
    options = [p.name for p in preset_files] + ["+ Upload Custom Clip..."]
    selected_source = st.selectbox(
        "SELECT KERBSIDE SCENARIO (`data/demo_clips/`)",
        options=options,
        index=0,
    )

with ctrl_right:
    quick_map = {
        "Commercial Truck": VehicleClass.TRUCK,
        "Delivery Van": VehicleClass.VAN,
        "Passenger Car": VehicleClass.SEDAN,
    }
    quick_choice = st.radio(
        "ACTIVE VEHICLE CLASS",
        list(quick_map.keys()),
        index=0 if selected_vehicle == VehicleClass.TRUCK else (1 if selected_vehicle == VehicleClass.VAN else 2),
        horizontal=True,
    )
    selected_vehicle = quick_map[quick_choice]

active_video_path = None

if selected_source == "+ Upload Custom Clip...":
    uploaded = st.file_uploader(
        "Drop an MP4 or MOV street approach clip",
        type=["mp4", "mov", "m4v"],
    )
    if uploaded is not None:
        if st.session_state.get("uploaded_name") != uploaded.name:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            tfile.flush()
            st.session_state.custom_video_path = tfile.name
            st.session_state.uploaded_name = uploaded.name
            st.session_state.scan_output = None
        active_video_path = st.session_state.get("custom_video_path")
else:
    chosen_path = str(demo_dir / selected_source)
    if st.session_state.get("active_preset") != chosen_path:
        st.session_state.active_preset = chosen_path
        st.session_state.scan_output = None
    active_video_path = chosen_path


# ---------------------------------------------------------
# Main Stage: Left = Dashcam Video & Trigger | Right = Verdict & Inspector
# ---------------------------------------------------------
if not active_video_path:
    st.info(
        "Place video clips into `data/demo_clips/` (e.g. `.mp4`, `.mov`) or upload a clip above to test dashcam edge verification."
    )
else:
    cap_meta = cv2.VideoCapture(active_video_path)
    v_fps = cap_meta.get(cv2.CAP_PROP_FPS) or 30.0
    v_frames = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
    v_duration = max(1.0, round(v_frames / v_fps, 1))
    cap_meta.release()

    left_col, right_col = st.columns([1.25, 1], gap="large")

    with left_col:
        st.video(active_video_path, autoplay=True, muted=True)

        bar_slider, bar_btn = st.columns([2.2, 1], vertical_alignment="bottom")
        with bar_slider:
            park_ts = st.slider(
                f"VEHICLE STOP TIMESTAMP (Rewinds {lookback_sec:.1f}s window prior to stop)",
                min_value=0.5,
                max_value=float(v_duration),
                value=float(v_duration),
                step=0.2,
                format="%.1fs",
            )
        with bar_btn:
            trigger_scan = st.button(
                "[ PARK // SCAN BUFFER ]",
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
    # Right Column: SDK Verdict & Pipeline Inspector
    # ---------------------------------------------------------
    with right_col:
        if out is None:
            st.markdown(
                """
                <div class="verdict-box" style="border-left-color: #525252;">
                    <span class="verdict-tag" style="background:#262626; color:#a3a3a3;">STANDBY // BUFFER ACTIVE</span>
                    <h3 class="verdict-heading">AWAITING VEHICLE STOP TRIGGER</h3>
                    <p class="verdict-detail">
                        Rolling lookback buffer active. Click <strong>[ PARK // SCAN BUFFER ]</strong> once the vehicle
                        comes to a halt to scan the preceding rolling approach window with the edge SDK.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif not out["found"]:
            st.markdown(
                f"""
                <div class="verdict-box verdict-warn">
                    <span class="verdict-tag tag-warn">NO VERIFIED SIGN IN WINDOW</span>
                    <h3 class="verdict-heading">NO KERBSIDE PANEL LOCKED</h3>
                    <p class="verdict-detail">
                        Evaluated 5 keyframes across the {lookback_sec:.1f}s window prior to <strong>{park_ts:.1f}s</strong>.
                        Zero candidate bounding boxes passed both YOLOv8 ({conf_thresh:.0%}) and the OCR verification gate.
                    </p>
                    <div class="spec-grid">
                        <div class="spec-cell">
                            <div class="spec-label">WINDOW</div>
                            <div class="spec-val">{max(0.0, park_ts - lookback_sec):.1f}s – {park_ts:.1f}s</div>
                        </div>
                        <div class="spec-cell">
                            <div class="spec-label">FRAMES TESTED</div>
                            <div class="spec-val">5 KEYFRAMES</div>
                        </div>
                        <div class="spec-cell">
                            <div class="spec-label">LATENCY</div>
                            <div class="spec-val">{out["latency_ms"]:.0f} ms</div>
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
            rules = locked.get("rules", [])

            if verdict.warning:
                v_class = "verdict-warn"
                t_class = "tag-warn"
                v_tag = "MANUAL CHECK REQUIRED"
                v_title = "SIGN LOCALIZED // CHECK SIGN"
            elif not verdict.allowed:
                v_class = "verdict-stop"
                t_class = "tag-stop"
                v_tag = "PROHIBITED // ACTIVE RESTRICTION"
                v_title = "DO NOT STOP / PARK"
            else:
                v_class = "verdict-go"
                t_class = "tag-go"
                v_tag = f"PERMITTED // {selected_vehicle.value.upper()} ELIGIBLE"
                v_title = "PARKING PERMITTED"

            if verdict.max_stay_minutes is not None:
                if verdict.max_stay_minutes == 0:
                    dwell_str = "0 MIN (IMMEDIATE INFRINGEMENT)"
                elif verdict.max_stay_minutes < 60:
                    dwell_str = f"{verdict.max_stay_minutes} MIN MAX"
                else:
                    hrs = verdict.max_stay_minutes // 60
                    dwell_str = f"{hrs} HOUR{'S' if hrs > 1 else ''} MAX"
            elif not verdict.allowed:
                dwell_str = "0 MIN (PROHIBITED)"
            else:
                dwell_str = "UNRESTRICTED"

            st.markdown(
                f"""
                <div class="verdict-box {v_class}">
                    <span class="verdict-tag {t_class}">{v_tag}</span>
                    <h3 class="verdict-heading">{v_title}</h3>
                    <p class="verdict-detail">{verdict.reason}</p>
                    <div class="spec-grid">
                        <div class="spec-cell">
                            <div class="spec-label">MAX DWELL TIME</div>
                            <div class="spec-val">{dwell_str}</div>
                        </div>
                        <div class="spec-cell">
                            <div class="spec-label">LOCKED FRAME</div>
                            <div class="spec-val">{locked["locked_ts"]:.1f}s ({locked["offset_sec"]:+.1f}s)</div>
                        </div>
                        <div class="spec-cell">
                            <div class="spec-label">PIPELINE LATENCY</div>
                            <div class="spec-val">{out["latency_ms"]:.0f} ms</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Parsed Regulatory Rules Breakdown (From SDK)
            if rules:
                with st.expander("PARSED NSW REGULATORY RULES (SDK OUTPUT)", expanded=False):
                    for r_idx, r in enumerate(rules, start=1):
                        days_str = "-".join([r.days[0], r.days[-1]]) if len(r.days) > 1 else (r.days[0] if r.days else "ALL")
                        st.markdown(
                            f"- **Rule #{r_idx}:** `{r.restriction_type.value}` | "
                            f"Window: `{r.start_time.strftime('%H:%M')} – {r.end_time.strftime('%H:%M')}` | "
                            f"Days: `{days_str}` | "
                            f"Commercial Only: `{r.commercial_only}` | "
                            f"Stay Limit: `{r.max_duration_minutes or 'None'} min`"
                        )

            # Isolated Sign Crops + Raw OCR Telemetry
            st.markdown("###### EXTRACTED SIGN PANELS & OCR TOKENS")
            for idx, ((crop_bgr, meta), res) in enumerate(zip(crops, results), start=1):
                with st.container(border=True):
                    c_img, c_meta = st.columns([1, 1.6], vertical_alignment="center")
                    c_img.image(
                        cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB),
                        use_container_width=True,
                    )
                    with c_meta:
                        det_c = meta.get("confidence", 0.0)
                        ocr_c = res.get("mean_conf", 0.0)
                        st.markdown(
                            f"<span class='mono' style='font-size:0.78rem; color:#a3a3a3;'>"
                            f"PANEL_0{idx} // YOLO_CONF: <strong>{det_c:.0%}</strong> // "
                            f"OCR_CONF: <strong>{ocr_c:.0%}</strong></span>",
                            unsafe_allow_html=True,
                        )
                        raw_tokens = "\n".join(res.get("lines", [])) or "[NO_READABLE_TOKENS]"
                        st.code(raw_tokens, language=None)

    # ---------------------------------------------------------
    # Full-Width Bottom Section: 5-Frame Backward Lookback Filmstrip
    # ---------------------------------------------------------
    if out is not None and out.get("trace"):
        st.divider()
        st.markdown(
            "<div class='mono' style='font-size:0.8rem; color:#a3a3a3; margin-bottom:0.5rem;'>"
            "BACKWARD ROLLING BUFFER TRACE (MOST RECENT FRAME AT LEFT &rarr; REWINDING APPROACH)"
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
                        <div><strong>T{item['offset']:+.1f}s</strong> ({item['ts']:.1f}s)</div>
                        <div style="margin-top:2px;">{item['note']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )