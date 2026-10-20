import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

st.set_page_config(
    page_title="Offsider",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Parking Check")

webrtc_streamer(
    key="parking-camera",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    media_stream_constraints={
        "video": {
            # Prefer the phone's rear camera.
            "facingMode": {"ideal": "environment"},
            "width": {"ideal": 640},
            "height": {"ideal": 480},
        },
        "audio": False,
    },
)

st.info("Waiting for sign — parking status unknown")