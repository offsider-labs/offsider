import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Make src/ importable when run as `streamlit run streamlit_app/app.py`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detector import SignDetector  # noqa: E402
from src.ocr import SignReader  # noqa: E402

st.set_page_config(
    page_title="Offsider",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# Models are slow to load, so keep one of each for the whole server
@st.cache_resource
def load_detector() -> SignDetector:
    # Real panels score 0.9+; background clutter (trees, roofs) has scored around 0.4
    return SignDetector(weights_path=str(ROOT / "models" / "best.pt"), conf_threshold=0.6)


@st.cache_resource
def load_reader() -> SignReader:
    return SignReader()


st.title("Parking Check")

live_tab, ocr_tab = st.tabs(["Live camera", "Test OCR"])

with live_tab:
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

with ocr_tab:
    photo = st.file_uploader(
        "Upload or take a photo of a parking sign",
        type=["jpg", "jpeg", "png"],
    )

    if photo is not None:
        img = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)

        detector = load_detector()
        crops = detector.detect_and_crop(img) if detector.model is not None else []

        if not crops:
            # No weights, or YOLO found nothing: still useful to see what OCR makes of the photo
            st.caption("No sign detected (or no YOLO weights in models/), so reading the whole photo.")
            h, w = img.shape[:2]
            crops = [(img, {"confidence": 1.0, "bbox": [0, 0, w, h], "whole_photo": True})]

        with st.spinner("Reading text..."):
            results = load_reader().read(crops)

        for i, ((crop, meta), result) in enumerate(zip(crops, results), start=1):
            st.subheader(f"Sign {i}")
            image_col, text_col = st.columns(2)

            image_col.image(crop, channels="BGR")

            if result["lines"]:
                text_col.metric("OCR confidence", f"{result['mean_conf']:.0%}")
            else:
                text_col.metric("OCR confidence", "n/a")
            if meta.get("whole_photo"):
                text_col.metric("Detector confidence", "n/a")
            else:
                text_col.metric("Detector confidence", f"{meta['confidence']:.0%}")
            text_col.code(
                "\n".join(result["lines"]) or "(no text found, possibly a symbol-only panel)",
                language=None,
            )

            with st.expander("Raw EasyOCR output"):
                st.json(result["raw"])
