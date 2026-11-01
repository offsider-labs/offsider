"""
Offsider end-to-end vision pipeline.
Orchestrates SignDetector -> OCR -> rule parsing -> clearance evaluation.
"""

from datetime import datetime, time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.detector import SignDetector
from src.ocr import SignReader
from src.rules import evaluate_parking_permission, parse_text_to_rules
from src.schemas import ClearanceVerdict, ParkingRule, VehicleClass

# Panels whose average OCR confidence is below this are treated as unreadable.
# Not tuned yet: correctly read crops in fixtures/sample_crops mostly score 0.6+.
MIN_OCR_CONF = 0.4

# Banner colours (BGR)
GREEN = (0, 200, 0)
RED = (0, 0, 220)
AMBER = (0, 165, 255)

# Models are slow to load, so each is created on first use and then reused
_detector: Optional[SignDetector] = None
_reader: Optional[SignReader] = None


def get_detector() -> SignDetector:
    global _detector
    if _detector is None:
        _detector = SignDetector(weights_path="models/best.pt", conf_threshold=0.4)
    return _detector


def get_reader() -> SignReader:
    global _reader
    if _reader is None:
        _reader = SignReader()
    return _reader


def process_frame(
    frame: np.ndarray,
    vehicle: VehicleClass,
    arrival_time: Optional[time] = None,
    arrival_day: Optional[str] = None,
) -> Tuple[ClearanceVerdict, np.ndarray]:
    """
    Primary pipeline entry point for app.py.

    Args:
        frame: NumPy image array in BGR order (as from cv2). Convert RGB frames from
            Streamlit/PIL first with cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).
        vehicle: VehicleClass enum (TRUCK, VAN, SEDAN).
        arrival_time: Optional override; defaults to system time.
        arrival_day: Optional override; defaults to current day ('MON', 'TUE').

    Returns:
        verdict: Structured ClearanceVerdict schema. If no sign was found, or a panel
            couldn't be read, allowed is False and warning says to check the sign.
        annotated_frame: Frame with bounding boxes, OCR text and status drawn.
    """
    now = datetime.now()
    eval_time = arrival_time or now.time()
    eval_day = arrival_day or now.strftime("%a").upper()

    annotated_frame = frame.copy()

    detections = get_detector().detect_and_crop(frame)
    ocr_results = get_reader().read(detections)

    # each panel is parsed on its own, so its times and days stay with its own restriction
    rules: List[ParkingRule] = []
    unread_panels = 0
    for (crop_panel, meta), result in zip(detections, ocr_results):
        text = "\n".join(result["lines"])  # parser expects one line per sign line
        panel_rules = parse_text_to_rules(text)
        rules.extend(panel_rules)

        readable = bool(panel_rules) and result["mean_conf"] >= MIN_OCR_CONF
        if not readable:
            unread_panels += 1

        # draw box (green if understood, amber if not) and the first line OCR read
        colour = GREEN if readable else AMBER
        x1, y1, x2, y2 = meta["bbox"]
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            annotated_frame,
            result["lines"][0] if result["lines"] else "(no text)",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
        )

    # evaluate parking permission
    verdict = evaluate_parking_permission(
        rules=rules,
        vehicle=vehicle,
        arrival_time=eval_time,
        arrival_day=eval_day,
    )

    # never say "allowed" when part of the sign wasn't understood
    if verdict.allowed and not detections:
        verdict = ClearanceVerdict(
            allowed=False,
            restriction_active=False,
            reason="No parking sign detected.",
            warning="No sign found. Point the camera at the sign.",
        )
    elif verdict.allowed and unread_panels:
        verdict = ClearanceVerdict(
            allowed=False,
            restriction_active=False,
            reason=f"Couldn't read {unread_panels} of {len(detections)} sign panels.",
            warning="Part of the sign couldn't be read. Check the sign yourself.",
        )

    # add verdict banner on top of the image
    if verdict.warning:
        status_text, banner_color = "CHECK SIGN", AMBER
    elif verdict.allowed:
        status_text, banner_color = "ALLOWED", GREEN
    else:
        status_text, banner_color = "RESTRICTED", RED

    cv2.putText(
        annotated_frame,
        f"[{status_text}] {verdict.reason}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        banner_color,
        2,
        cv2.LINE_AA,
    )

    return verdict, annotated_frame
