"""
Offsider end-to-end vision pipeline.
Orchestrates SignDetector -> OCR -> rule parsing -> clearance evaluation.
"""

from datetime import datetime, time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.detector import SignDetector
from src.rules import evaluate_parking_permission, parse_ocr_to_rules
from src.schemas import ClearanceVerdict, ParkingRule, VehicleClass

detector = SignDetector(weights_path="models/best.pt", conf_threshold=0.4)

def process_frame(
    frame: np.ndarray,
    vehicle: VehicleClass,
    arrival_time: Optional[time] = None,
    arrival_day: Optional[str] = None,
) -> Tuple[ClearanceVerdict, np.ndarray]:
    """
    Primary pipeline entry point for app.py.

    Args:
        frame: NumPy image array (RGB from Streamlit/PIL, or BGR from cv2).
        vehicle: VehicleClass enum (TRUCK, VAN, SEDAN).
        arrival_time: Optional override; defaults to system time.
        arrival_day: Optional override; defaults to current day ('MON', 'TUE').

    Returns:
        verdict: Structured ClearanceVerdict schema.
        annotated_frame: Frame with bounding boxes and status labels drawn.
    """
    now = datetime.now()
    eval_time = arrival_time or now.time()
    eval_day = arrival_day or now.strftime("%a").upper()

    annotated_frame = frame.copy()

    detections = detector.detect_and_crop(frame)

    # extract text from each panel detected
    extracted_texts: List[str] = []
    for crop_panel, meta in detections:
        # TODO: Replace with real OCR call:
        # text = ocr_engine.extract_text(crop_panel)
        # extracted_texts.append(text)

        # Draw bounding box on annotated_frame
        x1, y1, x2, y2 = meta["bbox"]
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_frame,
            f"Sign {meta.get('confidence', 1.0):.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # parse extracted text to rules
    rules = parse_ocr_to_rules(extracted_texts)

    # evaluate parking permission
    verdict = evaluate_parking_permission(
        rules=rules,
        vehicle=vehicle,
        arrival_time=eval_time,
        arrival_day=eval_day,
    )

    # add verdict banner on top of the image
    status_text = "ALLOWED" if verdict.allowed else "RESTRICTED"
    banner_color = (0, 200, 0) if verdict.allowed else (220, 0, 0)  # Green / Red

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