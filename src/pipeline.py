"""
Offsider end-to-end vision pipeline.
Orchestrates SignDetector -> OCR -> rule parsing -> clearance evaluation.
Edge SDK for dashcam parking sign detection and NSW kerbside compliance.
"""

from datetime import datetime, time
import time as time_module
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.detector import SignDetector
from src.ocr import SignReader
from src.rules import evaluate_parking_permission, parse_text_to_rules
from src.schemas import ClearanceVerdict, ParkingRule, VehicleClass

# Panels whose average OCR confidence is below this are treated as unreadable.
MIN_OCR_CONF = 0.4

# Banner colours (BGR)
GREEN = (0, 200, 0)
RED = (0, 0, 220)
AMBER = (0, 165, 255)

# Models are slow to load, so each is created on first use and then reused
_detector: Optional[SignDetector] = None
_reader: Optional[SignReader] = None


def get_detector(conf_threshold: Optional[float] = None, weights_path: Optional[str] = "models/best.pt") -> SignDetector:
    """Returns cached SignDetector instance, optionally updating confidence threshold."""
    global _detector
    if _detector is None:
        _detector = SignDetector(weights_path=weights_path, conf_threshold=conf_threshold if conf_threshold is not None else 0.4)
    elif conf_threshold is not None:
        _detector.conf_threshold = conf_threshold
    return _detector


def get_reader(gpu: bool = False) -> SignReader:
    """Returns cached SignReader instance."""
    global _reader
    if _reader is None:
        _reader = SignReader(gpu=gpu)
    return _reader


def add_frame_border(img_bgr: np.ndarray, status: str, boxes: Optional[list] = None) -> np.ndarray:
    """Draws bounding boxes and a clean status border for rolling buffer filmstrip frames."""
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    # BGR tuples for OpenCV
    bgr_map = {
        "LOCKED": (94, 197, 34),     # Green
        "REJECTED": (8, 179, 234),   # Amber
        "NO_SIGN": (80, 80, 80),     # Dark grey
        "SKIPPED": (45, 45, 45),     # Muted grey
    }
    col = bgr_map.get(status, (80, 80, 80))

    if boxes:
        for b in boxes:
            if len(b) == 4:
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, max(2, w // 300))

    border_px = max(6, w // 120)
    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), col, border_px)
    return vis


def process_frame(
    frame: np.ndarray,
    vehicle: VehicleClass,
    arrival_time: Optional[time] = None,
    arrival_day: Optional[str] = None,
) -> Tuple[ClearanceVerdict, np.ndarray]:
    """
    Primary pipeline entry point for single-frame processing.

    Args:
        frame: NumPy image array in BGR order (as from cv2). Convert RGB frames from
            Streamlit/PIL first with cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).
        vehicle: VehicleClass enum (TRUCK, VAN, SEDAN, MOTORCYCLE).
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
    verdict.rules = rules

    # never say "allowed" when part of the sign wasn't understood
    if verdict.allowed and not detections:
        verdict = ClearanceVerdict(
            allowed=False,
            restriction_active=False,
            reason="No parking sign detected.",
            warning="No sign found. Point the camera at the sign.",
            rules=[],
        )
    elif verdict.allowed and unread_panels:
        verdict = ClearanceVerdict(
            allowed=False,
            restriction_active=False,
            reason=f"Couldn't read {unread_panels} of {len(detections)} sign panels.",
            warning="Part of the sign couldn't be read. Check the sign yourself.",
            rules=rules,
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


def scan_lookback_buffer(
    video_path: str,
    park_timestamp_sec: float,
    lookback_sec: float = 3.0,
    num_steps: int = 5,
    conf_thresh: float = 0.4,
    ocr_gate: bool = True,
    min_ocr_conf: float = MIN_OCR_CONF,
    vehicle: VehicleClass = VehicleClass.SEDAN,
    arrival_time: Optional[time] = None,
    arrival_day: Optional[str] = None,
) -> dict:
    """
    Dashcam rolling lookback buffer scanner.
    Samples `num_steps` frames backwards from `park_timestamp_sec` (when the vehicle stops).
    Scans each keyframe for parking sign panels, filters with OCR gate, parses NSW rules,
    and returns verified compliance verdict along with the full rolling buffer trace.
    """
    now = datetime.now()
    eval_time = arrival_time or now.time()
    eval_day = arrival_day or now.strftime("%a").upper()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0.0

    end_sec = min(max(0.5, park_timestamp_sec), duration_sec)
    start_sec = max(0.0, end_sec - lookback_sec)
    timestamps = np.linspace(end_sec, start_sec, num=num_steps)

    try:
        detector = get_detector(conf_threshold=conf_thresh)
    except TypeError:
        detector = get_detector()
    reader = get_reader()

    t0 = time_module.perf_counter()
    trace = []
    locked_frame_data = None
    final_verdict = None

    for idx, ts in enumerate(timestamps):
        offset_sec = ts - end_sec
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue

        if locked_frame_data is not None:
            thumb = add_frame_border(frame_bgr, "SKIPPED")
            trace.append({
                "ts": ts,
                "offset": offset_sec,
                "status": "SKIPPED",
                "note": "HALTED (Sign locked)",
                "thumb_rgb": cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB),
            })
            continue

        crops = detector.detect_and_crop(frame_bgr) if (detector and detector.model is not None) else []

        if not crops:
            thumb = add_frame_border(frame_bgr, "NO_SIGN")
            trace.append({
                "ts": ts,
                "offset": offset_sec,
                "status": "NO_SIGN",
                "note": f"0 boxes > {conf_thresh:.0%}",
                "thumb_rgb": cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB),
            })
            continue

        boxes = []
        for _, meta in crops:
            b = meta.get("bbox") or meta.get("box") or meta.get("xyxy")
            if b and len(b) == 4:
                boxes.append(tuple(map(int, b)))

        results = reader.read(crops)

        # Parse rules and verify readable sign tokens
        verified_crops = []
        verified_results = []
        frame_rules: List[ParkingRule] = []

        for crop_tuple, res in zip(crops, results):
            lines = res.get("lines", [])
            text = "\n".join(lines)
            p_rules = parse_text_to_rules(text)
            frame_rules.extend(p_rules)

            # OCR gate checks rule readability and mean confidence threshold
            is_valid = bool(p_rules) and (res.get("mean_conf", 0.0) >= min_ocr_conf)
            if is_valid or not ocr_gate:
                verified_crops.append(crop_tuple)
                verified_results.append(res)

        if ocr_gate and not verified_crops:
            max_det = max((m.get("confidence", 0.0) for _, m in crops), default=0.0)
            thumb = add_frame_border(frame_bgr, "REJECTED", boxes)
            trace.append({
                "ts": ts,
                "offset": offset_sec,
                "status": "REJECTED",
                "note": f"GATE REJECT (YOLO {max_det:.0%}, No valid sign rules)",
                "thumb_rgb": cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB),
            })
            continue

        active_crops = verified_crops if ocr_gate else crops
        active_results = verified_results if ocr_gate else results

        # Evaluate parking permission using repo function
        verdict = evaluate_parking_permission(
            rules=frame_rules,
            vehicle=vehicle,
            arrival_time=eval_time,
            arrival_day=eval_day,
        )
        verdict.rules = frame_rules
        final_verdict = verdict

        max_det = max((m.get("confidence", 0.0) for _, m in active_crops), default=0.0)
        thumb = add_frame_border(frame_bgr, "LOCKED", boxes)
        trace.append({
            "ts": ts,
            "offset": offset_sec,
            "status": "LOCKED",
            "note": f"LOCKED ({len(active_crops)} panel · YOLO {max_det:.0%})",
            "thumb_rgb": cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB),
        })

        locked_frame_data = {
            "frame_bgr": frame_bgr,
            "crops": active_crops,
            "results": active_results,
            "rules": frame_rules,
            "verdict": verdict,
            "locked_ts": ts,
            "offset_sec": offset_sec,
            "step_index": idx + 1,
        }

    latency_ms = (time_module.perf_counter() - t0) * 1000.0
    cap.release()

    return {
        "found": locked_frame_data is not None,
        "verdict": final_verdict,
        "locked": locked_frame_data,
        "trace": trace,
        "latency_ms": latency_ms,
    }
