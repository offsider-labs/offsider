"""Tests for the detector -> OCR -> rules pipeline."""

from datetime import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from src import pipeline
from src.schemas import ClearanceVerdict, VehicleClass

FRAME = np.zeros((400, 300, 3), dtype=np.uint8)


class FakeDetector:
    def __init__(self, n_panels):
        self.n_panels = n_panels

    def detect_and_crop(self, frame):
        crop = np.zeros((100, 80, 3), dtype=np.uint8)
        return [(crop, {"bbox": [10, 10 + 110 * i, 90, 110 + 110 * i], "confidence": 0.9})
                for i in range(self.n_panels)]


class FakeReader:
    def __init__(self, panels, conf=0.9):
        self.panels = panels
        self.conf = conf

    def read(self, crops):
        return [{"bbox": meta["bbox"], "lines": lines, "raw": [], "mean_conf": self.conf}
                for (_, meta), lines in zip(crops, self.panels)]


def run(monkeypatch, panels, vehicle=VehicleClass.SEDAN, at=time(12, 0), day="TUE", conf=0.9):
    """Runs process_frame with fake models whose OCR reads the given lines for each panel."""
    monkeypatch.setattr(pipeline, "get_detector", lambda: FakeDetector(len(panels)))
    monkeypatch.setattr(pipeline, "get_reader", lambda: FakeReader(panels, conf))
    return pipeline.process_frame(FRAME, vehicle, arrival_time=at, arrival_day=day)


def test_no_sign_detected_says_check_sign(monkeypatch):
    verdict, _ = run(monkeypatch, [])
    assert verdict.allowed is False
    assert verdict.warning


def test_unreadable_panel_says_check_sign(monkeypatch):
    # "ZONE" without "LOADING": text was read but no rule could be parsed
    verdict, _ = run(monkeypatch, [["ZONE", "MON-FRI"]])
    assert verdict.allowed is False
    assert verdict.warning
    assert "1 of 1" in verdict.reason


def test_low_ocr_confidence_says_check_sign(monkeypatch):
    verdict, _ = run(monkeypatch, [["LOADING", "ZONE", "8AM-6PM", "MON-FRI"]],
                     at=time(20, 0), conf=0.2)
    assert verdict.allowed is False
    assert verdict.warning


def test_clearway_restricts_during_hours(monkeypatch):
    verdict, _ = run(monkeypatch, [["6AM-10AM", "MON-FRI", "CLEARWAY"]], at=time(8, 0))
    assert verdict.allowed is False
    assert verdict.restriction_active is True
    assert verdict.warning is None


def test_loading_zone_allows_truck(monkeypatch):
    verdict, _ = run(monkeypatch, [["LOADING", "ZONE", "8AM-6PM", "MON-FRI"]],
                     vehicle=VehicleClass.TRUCK, at=time(10, 0))
    assert verdict.allowed is True
    assert verdict.warning is None


def test_panels_are_parsed_separately(monkeypatch):
    # Clearway hours must not be applied to the loading zone panel, or vice versa
    panels = [["LOADING", "ZONE", "8AM-6PM", "MON-FRI"], ["6AM-10AM", "MON-FRI", "CLEARWAY"]]
    verdict, _ = run(monkeypatch, panels, vehicle=VehicleClass.TRUCK, at=time(17, 0))
    assert verdict.allowed is True


def test_annotated_frame_is_a_copy(monkeypatch):
    _, annotated = run(monkeypatch, [["NO", "STOPPING"]])
    assert annotated.shape == FRAME.shape
    assert not FRAME.any()  # input frame isn't drawn on


# End-to-end on real photos (needs YOLO weights, easyocr and the Roboflow v3 download)

ROOT = Path(__file__).parent.parent
PHOTOS = sorted((ROOT / "data" / "sign-detection-full").glob("valid/images/*.jpg"))[:2]


@pytest.mark.skipif(not PHOTOS, reason="no photos in data/sign-detection-full")
@pytest.mark.skipif(not (ROOT / "models" / "best.pt").exists(), reason="no YOLO weights")
@pytest.mark.parametrize("photo", PHOTOS, ids=lambda p: p.name[:8])
def test_process_frame_on_real_photo(photo, monkeypatch):
    monkeypatch.chdir(ROOT)  # weights path in pipeline is relative to the project root
    frame = cv2.imread(str(photo))
    verdict, annotated = pipeline.process_frame(
        frame, VehicleClass.SEDAN, arrival_time=time(12, 0), arrival_day="TUE"
    )
    assert isinstance(verdict, ClearanceVerdict)
    assert annotated.shape == frame.shape
