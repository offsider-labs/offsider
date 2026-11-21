"""Tests for the detector -> OCR -> rules pipeline."""

from datetime import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from eval.score_pipeline import load_scenarios, score_scenario
from src import pipeline
from src.schemas import VehicleClass

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


def test_unreadable_hours_do_not_become_all_day_rule(monkeypatch):
    verdict, _ = run(monkeypatch, [["2P", "8-6", "MON-FRI"]], at=time(7))
    assert verdict.warning
    assert verdict.reason == "Couldn't read the operating hours."
    assert not verdict.restriction_active
    assert verdict.max_stay_minutes is None
    assert verdict.rules == []


def test_unreadable_day_range_requires_checking_the_sign(monkeypatch):
    verdict, _ = run(monkeypatch, [["2P", "8AM-6PM", "MON-F?"]])

    assert verdict.warning
    assert not verdict.allowed
    assert verdict.rules == []


@pytest.mark.parametrize(
    "extra_panel, expected_allowed, expected_warning",
    [
        (["SDOWE", "ORD STREE"], True, False),
        (["BUS", "ZONE", "ORD STREE"], False, False),
        ([], False, True),
    ],
)
def test_wide_street_name_panel_does_not_override_parking_rule(
    monkeypatch, extra_panel, expected_allowed, expected_warning
):
    detector = FakeDetector(2)
    detections = detector.detect_and_crop(FRAME)
    detections[1][1]["bbox"] = [0, 120, 140, 170]
    monkeypatch.setattr(pipeline, "get_detector", lambda: detector)
    monkeypatch.setattr(detector, "detect_and_crop", lambda frame: detections)
    monkeypatch.setattr(pipeline, "get_reader", lambda: FakeReader([["2P", "8AM10PM"], extra_panel]))

    verdict, _ = pipeline.process_frame(FRAME, VehicleClass.SEDAN, time(12), "MON")

    assert bool(verdict.warning) == expected_warning
    assert verdict.allowed == expected_allowed


def test_lookback_flags_unreadable_hours(monkeypatch, tmp_path):
    path = tmp_path / "frame.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4, (300, 400))
    assert writer.isOpened()
    for _ in range(8):
        writer.write(FRAME)
    writer.release()

    detector = FakeDetector(1)
    detector.model = object()
    monkeypatch.setattr(pipeline, "get_detector", lambda **kwargs: detector)
    monkeypatch.setattr(pipeline, "get_reader", lambda: FakeReader([["2P", "8-6"]]))

    result = pipeline.scan_lookback_buffer(str(path), 1.5, arrival_time=time(7), arrival_day="MON")

    assert result["found"]
    assert result["verdict"].reason == "Couldn't read the operating hours."
    assert result["verdict"].max_stay_minutes is None


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


# Known-good end-to-end cases from the benchmark's labelled image set.
SCENARIOS = {
    (row["image"], row["day"], row["time"], row["vehicle"]): row
    for row in load_scenarios()
}


@pytest.mark.skipif(not Path(pipeline.DEFAULT_WEIGHTS).exists(), reason="no YOLO weights")
@pytest.mark.parametrize(
    "image, day, at, vehicle",
    [
        ("valid-2.png", "MON", "12:00", "sedan"),
        ("valid-2.png", "MON", "23:00", "sedan"),
        ("invalid-13.png", "SUN", "12:00", "sedan"),
    ],
)
def test_process_frame_matches_expected_verdict(image, day, at, vehicle):
    result = score_scenario(SCENARIOS[image, day, at, vehicle])
    assert result["exact_match"], result
