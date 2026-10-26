from pathlib import Path

import numpy as np
import pytest

from src.ocr import _group_lines, _normalise, _preprocess, easyocr


def piece(text, x1, y1, x2, y2, conf=0.9):
    """Builds a fake EasyOCR piece with a rectangular box."""
    return {"text": text, "conf": conf, "box": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]}


# _group_lines

def test_group_lines_orders_top_to_bottom():
    pieces = [
        piece("MON-SAT", 10, 100, 120, 130),
        piece("LOADING ZONE", 10, 10, 170, 40),
        piece("7AM-MIDNIGHT", 10, 55, 160, 85),
    ]
    assert _group_lines(pieces) == ["LOADING ZONE", "7AM-MIDNIGHT", "MON-SAT"]


def test_group_lines_joins_split_line_left_to_right():
    pieces = [
        piece("ZONE", 100, 12, 170, 40),
        piece("LOADING", 10, 10, 90, 42),
        piece("MON-SAT", 10, 60, 120, 90),
    ]
    assert _group_lines(pieces) == ["LOADING ZONE", "MON-SAT"]


def test_group_lines_tolerates_slight_tilt():
    pieces = [
        piece("8AM-6PM", 10, 10, 90, 40),
        piece("MON-FRI", 100, 18, 180, 48),
    ]
    assert _group_lines(pieces) == ["8AM-6PM MON-FRI"]


def test_group_lines_tall_text_does_not_swallow_line_below():
    # Real EasyOCR boxes from a 1P / 8AM-6PM / MON-SAT panel: the "8am" box
    # reaches down into "MON", which used to merge everything into one line
    pieces = [
        piece("1P", 21, 35, 145, 143),
        piece("8xn", 16, 190, 80, 242),
        piece("62m", 80, 180, 148, 232),
        piece("MON", 26, 234, 74, 262),
        piece("SAT", 88, 226, 130, 254),
    ]
    assert _group_lines(pieces) == ["1P", "8xn 62m", "MON SAT"]


def test_group_lines_empty():
    assert _group_lines([]) == []


# _normalise

@pytest.mark.parametrize("raw, expected", [
    ("7am - midnight", "7AM-MIDNIGHT"),
    ("8AM – 6PM", "8AM-6PM"),
    ("MON — FRI", "MON-FRI"),
    ("I0AM", "10AM"),
    ("1O MIN", "10 MIN"),
    ("8:3O PM", "8:30 PM"),
    ("IP", "1P"),
    ("¼P", "1/4P"),
    ("loading   zone", "LOADING ZONE"),
])
def test_normalise(raw, expected):
    assert _normalise(raw) == expected


@pytest.mark.parametrize("word", ["LOADING ZONE", "NO STOPPING", "CLEARWAY", "PERMIT HOLDERS EXCEPTED"])
def test_normalise_leaves_words_alone(word):
    assert _normalise(word) == word


# _preprocess

def test_preprocess_upscales_small_crop():
    crop = np.zeros((60, 100, 3), dtype=np.uint8)
    img, scale = _preprocess(crop)
    assert scale == 10.0
    assert img.shape == (600, 1000)


def test_preprocess_keeps_large_crop_size():
    crop = np.zeros((700, 200, 3), dtype=np.uint8)
    img, scale = _preprocess(crop)
    assert scale == 1.0
    assert img.shape == (700, 200)


# End-to-end on real crops (needs easyocr and fixtures/sample_crops/)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "sample_crops"


@pytest.mark.skipif(easyocr is None, reason="easyocr not installed")
@pytest.mark.skipif(not any(FIXTURES.glob("*.[jp][pn]g")), reason="no fixture crops")
def test_read_signs_on_fixtures():
    from src.detector import SignDetector
    from src.ocr import read_signs

    crops = SignDetector(use_mock=True, mock_dir=str(FIXTURES)).detect_and_crop("")
    results = read_signs(crops)

    assert len(results) == len(crops)
    assert any(r["lines"] for r in results)
