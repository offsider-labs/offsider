from pathlib import Path

import numpy as np
import pytest

from src.ocr import _attach_meridiems, _group_lines, _normalise, _preprocess, easyocr


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
    ("8:3O PM", "8:30PM"),
    ("IP", "1P"),
    ("¼P", "1/4P"),
    ("loading   zone", "LOADING ZONE"),
])
def test_normalise(raw, expected):
    assert _normalise(raw) == expected


@pytest.mark.parametrize("word", ["LOADING ZONE", "NO STOPPING", "CLEARWAY", "PERMIT HOLDERS EXCEPTED"])
def test_normalise_leaves_words_alone(word):
    assert _normalise(word) == word


# Word correction: real misreads from fixtures/sample_crops

@pytest.mark.parametrize("raw, expected", [
    ("(LOADIG", "LOADING"),
    ("LOADINGI", "LOADING"),
    ("@OADING", "LOADING"),
    ("STOPPG", "STOPPING"),
    ("(STOPPING]", "STOPPING"),
    ("'PERMI HOLDERS", "PERMIT HOLDERS"),
    ("ZONEI", "ZONE"),
    ("PM:", "PM"),
])
def test_normalise_corrects_misread_words(raw, expected):
    assert _normalise(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("MON FRI", "MON-FRI"),
    ("SAT SUN", "SAT-SUN"),
    ("MON - FRI", "MON-FRI"),
])
def test_normalise_joins_day_ranges(raw, expected):
    assert _normalise(raw) == expected


def test_normalise_splits_number_stuck_to_word():
    assert _normalise("15MINUTES") == "15 MINUTES"


@pytest.mark.parametrize("text", [
    "WENTWORTH PARK",  # place name: no close sign word
    "MUN",             # as close to MON as to SUN, so left alone
    "12",
    "4P",
    "7AM",
    "NO",
])
def test_normalise_does_not_force_corrections(text):
    assert _normalise(text) == text


# Times: AM/PM attachment (boxes are real EasyOCR output from fixtures/sample_crops)

def texts(pieces):
    return [p["text"] for p in pieces]


def test_attach_meridiems_under_minutes_and_beside_hour():
    # IMG_1282_panel2: 9:30AM - 6PM
    pieces = [
        piece("930 -", 29, 288, 172, 357),
        piece("AM", 91, 331, 134, 359),
        piece("6", 163, 298, 220, 362),
        piece("PM", 221, 335, 260, 361),
    ]
    assert texts(_attach_meridiems(pieces)) == ["930AM -", "6PM"]


def test_attach_meridiems_two_times_in_one_piece():
    # IMG_1529_panel1: 7:30AM - 7:30PM, read as one piece plus "AM" and "PMA"
    pieces = [
        piece("730- 730", 20, 267, 213, 344),
        piece("AM", 62, 312, 103, 336),
        piece("PMA", 167, 318, 207, 345),
    ]
    assert texts(_attach_meridiems(pieces)) == ["730AM- 730PM"]


def test_attach_meridiems_ignores_far_away_am_pm():
    pieces = [piece("4P", 87, 9, 186, 68), piece("PM", 100, 200, 130, 215)]
    assert texts(_attach_meridiems(pieces)) == ["4P", "PM"]


def test_attach_meridiems_ignores_am_pm_as_tall_as_the_number():
    # Full-size "PM" text next to a number isn't the small superscript style
    pieces = [piece("3", 10, 10, 50, 60), piece("PM", 55, 10, 120, 60)]
    assert texts(_attach_meridiems(pieces)) == ["3", "PM"]


# Times: text rules

@pytest.mark.parametrize("raw, expected", [
    ("930AM - 6PM", "9:30AM-6PM"),
    ("730AM- 730PM", "7:30AM-7:30PM"),
    ("1230PM -10PM", "12:30PM-10PM"),
    ("8AM -I0PM", "8AM-10PM"),
    ("64M-10AM", "6AM-10AM"),
    ("84M 12AM", "8AM-12AM"),
    ("3 PM = 7PM", "3PM-7PM"),
    ("9 30AM-6PM", "9:30AM-6PM"),      # hour and minutes read as separate pieces
    ("9 30 AM - 6 PM", "9:30AM-6PM"),
    ("8AM-12 30PM", "8AM-12:30PM"),
    ("9 30-6PM", "9:30-6PM"),
])
def test_normalise_rebuilds_times(raw, expected):
    assert _normalise(raw) == expected


@pytest.mark.parametrize("text", [
    "828 AM",     # 8:28 isn't a quarter hour, so it's a misread, not guessed
    "123-10PM",   # same: could be 12:3 or 1:23
    "20603001",   # long numbers (e.g. pay-by-phone zone) are untouched
    "15 MINUTES",
    "1 15 MINUTES",  # two numbers not followed by AM/PM or a dash aren't joined
    "9 20AM",        # 20 isn't a quarter hour
])
def test_normalise_leaves_unclear_times_alone(text):
    assert _normalise(text) == text


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
