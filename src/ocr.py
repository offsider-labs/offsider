""" EasyOCR - text extraction from cropped sign panels."""

import re

import cv2
import numpy as np

try:
    import easyocr
except ImportError:
    easyocr = None


# Crops shorter than this are upscaled before OCR. Panel crops are usually
# 300-400px tall, where small text like the AM/PM next to times reads poorly.
MIN_HEIGHT = 600

# Two pieces are on the same line if they share at least this fraction of the taller one's height
SAME_LINE_OVERLAP = 0.5

# Any dash variant (hyphen, en dash, em dash, ...) plus surrounding spaces
_DASH = re.compile(r"\s*[‐-―-]\s*")

# Number-like tokens followed by a unit, e.g. "7AM", "8:30 PM", "1P", "10 MIN".
# Digits may be misread as O, I or L, which are only corrected inside these tokens.
_UNIT = r"(?:AM|PM|P|MINS?|MINUTES?|HRS?|HOURS?)"
_NUMBER = re.compile(rf"\b([0-9OIL]{{1,2}}(?:[:.][0-9OIL]{{2}})?)(\s*{_UNIT})\b")
_DIGIT_FIXES = str.maketrans("OIL", "011")

# Fraction-hour parking, written either as a glyph or with a slash
_FRACTIONS = {"¼": "1/4", "½": "1/2"}


class SignReader:
    def __init__(
        self,
        languages: tuple[str, ...] = ("en",),
        min_conf: float = 0.2,
        gpu: bool = False,
    ):
        if easyocr is None:
            raise ImportError("easyocr is not installed. Run `uv sync`.")

        self.min_conf = min_conf
        # Loading the model is slow, so it is done once and reused for every frame
        self.reader = easyocr.Reader(list(languages), gpu=gpu)

    def read(self, crops: list[tuple[np.ndarray, dict]]) -> list[dict]:
        """
        Reads the text on each sign crop from SignDetector.detect_and_crop().

        Returns one dict per crop, in the same order:
            bbox: sign location in the original frame (from the detector)
            lines: normalised text lines, top to bottom
            raw: per-piece EasyOCR output (text, conf, box in crop coordinates)
            mean_conf: average OCR confidence, 0.0 if nothing was read
        """
        results: list[dict] = []

        for crop, meta in crops:
            img, scale = _preprocess(crop)
            pieces = self._run_ocr(img, scale)
            lines = [_normalise(line) for line in _group_lines(pieces)]
            confs = [p["conf"] for p in pieces]

            results.append({
                "bbox": meta.get("bbox"),
                "lines": lines,
                "raw": pieces,
                "mean_conf": float(np.mean(confs)) if confs else 0.0,
            })

        return results

    def _run_ocr(self, img: np.ndarray, scale: float) -> list[dict]:
        """Runs EasyOCR and drops low-confidence pieces. Boxes are mapped back to crop coordinates."""
        pieces: list[dict] = []

        for box, text, conf in self.reader.readtext(img):
            text = text.strip()
            if conf < self.min_conf or not text:
                continue
            pieces.append({
                "text": text,
                "conf": float(conf),
                "box": [[int(x / scale), int(y / scale)] for x, y in box],
            })

        return pieces


_reader: SignReader | None = None


def read_signs(crops: list[tuple[np.ndarray, dict]]) -> list[dict]:
    """Reads sign crops using one shared SignReader, created on first use."""
    global _reader
    if _reader is None:
        _reader = SignReader()
    return _reader.read(crops)


def _preprocess(crop: np.ndarray) -> tuple[np.ndarray, float]:
    """Upscales small crops and boosts contrast. Returns the image and the scale applied."""
    scale = max(1.0, MIN_HEIGHT / crop.shape[0])
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    # CLAHE evens out glare and faded paint without blowing out the whole crop
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray), scale


def _group_lines(pieces: list[dict]) -> list[str]:
    """
    Puts OCR pieces into reading order and joins pieces on the same row.

    A piece joins a line if it vertically overlaps a piece already on that
    line by at least SAME_LINE_OVERLAP of the taller piece's height.
    Comparing piece to piece (not to the line's whole span) stops a line of
    tall text, like "8-6" on a time panel, from swallowing the line below.
    """
    spans = []
    for p in pieces:
        xs = [pt[0] for pt in p["box"]]
        ys = [pt[1] for pt in p["box"]]
        spans.append((min(ys), max(ys), min(xs), p["text"]))

    spans.sort(key=lambda s: (s[0] + s[1]) / 2)

    lines: list[list[tuple]] = []
    for span in spans:
        for line in lines:
            if any(_vertical_overlap(span, other) >= SAME_LINE_OVERLAP for other in line):
                line.append(span)
                break
        else:
            lines.append([span])

    return [" ".join(s[3] for s in sorted(line, key=lambda s: s[2])) for line in lines]


def _vertical_overlap(a: tuple, b: tuple) -> float:
    """Shared height of two (top, bottom, ...) spans, as a fraction of the taller one."""
    shared = min(a[1], b[1]) - max(a[0], b[0])
    return max(0, shared) / max(a[1] - a[0], b[1] - b[0], 1)


def _normalise(text: str) -> str:
    """Cleans up common OCR noise so the parser sees one consistent spelling."""
    text = " ".join(text.upper().split())

    for glyph, ascii_form in _FRACTIONS.items():
        text = text.replace(glyph, ascii_form)

    text = _DASH.sub("-", text)
    text = _NUMBER.sub(lambda m: m.group(1).translate(_DIGIT_FIXES) + m.group(2), text)
    return text
