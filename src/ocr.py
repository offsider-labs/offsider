"""Text extraction from cropped sign panels, with PaddleOCR (default) or EasyOCR."""

import os
import re

import cv2
import numpy as np

try:
    import easyocr
except ImportError:
    easyocr = None

# Skips PaddleX's network check of its model hosts on every start. Must be set before
# paddleocr is imported. The models still download once on first use (to ~/.paddlex/).
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

try:
    import paddleocr
except ImportError:
    paddleocr = None

# Which OCR engine SignReader uses. PaddleOCR read words and times better than EasyOCR on
# both labelled crop sets (scripts/eval_cleanup.py). EasyOCR is kept as a fallback, chosen
# with SignReader(engine="easyocr") or the OFFSIDER_OCR_ENGINE environment variable.
ENGINES = ("paddle", "easyocr")
DEFAULT_ENGINE = os.environ.get("OFFSIDER_OCR_ENGINE", "paddle")


# Crops shorter than this are upscaled before OCR. Panel crops are usually
# 300-400px tall, where small text like the AM/PM next to times reads poorly.
MIN_HEIGHT = 600

# Two pieces are on the same line if they share at least this fraction of the taller one's height
SAME_LINE_OVERLAP = 0.5

# Blank border added around each crop before OCR, as a fraction of its shorter side.
# Meant to help with text touching the crop edge, but on fixtures/sample_crops every size
# tried (0.05, 0.08, 0.12) scored worse than none, so it's off. 0 turns the border off.
BORDER = 0.0

# EasyOCR readtext() settings. Empty means EasyOCR's own defaults
# (low_text=0.4, mag_ratio=1.0, width_ths=0.5).
READTEXT_OPTIONS: dict = {}

# Any dash variant (hyphen, en dash, em dash, ...) plus surrounding spaces
_DASH = re.compile(r"\s*[‐-―-]\s*")

# Number-like tokens followed by a unit, e.g. "7AM", "8:30 PM", "1P", "10 MIN".
# Digits may be misread as O, I or L, which are only corrected inside these tokens.
_UNIT = r"(?:AM|PM|P|MINS?|MINUTES?|HRS?|HOURS?)"
_NUMBER = re.compile(rf"\b([0-9OIL]{{1,2}}(?:[:.][0-9OIL]{{2}})?)(\s*{_UNIT})\b")
_DIGIT_FIXES = str.maketrans("OIL", "011")

# Fraction-hour parking, written either as a glyph or with a slash
_FRACTIONS = {"¼": "1/4", "½": "1/2"}

# Words that appear on NSW parking signs. OCR words close to one of these are snapped to it.
_DAYS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")
SIGN_WORDS = frozenset(_DAYS + (
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
    "LOADING", "ZONE", "STOPPING", "PARKING", "PARK", "CLEARWAY", "TOW", "AWAY",
    "METER", "TICKET", "PERMIT", "HOLDERS", "EXCEPTED", "EXCEPT", "AREA",
    "MIN", "MINS", "MINUTE", "MINUTES", "HRS", "HOUR", "HOURS", "MIDNIGHT", "MIDDAY", "NOON",
    "PUBLIC", "HOLIDAYS", "ALL", "OTHER", "TIMES", "FREE", "ONLY", "SPECIAL", "EVENTS",
    "TRUCK", "MOTOR", "BIKES", "BUS", "TAXI", "MAIL", "DISABLED", "SCHOOL", "DAYS",
    "RACE", "WORKS", "AUTHORISED", "VEHICLES", "COUNCIL", "CAR", "SHARE", "GOODS",
    "ANGLE", "REAR", "KERB", "PASSENGER", "SERVICE",
))

# Junk characters OCR leaves at the start or end of words, e.g. "(STOPPING]" or "@OADING"
_EDGE_JUNK = "()[]{}'\"`~_*;,:.!|@#$%^+=<>?"

# A number stuck to a word, e.g. "15MINUTES". Needs 3+ letters so "4P" and "7AM" are left alone.
_STUCK_NUMBER = re.compile(r"\b(\d+)([A-Z]{3,})\b")

# Two days next to each other are a range on signs: "MON FRI" -> "MON-FRI"
_DAY_PAIR = re.compile(rf"\b({'|'.join(_DAYS)}) ({'|'.join(_DAYS)})\b")

# --- Times ---
# NSW signs print times as a big hour, small raised minutes and a small AM/PM at the bottom
# right, e.g. 9³⁰ₐₘ. EasyOCR reads the hour and minutes as one number ("930") and the AM/PM
# as its own small piece below it, so both are put back together here.

# A small piece reading AM or PM, allowing for a stray extra letter ("PMA")
_MERIDIEM_PIECE = re.compile(r"([AP])M[A-Z]?")

# An AM/PM piece belongs to a number if it's at most this fraction of the number's height,
# its bottom is within this fraction of the number's height of the number's bottom,
# and it starts no further right of the number than this fraction of the number's height
MERIDIEM_MAX_HEIGHT = 0.7
MERIDIEM_BOTTOM_TOLERANCE = 0.35
MERIDIEM_MAX_GAP = 0.6

# "A" misread as "4" after an hour: "64M" -> "6AM"
_A_AS_4 = re.compile(r"(?<=\d)4M\b")

# A space between a time and its AM/PM: "3 PM" -> "3PM"
_SPACED_MERIDIEM = re.compile(r"\b(\d{1,2}(?::\d{2})?) (AM|PM)\b")

# Hour and minutes read as one number. Only quarter hours are accepted, as signs never use
# anything else, so a misread like "828" is left alone rather than guessed as 8:28.
_HOUR_MINUTES = r"(1[0-2]|[1-9])(00|15|30|45)"
_JOINED_BEFORE = re.compile(rf"(?<![\d:]){_HOUR_MINUTES}(?=AM\b|PM\b|-)")
_JOINED_AFTER = re.compile(rf"(?<=-){_HOUR_MINUTES}(?![\d:])")

# Hour and minutes read as two pieces: "9 30AM" -> "9:30AM". Same quarter-hour rule,
# and the minutes must be followed by AM/PM or a dash so plain numbers aren't joined.
_SPLIT_HOUR_MINUTES = re.compile(r"\b(1[0-2]|[1-9]) (00|15|30|45)(?=AM\b|PM\b|-)")

# Two times side by side whose dash was lost: "3PM 7PM" -> "3PM-7PM"
_TIME = r"\d{1,2}(?::\d{2})?(?:AM|PM)"
_TIME_PAIR = re.compile(rf"\b({_TIME}) ({_TIME})\b")

# A range whose start lost its AM/PM, e.g. "7:30-7:30PM" (OCR often misses the small AM/PM).
# Both AM and PM are tried for the start, and one is filled in only when the other is
# implausible for a parking sign (see _fill_start_meridiem). "6-10PM" stays as it is, since
# 6AM-10PM and 6PM-10PM are both real signs.
_CLOCK = r"(1[0-2]|[1-9])(?::(00|15|30|45))?"
_MISSING_START_MERIDIEM = re.compile(rf"(?<![\d:]){_CLOCK}-{_CLOCK}(AM|PM)\b")

# Longest restriction a sign's time range covers. Longer readings are treated as misreads:
# "12:30-10PM" can't be 12:30AM-10PM (21.5 hours), so it's 12:30PM-10PM.
MAX_RANGE_HOURS = 18

# --- Clean-up for how PaddleOCR reads times ---
# "TO" between two times means a range: "1AM TO 12 MIDDAY" -> "1AM-12 MIDDAY"
_TO_RANGE = re.compile(r"(?<=\d)(AM|PM)? TO (?=\d)")
# Doubled dashes: "6AM--10AM" -> "6AM-10AM"
_MULTI_DASH = re.compile(r"-{2,}")
# Stray dots or commas between a number and a dash: "8.-10" -> "8-10"
_PUNCT_BEFORE_DASH = re.compile(r"(?<=\d)[.,;]+(?=-)")

# --- Times that couldn't be read cleanly ---
# A finished range: both ends have AM/PM (or are MIDNIGHT / 12 MIDDAY)
_VALID_END = r"(?:\d{1,2}(?::\d{2})?(?:AM|PM)|MIDNIGHT|12 MIDDAY)"
_VALID_RANGE = re.compile(rf"\b\d{{1,2}}(?::\d{{2}})?(?:AM|PM)-{_VALID_END}(?![\w:])")
# A single time with AM/PM, which on a sign should always be one end of a range
_LONE_TIME = re.compile(r"\d{1,2}(?::\d{2})?(?:AM|PM)")


class SignReader:
    def __init__(
        self,
        languages: tuple[str, ...] = ("en",),
        min_conf: float = 0.2,
        gpu: bool = False,
        readtext_options: dict | None = None,
        engine: str | None = None,
    ):
        """
        engine: "paddle" or "easyocr" (default: DEFAULT_ENGINE).
        languages, gpu and readtext_options only apply to EasyOCR; PaddleOCR runs its
        English models on the CPU.
        """
        self.engine = engine or DEFAULT_ENGINE
        if self.engine not in ENGINES:
            raise ValueError(f"Unknown OCR engine {self.engine!r}, expected one of {ENGINES}")
        self.min_conf = min_conf

        # Loading a model is slow, so it is done once and reused for every frame
        if self.engine == "paddle":
            if paddleocr is None:
                raise ImportError("paddleocr is not installed. Run `uv sync`.")
            # The orientation and unwarping steps are for scanned pages; sign crops are upright
            self.reader = paddleocr.PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        else:
            if easyocr is None:
                raise ImportError("easyocr is not installed. Run `uv sync`.")
            # Settings passed to EasyOCR's readtext(), e.g. low_text, mag_ratio, width_ths
            self.readtext_options = {**READTEXT_OPTIONS, **(readtext_options or {})}
            self.reader = easyocr.Reader(list(languages), gpu=gpu)

    def read(self, crops: list[tuple[np.ndarray, dict]]) -> list[dict]:
        """
        Reads the text on each sign crop from SignDetector.detect_and_crop().

        Returns one dict per crop, in the same order:
            bbox: sign location in the original frame (from the detector)
            lines: normalised text lines, top to bottom
            raw: per-piece OCR output (text, conf, box in crop coordinates)
            mean_conf: average OCR confidence, 0.0 if nothing was read
            unclear_times: times that couldn't be read cleanly (see find_unclear_times)
        """
        results: list[dict] = []

        for crop, meta in crops:
            img, scale = _preprocess(crop)
            img, offset = _add_border(img)
            pieces = self._run_ocr(img, scale, offset)
            lines = [_normalise(line) for line in _group_lines(_attach_meridiems(pieces))]
            confs = [p["conf"] for p in pieces]

            results.append({
                "bbox": meta.get("bbox"),
                "lines": lines,
                "raw": pieces,
                "mean_conf": float(np.mean(confs)) if confs else 0.0,
                "unclear_times": find_unclear_times(lines),
            })

        return results

    def _run_ocr(self, img: np.ndarray, scale: float, offset: int = 0) -> list[dict]:
        """
        Runs the OCR engine and drops low-confidence pieces. Boxes are mapped back to crop
        coordinates by removing the border offset, then undoing the upscale.
        """
        pieces: list[dict] = []

        if self.engine == "paddle":
            # PaddleOCR expects a 3-channel image; _preprocess returns greyscale
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            result = self.reader.predict(img)[0]
            found = zip(
                [p.tolist() for p in result["rec_polys"]], result["rec_texts"], result["rec_scores"]
            )
        else:
            found = self.reader.readtext(img, **self.readtext_options)

        for box, text, conf in found:
            text = text.strip()
            if conf < self.min_conf or not text:
                continue
            pieces.append({
                "text": text,
                "conf": float(conf),
                "box": [[int((x - offset) / scale), int((y - offset) / scale)] for x, y in box],
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


def _add_border(img: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Pads the image with a plain border in its own edge colour. Returns the image and the
    border width, which is needed to map OCR boxes back to the original crop.

    A flat colour is used rather than repeating the edge pixels, which would smear any
    text touching the edge out into the border.
    """
    width = int(BORDER * min(img.shape[:2]))
    if width == 0:
        return img, 0

    edges = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    colour = np.median(edges, axis=0).tolist()
    padded = cv2.copyMakeBorder(img, width, width, width, width, cv2.BORDER_CONSTANT, value=colour)
    return padded, width


def _attach_meridiems(pieces: list[dict]) -> list[dict]:
    """
    Joins each small AM/PM piece onto the number it belongs to, e.g. "930 -" + "AM" -> "930AM -".

    On signs the AM/PM sits under the minutes or just right of the hour, with its bottom
    level with the digits' bottom. If a number piece holds two times ("730 - 730"), the
    AM/PM's horizontal position decides which one it goes after. AM/PM pieces that don't
    fit any number are kept as they are.
    """
    numbers = [p for p in pieces if re.search(r"\d", p["text"])]
    inserts: dict[int, list[tuple[int, str]]] = {}  # id(number piece) -> [(text index, "AM"/"PM")]
    attached: set[int] = set()

    for p in pieces:
        m = _MERIDIEM_PIECE.fullmatch(p["text"].upper().strip(_EDGE_JUNK))
        if not m:
            continue
        x1, y1, x2, y2 = _bounds(p)
        centre = (x1 + x2) / 2

        best, best_gap = None, None
        for n in numbers:
            nx1, ny1, nx2, ny2 = _bounds(n)
            h = max(ny2 - ny1, 1)
            gap = max(0.0, centre - nx2) if centre >= nx1 else None
            if (
                gap is None
                or y2 - y1 > MERIDIEM_MAX_HEIGHT * h
                or abs(y2 - ny2) > MERIDIEM_BOTTOM_TOLERANCE * h
                or gap > MERIDIEM_MAX_GAP * h
            ):
                continue
            if best is None or gap < best_gap:
                best, best_gap = n, gap

        if best is None:
            continue

        # Estimate which character of the number piece the AM/PM sits under,
        # then insert after the run of digits there (or the nearest run to its left)
        nx1, _, nx2, _ = _bounds(best)
        text = best["text"]
        at = min(len(text) - 1, max(0, int((centre - nx1) / max(nx2 - nx1, 1) * len(text))))
        runs = [r for r in re.finditer(r"[0-9OIL]+", text.upper()) if re.search(r"\d", r.group())]
        if not runs:
            continue
        run = next((r for r in runs if r.start() <= at < r.end()), None)
        run = run or max((r for r in runs if r.end() <= at), key=lambda r: r.end(), default=runs[0])

        inserts.setdefault(id(best), []).append((run.end(), m.group(1) + "M"))
        attached.add(id(p))

    result = []
    for p in pieces:
        if id(p) in attached:
            continue
        if id(p) in inserts:
            text = p["text"]
            # Insert right to left so earlier positions stay valid
            for pos, meridiem in sorted(set(inserts[id(p)]), reverse=True):
                text = text[:pos] + meridiem + text[pos:]
            p = {**p, "text": text}
        result.append(p)
    return result


def _bounds(piece: dict) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of a piece's box."""
    xs = [pt[0] for pt in piece["box"]]
    ys = [pt[1] for pt in piece["box"]]
    return min(xs), min(ys), max(xs), max(ys)


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
    text = _MULTI_DASH.sub("-", text)
    text = _PUNCT_BEFORE_DASH.sub("", text)
    text = _TO_RANGE.sub(lambda m: (m.group(1) or "") + "-", text)
    text = " ".join(w.strip(_EDGE_JUNK) for w in text.split())
    text = _STUCK_NUMBER.sub(r"\1 \2", text)
    text = " ".join("-".join(_correct_word(part) for part in w.split("-")) for w in text.split())
    text = _DAY_PAIR.sub(r"\1-\2", text)
    text = _A_AS_4.sub("AM", text)
    text = _NUMBER.sub(lambda m: m.group(1).translate(_DIGIT_FIXES) + m.group(2), text)
    text = _SPACED_MERIDIEM.sub(r"\1\2", text)
    text = _SPLIT_HOUR_MINUTES.sub(r"\1:\2", text)
    text = _JOINED_BEFORE.sub(r"\1:\2", text)
    text = _JOINED_AFTER.sub(r"\1:\2", text)
    text = _TIME_PAIR.sub(r"\1-\2", text)
    text = _MISSING_START_MERIDIEM.sub(_fill_start_meridiem, text)
    return " ".join(text.split())


def _fill_start_meridiem(m: re.Match) -> str:
    """
    Adds the AM/PM to a range's start when only one choice is plausible on a parking sign.

    A choice is implausible if the range would last longer than MAX_RANGE_HOURS
    ("7:30-7:30PM" as 7:30PM-7:30PM is 24 hours). If both are plausible, a range within one
    day beats one running overnight ("11-12PM" is 11AM-12PM, not 11PM to noon the next day).
    Otherwise it's left alone: "7-12AM" could be 7AM or 7PM to midnight.
    """
    start_h, start_m, end_h, end_m, end_meridiem = m.groups()

    def minute_of_day(hour: str, mins: str | None, meridiem: str) -> int:
        return (int(hour) % 12 + (12 if meridiem == "PM" else 0)) * 60 + int(mins or 0)

    end = minute_of_day(end_h, end_m, end_meridiem) or 24 * 60  # ending at 12AM means midnight
    plausible = []
    for meridiem in ("AM", "PM"):
        start = minute_of_day(start_h, start_m, meridiem)
        overnight = end <= start
        length = end - start + (24 * 60 if overnight else 0)
        if length <= MAX_RANGE_HOURS * 60:
            plausible.append((meridiem, overnight))

    if len(plausible) == 2:
        plausible = [p for p in plausible if not p[1]]
    if len(plausible) != 1:
        return m.group(0)
    return m.group(0).replace("-", f"{plausible[0][0]}-", 1)


def find_unclear_times(lines: list[str]) -> list[str]:
    """
    Returns times in the normalised lines that couldn't be read cleanly: ranges missing an
    AM/PM ("8-10"), garbled ranges ("83M-6PM"), a time with no other end ("8AM"), or an
    AM/PM with no number ("AM").

    Used to flag a reading for a manual check rather than trusting a guess. Dates and
    durations ("SAT 26 SEP", "15 MINUTES") aren't time attempts, so they're never flagged.
    """
    unclear = []
    for line in lines:
        for word in _VALID_RANGE.sub(" ", line).split():  # finished ranges are fine
            has_digit = any(c.isdigit() for c in word)
            # An AM/PM on its own means its numbers weren't read ("AM PM" on IMG_1529_panel1)
            if (has_digit and "-" in word) or _LONE_TIME.fullmatch(word) or word in ("AM", "PM"):
                unclear.append(word)
    return unclear


def _correct_word(word: str) -> str:
    """
    Snaps a misread word to the closest SIGN_WORDS entry, e.g. "LOADIG" -> "LOADING".

    Only all-letter words of 3+ letters are changed, so numbers, times and short words
    like "NO" or "P" are never touched. Up to 1 edit is allowed for words of 5 letters
    or fewer and 2 for longer ones. Words with no close match, or a tie between two
    matches, are left as they are.
    """
    if len(word) < 3 or not word.isalpha() or word in SIGN_WORDS:
        return word

    max_edits = 1 if len(word) <= 5 else 2
    scored = sorted((_edit_distance(word, w), w) for w in SIGN_WORDS)
    best_dist, best = scored[0]
    if best_dist > max_edits or scored[1][0] == best_dist:
        return word
    return best


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance: the fewest single-letter inserts, deletes or swaps to turn a into b."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
