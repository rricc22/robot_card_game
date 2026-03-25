"""
Card vision module.

Detects two playing cards in the arena using OpenCV and reads their rank
with EasyOCR. Designed for an overhead camera view of the table.

Card detection: threshold → contour → filter by area + aspect ratio
Rank extraction: crop top-left corner of each card → upscale → OCR
"""

import cv2
import numpy as np
from typing import Optional
import easyocr

_reader: Optional[easyocr.Reader] = None

# Map OCR output quirks to canonical rank strings
OCR_RANK_MAP = {
    '1':  '10',  # "10" sometimes reads as "1"
    'T':  '10',  # some fonts render 10 as T
    '0':  '10',
    '1O': '10',
    '10': '10',
    'IO': '10',
    'A':  'A',
    'J':  'J',
    'Q':  'Q',
    'K':  'K',
}
VALID_RANKS = {'2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'}


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader


def capture_frame(camera_index: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def _detect_card_regions(frame: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """
    Find rectangular white-ish card regions in frame.
    Returns list of (x_position, card_image) sorted left-to-right.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_area = frame.shape[0] * frame.shape[1]
    cards = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Card should be 5%–50% of frame area
        if area < frame_area * 0.05 or area > frame_area * 0.50:
            continue

        # Must approximate to a quadrilateral
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        # Card aspect ratio: ~1.4 (portrait) or ~0.7 (landscape)
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = max(w, h) / min(w, h)
        if ratio < 1.2 or ratio > 2.2:
            continue

        card_img = frame[y:y + h, x:x + w]
        cards.append((x, card_img))

    cards.sort(key=lambda c: c[0])
    return cards


def _ocr_rank(card_img: np.ndarray) -> Optional[str]:
    """Extract rank from the top-left corner of a card image."""
    h, w = card_img.shape[:2]

    # Crop top-left corner (~15% of width, ~20% of height)
    corner = card_img[:max(1, h // 5), :max(1, w // 6)]

    # Upscale for better OCR accuracy
    corner = cv2.resize(corner, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Increase contrast
    corner = cv2.convertScaleAbs(corner, alpha=1.5, beta=10)

    reader = _get_reader()
    results = reader.readtext(
        corner,
        detail=0,
        allowlist='A23456789TJQK10',
        paragraph=False,
    )

    for text in results:
        text = text.upper().strip().replace(' ', '')
        # Direct map
        rank = OCR_RANK_MAP.get(text, text)
        if rank in VALID_RANKS:
            return rank
        # Try digits only (e.g. "10J" → "10")
        for r in ['10', '9', '8', '7', '6', '5', '4', '3', '2']:
            if text.startswith(r):
                return r

    return None


def read_cards(camera_index: int, attempts: int = 3) -> tuple[Optional[str], Optional[str]]:
    """
    Try `attempts` times to read two card ranks from camera.
    Returns (robot_rank, human_rank) — either can be None on failure.
    Robot card is on the LEFT, human card is on the RIGHT (by convention).
    """
    for attempt in range(attempts):
        frame = capture_frame(camera_index)
        if frame is None:
            continue

        cards = _detect_card_regions(frame)
        if len(cards) < 2:
            continue

        rank_left = _ocr_rank(cards[0][1])
        rank_right = _ocr_rank(cards[1][1])

        if rank_left and rank_right:
            return rank_left, rank_right

    return None, None


def save_debug_frame(camera_index: int, path: str = "debug_frame.jpg") -> bool:
    """Capture a frame and save it for debugging card detection."""
    frame = capture_frame(camera_index)
    if frame is None:
        return False

    cards = _detect_card_regions(frame)
    for i, (x, card_img) in enumerate(cards):
        cv2.imwrite(f"debug_card_{i}.jpg", card_img)

    # Draw detected regions on the original frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 160, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(path, frame)
    return True
