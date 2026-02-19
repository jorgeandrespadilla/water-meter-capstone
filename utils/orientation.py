"""Utilities for resolving 0°/180° orientation of odometer crops.

Designed for odometer crops where the remaining ambiguity after OBB
extraction is upright vs. upside-down.

Strategy (strict cascade):

1. **Ensure landscape** — if the crop is portrait, rotate 90° clockwise
   so that width >= height.  This is a prerequisite for the next stages.

2. **Red-position cue** — if red decimal digits are detected and their
   centroid is clearly on one side of the crop, the orientation is resolved
   instantly (red digits are always on the right in upright odometers).
   Zero OCR cost.

3. **Dual-OCR with enriched scoring** — when the red cue is absent or
   ambiguous, OCR is run on both 0° and 180°.  The orientation with
   the higher composite score wins.  The composite score combines:
     - OCR recognition confidence (primary signal)
     - Digit-character ratio (penalizes non-digit artifacts)
     - Digit count plausibility (rewards readings in the 4–8 digit range)
"""

import re
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from utils.masking import apply_red_mask, create_red_mask

# ---------------------------------------------------------------------------
# Constants — Stage 1 (red-position cue)
# ---------------------------------------------------------------------------

# Minimum red pixels required for the red-position cue to be trusted.
_RED_MIN_PIXELS = 100

# Minimum percentage of total crop pixels that must be red.
# Real red digits typically occupy 0.5–2% of the crop; noise is < 0.3%.
_RED_MIN_PCT = 1.0

# If the mean x-position of red pixels is within this margin of the center
# (as a fraction of image width), the cue is considered ambiguous.
_RED_CENTER_MARGIN = 0.05

# Minimum spatial concentration: fraction of red pixels that must reside
# in whichever half (left or right) contains the centroid.
# Real digits are spatially grouped; noise is scattered.
_RED_MIN_CONCENTRATION = 0.65

# ---------------------------------------------------------------------------
# Constants — Stage 2 (dual-OCR enriched scoring)
# ---------------------------------------------------------------------------

_W_CONF = 0.6          # weight for raw OCR confidence
_W_DIGIT_RATIO = 0.3   # weight for digit-char ratio in output
_W_COUNT_BONUS = 0.1   # weight for plausible digit count

_EXPECTED_DIGIT_MIN = 4
_EXPECTED_DIGIT_MAX = 8

# If composite score delta is below this, flag as ambiguous.
_DUAL_OCR_AMBIGUITY_THRESHOLD = 0.03

# ---------------------------------------------------------------------------
# Constants — Confidence normalization (both stages → 0–1 scale)
# ---------------------------------------------------------------------------

# Maximum possible centroid offset (0.5 = fully at one edge).
_RED_MAX_OFFSET = 0.5

# Reference delta for dual-OCR: a delta >= this maps to confidence 1.0.
_DUAL_OCR_REF_DELTA = 0.15

_DIGITS_RE = re.compile(r"\D")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class OrientationResult:
    """Result of the hybrid orientation resolution."""

    angle: int
    """Total rotation applied: 0, 90, 180, or 270."""

    oriented: np.ndarray
    """BGR image after applying the chosen rotation."""

    masked: np.ndarray
    """BGR image after orientation + red masking (ready for OCR)."""

    method: str
    """Which strategy resolved 0/180: 'red_position' or 'dual_ocr'."""

    confidence: float
    """Normalized orientation confidence (0–1 scale).
    For red_position: centroid offset normalized by max offset (0.5).
    For dual_ocr: score delta normalized by reference delta (0.15)."""

    raw_text: str
    """OCR raw text from the chosen orientation (set by dual_ocr, empty for red_position)."""

    recognition_confidence: float
    """OCR confidence from the chosen orientation (set by dual_ocr, 0.0 for red_position)."""

    low_confidence: bool = False
    """True if the orientation decision was ambiguous (dual_ocr delta below threshold)."""

    ocr_elapsed_s: float = 0.0
    """Seconds spent on OCR during orientation (non-zero only for dual_ocr)."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ensure_landscape(image: np.ndarray) -> np.ndarray:
    """Rotate an image 90° clockwise if it is portrait (height > width).

    Useful for scripts that need landscape crops without the full
    orientation resolution pipeline.
    """
    if image.shape[0] > image.shape[1]:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def resolve_orientation(
    crop: np.ndarray,
    run_rec: Callable[[np.ndarray], tuple[str, float]],
) -> OrientationResult:
    """Resolve orientation for an odometer crop (cascading strategy).

    Stage 0: ensure landscape (rotate 90° CW if portrait).
    Stage 1: red-position cue (fast, no OCR).
    Stage 2: dual-OCR with enriched scoring (fallback).

    Args:
        crop: BGR crop from OBB detection (may be portrait or landscape).
        run_rec: callable(bgr_image) -> (text, confidence) — the OCR
                 recognition function for the dual-OCR fallback.

    Returns:
        OrientationResult with the chosen orientation and preprocessed images.
    """
    # Stage 0: ensure landscape
    was_portrait = crop.shape[0] > crop.shape[1]
    landscape = ensure_landscape(crop)

    # Stage 1: red-position cue
    result = _try_red_position(landscape)
    if result is not None:
        if was_portrait:
            result.angle = (result.angle + 90) % 360
        return result

    # Stage 2: dual-OCR fallback
    result = _dual_ocr_fallback(landscape, run_rec)
    if was_portrait:
        result.angle = (result.angle + 90) % 360
    return result


# ---------------------------------------------------------------------------
# Stage 1: red-position cue
# ---------------------------------------------------------------------------


def _try_red_position(crop: np.ndarray) -> OrientationResult | None:
    """Resolve orientation by checking which side the red decimals are on.

    In upright odometers the red digits sit on the right side of the crop.
    If the red mass centroid is clearly left-of-center the image is flipped.

    Returns None if the cue is absent or ambiguous.
    """
    red_mask = create_red_mask(crop)
    _, xs = np.where(red_mask > 0)

    # Gate 1: minimum absolute red pixel count
    if len(xs) < _RED_MIN_PIXELS:
        return None

    # Gate 2: minimum red pixel percentage
    h, w = crop.shape[:2]
    red_pct = (len(xs) / (h * w)) * 100
    if red_pct < _RED_MIN_PCT:
        return None

    # Gate 3: centroid must be clearly off-center
    mean_x_norm = float(xs.mean()) / w
    offset = abs(mean_x_norm - 0.5)

    if offset < _RED_CENTER_MARGIN:
        return None

    # Gate 4: red pixels must be spatially concentrated on one side
    if mean_x_norm < 0.5:
        same_half = int((xs < (w / 2)).sum())
    else:
        same_half = int((xs >= (w / 2)).sum())

    if (same_half / len(xs)) < _RED_MIN_CONCENTRATION:
        return None

    angle = 180 if mean_x_norm < 0.5 else 0
    oriented = _rotate_bgr(crop, angle)
    masked = apply_red_mask(oriented, include_stats=False).image

    return OrientationResult(
        angle=angle,
        oriented=oriented,
        masked=masked,
        method="red_position",
        confidence=round(min(offset / _RED_MAX_OFFSET, 1.0), 4),
        raw_text="",
        recognition_confidence=0.0,
    )


# ---------------------------------------------------------------------------
# Stage 2: dual-OCR with enriched scoring
# ---------------------------------------------------------------------------


def _compute_score(text: str, conf: float) -> float:
    """Compute composite orientation score for an OCR result."""
    digits = _DIGITS_RE.sub("", text)
    n_digits = len(digits)
    digit_ratio = len(digits) / len(text) if text else 0.0

    if _EXPECTED_DIGIT_MIN <= n_digits <= _EXPECTED_DIGIT_MAX:
        count_bonus = 1.0
    elif n_digits > 0:
        count_bonus = 0.5
    else:
        count_bonus = 0.0

    return _W_CONF * conf + _W_DIGIT_RATIO * digit_ratio + _W_COUNT_BONUS * count_bonus


def _dual_ocr_fallback(
    crop: np.ndarray,
    run_rec: Callable[[np.ndarray], tuple[str, float]],
) -> OrientationResult:
    """Run OCR on 0° and 180°, pick winner by composite score."""
    t_ocr_start = time.perf_counter()
    # Compute masking once in the base orientation and reuse it for 180°.
    # For a 180° ambiguity this is equivalent to masking the rotated image,
    # but avoids repeating the full masking pipeline.
    oriented_0 = crop
    masked_0 = apply_red_mask(oriented_0, include_stats=False).image
    text_0, conf_0 = run_rec(masked_0)
    score_0 = _compute_score(text_0, conf_0)

    oriented_180 = _rotate_bgr(crop, 180)
    masked_180 = _rotate_bgr(masked_0, 180)
    text_180, conf_180 = run_rec(masked_180)
    score_180 = _compute_score(text_180, conf_180)
    t_ocr_elapsed = time.perf_counter() - t_ocr_start

    if score_0 >= score_180:
        winner_score = score_0
        loser_score = score_180
        winner = OrientationResult(
            angle=0,
            oriented=oriented_0,
            masked=masked_0,
            method="dual_ocr",
            confidence=0.0,
            raw_text=text_0,
            recognition_confidence=conf_0,
        )
    else:
        winner_score = score_180
        loser_score = score_0
        winner = OrientationResult(
            angle=180,
            oriented=oriented_180,
            masked=masked_180,
            method="dual_ocr",
            confidence=0.0,
            raw_text=text_180,
            recognition_confidence=conf_180,
        )

    delta = winner_score - loser_score
    winner.confidence = round(min(delta / _DUAL_OCR_REF_DELTA, 1.0), 4)
    winner.low_confidence = delta < _DUAL_OCR_AMBIGUITY_THRESHOLD
    winner.ocr_elapsed_s = t_ocr_elapsed

    return winner


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _rotate_bgr(image: np.ndarray, angle: int) -> np.ndarray:
    """Rotate a BGR image by 0, 90, 180, or 270 degrees."""
    if angle == 0:
        return image
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported angle: {angle}")
