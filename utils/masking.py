"""Red-digit masking utilities for the water meter reading project.

Detects and removes red decimal digits from odometer crops using a
two-stage approach:

  1. **LAB relative redness** - the CIE-LAB a* channel measures the
     red-green axis. By computing each pixel's a* deviation from the
     image median, we detect pixels that are *redder than their context*,
     which cleanly handles reddish color casts that defeat absolute HSV
     thresholds.

  2. **HSV hue gating** - intersection with the HSV red / magenta range
     to suppress false positives on non-red bright regions.

The algorithm never skips masking - it always produces a mask. When
the image has very little red content the mask is simply small and
the erasure has negligible effect.

Erasure uses a two-stage approach:

  1. **TELEA inpainting** - ``cv2.inpaint`` with the Telea algorithm fills
     masked pixels from their boundary inward, preserving local texture.
  2. **Selective a* suppression** - residual redness in a soft fringe zone
     around the mask is reduced by suppressing excess a* (red-green) values
     in CIE-LAB space. Only pixels *above* the image median are affected,
     so the natural color cast is preserved and no green artifacts appear.

Key public API:
    apply_red_mask(image) -> MaskingResult
    analyze_red_distribution(image) -> dict
    create_red_mask(image) -> np.ndarray
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# LAB-based masking constants
# ---------------------------------------------------------------------------

# Relative a* threshold: pixels with a*_deviation > this are considered red.
_LAB_A_THRESHOLD = 5.0

# HSV hue gate:
#   - Keep the classic low-red band narrow to avoid orange/yellow leakage.
#   - Broaden only the high-red band toward magenta because several decimal
#     wheels are pink-ish rather than pure red.
#   - Require some saturation so the wider high band does not activate on
#     neutral reflections with unstable hue.
_HSV_LOW_RED_MAX = 25
_HSV_HIGH_RED_MIN = 130
_HSV_MIN_SATURATION = 20
_HSV_PURPLE_FALLBACK_MIN = 110
_HSV_PURPLE_FALLBACK_MIN_SATURATION = 70
_LAB_A_PURPLE_FALLBACK_THRESHOLD = 6.0

# Morphological close: fills small interior holes in red digit masks.
_CLOSE_KSIZE = 3
_CLOSE_ITERATIONS = 1

# Dilation kernel and iterations for the inpainting mask
_DILATE_ITERATIONS = 1

# TELEA inpainting radius (pixels)
_INPAINT_RADIUS = 3

# Fringe desaturation: dilation beyond the inpaint mask that defines
# the zone where residual redness is suppressed in LAB a* space.
_FRINGE_DILATE = 5
_FRINGE_BLUR_KSIZE = 11
# Fraction of excess a* to suppress (0 = none, 1 = full neutralization)
_FRINGE_SUPPRESSION = 0.8


# ---------------------------------------------------------------------------
# MaskingResult
# ---------------------------------------------------------------------------


@dataclass
class MaskingResult:
    """Result of a masking operation with full explainability."""

    image: np.ndarray
    """Output image - masked version."""

    raw_mask: np.ndarray
    """Binary mask of detected red pixels (before dilation)."""

    dilated_mask: np.ndarray
    """Binary mask after morphological dilation."""

    action: str
    """What happened: 'applied'."""

    reason: str
    """Human-readable explanation of the decision."""

    stats: dict = field(default_factory=dict)
    """Full analysis stats from analyze_red_distribution (for debugging)."""


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def analyze_red_distribution(image: np.ndarray) -> dict:
    """Analyze spatial and statistical distribution of red pixels.

    Uses the LAB a* channel relative-redness method to understand *where*
    and *how much* red is present in the image.

    Args:
        image: BGR image (numpy array).

    Returns:
        dict with keys:
            red_pixel_pct         - % of total pixels in the LAB red mask
            spatial_concentration - fraction of red pixels in rightmost 40%
            red_coverage_width    - fraction of columns containing any red
            mean_s, mean_v        - global HSV statistics
            median_a              - LAB a* channel median (image baseline)
            std_a                 - LAB a* channel std dev
            red_column_density    - 1D array of red pixel count per column
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    mean_s = float(hsv[:, :, 1].mean())
    mean_v = float(hsv[:, :, 2].mean())

    a_ch = lab[:, :, 1].astype(np.float32)
    median_a = float(np.median(a_ch))
    std_a = float(a_ch.std())

    # LAB relative redness mask
    red_mask = create_red_mask(image)

    total_pixels = h * w
    red_pixels = int(cv2.countNonZero(red_mask))
    red_pixel_pct = (red_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

    # Per-column red pixel density
    red_column_density = (red_mask > 0).sum(axis=0).astype(np.float64)

    # Coverage: fraction of columns with at least one red pixel
    cols_with_red = int((red_column_density > 0).sum())
    red_coverage_width = cols_with_red / w if w > 0 else 0.0

    # Spatial concentration: fraction of red pixels in rightmost 40%
    right_start = int(w * 0.60)
    right_red = int(red_mask[:, right_start:].astype(bool).sum())
    spatial_concentration = (right_red / red_pixels) if red_pixels > 0 else 0.0

    return {
        "red_pixel_pct": red_pixel_pct,
        "spatial_concentration": spatial_concentration,
        "red_coverage_width": red_coverage_width,
        "mean_s": mean_s,
        "mean_v": mean_v,
        "median_a": median_a,
        "std_a": std_a,
        "red_column_density": red_column_density,
    }


# ---------------------------------------------------------------------------
# Core mask creation
# ---------------------------------------------------------------------------


def create_red_mask(image: np.ndarray) -> np.ndarray:
    """Create a red mask using LAB a* relative redness + HSV hue gating.

    Two-stage detection:
      1. LAB a* channel: pixels whose a* exceeds the image median by more
         than ``_LAB_A_THRESHOLD`` are flagged. This is robust to overall
         color casts because it measures *relative* redness.
      2. HSV hue gate: the LAB mask is intersected with a red-to-magenta
         hue range plus a minimum saturation gate to exclude neutral regions
         that happen to sit slightly above the median.

    Returns:
        Binary mask (uint8, 0 or 255) - before dilation.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Stage 1: LAB relative redness
    a_ch = lab[:, :, 1].astype(np.float32)
    median_a = np.median(a_ch)
    relative_a = a_ch - median_a
    lab_mask = (relative_a > _LAB_A_THRESHOLD).astype(np.uint8) * 255

    # Stage 2: HSV hue gate - keep red / magenta pixels and a narrow fallback
    # band for cold purple decimals seen in a few samples after orientation.
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    hue_gate = (
        (
            ((h_ch <= _HSV_LOW_RED_MAX) | (h_ch >= _HSV_HIGH_RED_MIN))
            & (s_ch >= _HSV_MIN_SATURATION)
        )
        | (
            (h_ch >= _HSV_PURPLE_FALLBACK_MIN)
            & (s_ch >= _HSV_PURPLE_FALLBACK_MIN_SATURATION)
            & (relative_a > _LAB_A_PURPLE_FALLBACK_THRESHOLD)
        )
    ).astype(np.uint8) * 255

    # Intersection
    return cv2.bitwise_and(lab_mask, hue_gate)


# ---------------------------------------------------------------------------
# Erasure (TELEA inpainting + selective a* suppression)
# ---------------------------------------------------------------------------


def _erase_masked_region(
    image: np.ndarray,
    dilated_mask: np.ndarray,
) -> np.ndarray:
    """Replace masked pixels using TELEA inpainting, then suppress residual
    redness in a soft fringe zone around the mask.

    Stage 1 - TELEA inpainting fills masked pixels from boundary texture.
    Stage 2 - excess redness (LAB a* above median) is suppressed in a
    gaussian-blurred zone around the mask, eliminating color fringes
    without introducing green artifacts or grey smudges.

    Args:
        image:        BGR image (numpy array).
        dilated_mask: Binary mask (uint8, 0/255) of pixels to erase.

    Returns:
        BGR image with masked regions filled.
    """
    kernel = np.ones((3, 3), np.uint8)

    # 1. TELEA inpainting - precise, texture-preserving fill
    inpainted = cv2.inpaint(image, dilated_mask, _INPAINT_RADIUS, cv2.INPAINT_TELEA)

    # 2. Selective a* suppression in fringe zone
    fringe_zone = cv2.dilate(dilated_mask, kernel, iterations=_FRINGE_DILATE)
    spatial_w = fringe_zone.astype(np.float32) / 255.0
    spatial_w = cv2.GaussianBlur(
        spatial_w, (_FRINGE_BLUR_KSIZE, _FRINGE_BLUR_KSIZE), 0,
    )

    lab = cv2.cvtColor(inpainted, cv2.COLOR_BGR2LAB)
    a_ch = lab[:, :, 1].astype(np.float32)
    median_a = np.median(a_ch)

    # Only suppress *excess* redness (a* above median) - never push below
    excess = np.maximum(a_ch - median_a, 0)
    a_ch -= excess * spatial_w * _FRINGE_SUPPRESSION
    lab[:, :, 1] = np.clip(a_ch, 0, 255).astype(np.uint8)

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def apply_red_mask(
    image: np.ndarray,
    include_stats: bool = True,
) -> MaskingResult:
    """Detect red pixels and erase them using TELEA inpainting.

    Uses the LAB relative-redness algorithm which handles all image types
    including reddish color casts.

    Args:
        image:  BGR image (numpy array).
        include_stats: Whether to compute expensive debug statistics used by
            analysis tools. Disable in latency-sensitive inference paths.

    Returns:
        MaskingResult with the output image, masks, and decision metadata.
    """
    raw_mask = create_red_mask(image)
    stats = analyze_red_distribution(image) if include_stats else {}

    kernel = np.ones((3, 3), np.uint8)
    close_kernel = np.ones((_CLOSE_KSIZE, _CLOSE_KSIZE), np.uint8)
    closed_mask = cv2.morphologyEx(
        raw_mask, cv2.MORPH_CLOSE, close_kernel, iterations=_CLOSE_ITERATIONS,
    )
    dilated_mask = cv2.dilate(closed_mask, kernel, iterations=_DILATE_ITERATIONS)
    result = _erase_masked_region(image, dilated_mask)

    if include_stats:
        red_pct = stats["red_pixel_pct"]
        conc = stats["spatial_concentration"]
        reason = (
            f"LAB relative redness "
            f"(median_a={stats['median_a']:.0f}, "
            f"red={red_pct:.1f}%, "
            f"concentration={conc:.0%})"
        )
    else:
        reason = "LAB relative redness (stats disabled for fast inference)"

    return MaskingResult(
        image=result,
        raw_mask=raw_mask,
        dilated_mask=dilated_mask,
        action="applied",
        reason=reason,
        stats=stats,
    )
