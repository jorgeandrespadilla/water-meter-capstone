"""Utilities for cropping oriented bounding box (OBB) regions from images."""

from typing import Literal

import cv2
import numpy as np

# Default padding (pixels) added around OBB crops.
# Used by both the data-preparation script (02_crop_odometers.py) and at inference
# time (demo_detector.py) to keep train/inference consistency.
# Iter 5 tested padding=5 but it worsened EM (0.85 vs 0.90), so default is 0.
OBB_CROP_PADDING = 0


def _rotate_and_crop(
    image: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    angle_deg: float,
    padding: int,
    ensure_horizontal: bool = False,
) -> np.ndarray | None:
    """Core: rotate image to align OBB and crop the axis-aligned rectangle.

    Args:
        image: Source image (H, W, C).
        cx, cy: Center of the OBB in pixel coordinates.
        w, h: Width and height of the OBB.
        angle_deg: Rotation angle in degrees (OpenCV convention).
        padding: Extra pixels to add around the crop on each side.
        ensure_horizontal: If True, rotate the crop so width >= height.

    Returns:
        Cropped image or None if the crop is degenerate.
    """
    # Ensure width >= height so the angle aligns the longer axis
    if w < h:
        w, h = h, w
        angle_deg += 90

    if w < 5 or h < 5:
        return None

    w += 2 * padding
    h += 2 * padding

    img_h, img_w = image.shape[:2]

    # Rotate the entire image around the OBB center
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    # Expand canvas so nothing is clipped after rotation
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(img_h * sin_a + img_w * cos_a)
    new_h = int(img_h * cos_a + img_w * sin_a)
    M[0, 2] += (new_w - img_w) / 2
    M[1, 2] += (new_h - img_h) / 2

    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Transform OBB center through the same rotation matrix
    new_cx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
    new_cy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]

    # Crop the now axis-aligned rectangle
    x1 = max(0, int(new_cx - w / 2))
    y1 = max(0, int(new_cy - h / 2))
    x2 = min(rotated.shape[1], int(new_cx + w / 2))
    y2 = min(rotated.shape[0], int(new_cy + h / 2))

    crop = rotated[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    if ensure_horizontal and crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

    return crop


def crop_obb(
    image: np.ndarray,
    points: np.ndarray,
    *,
    padding: int = 0,
    ensure_horizontal: bool = False,
    coord_format: Literal["pixel", "normalized"] = "pixel",
) -> np.ndarray | None:
    """Crop an OBB region given its 4 corner points.

    Args:
        image: Source image (H, W, C).
        points: Corner points as a (4, 2) array.
        padding: Extra pixels to add around the crop on each side.
        ensure_horizontal: If True, rotate the crop so width >= height.
        coord_format: ``"pixel"`` if points are in pixel coordinates,
            ``"normalized"`` if in [0, 1] range (scaled to image size).

    Returns:
        Cropped image or None if the crop is degenerate (< 5 px).
    """
    img_h, img_w = image.shape[:2]

    pixel_points = points.astype(np.float32).copy()
    if coord_format == "normalized":
        pixel_points[:, 0] *= img_w
        pixel_points[:, 1] *= img_h

    # Fit a minimum-area rotated rectangle to the 4 corners
    (cx, cy), (w, h), angle = cv2.minAreaRect(pixel_points)

    return _rotate_and_crop(image, cx, cy, w, h, angle, padding, ensure_horizontal)


def crop_obb_from_xywhr(
    image: np.ndarray,
    xywhr: np.ndarray,
    *,
    padding: int = 0,
    ensure_horizontal: bool = False,
) -> np.ndarray | None:
    """Crop an OBB from YOLO's xywhr format (cx, cy, w, h, angle_rad).

    Convenience wrapper for use with ``obb.xywhr`` tensors from Ultralytics.

    Args:
        image: Source image (H, W, C).
        xywhr: Array of [cx, cy, w, h, angle_rad] in pixel coordinates.
        padding: Extra pixels to add around the crop on each side.
        ensure_horizontal: If True, rotate the crop so width >= height.

    Returns:
        Cropped image or None if the crop is degenerate.
    """
    cx, cy, w, h, angle_rad = xywhr
    angle_deg = float(np.degrees(angle_rad))

    return _rotate_and_crop(
        image, float(cx), float(cy), float(w), float(h),
        angle_deg, padding, ensure_horizontal,
    )
