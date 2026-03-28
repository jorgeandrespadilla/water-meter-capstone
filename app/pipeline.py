"""End-to-end water meter reading pipeline.

Combines YOLO OBB odometer detection with PaddleOCR digit recognition
to produce a reading from a raw water meter image.

Pipeline flow:
    Raw image → YOLO OBB detect → crop → hybrid orientation (ensure
    landscape → red-position cue → dual-OCR fallback) → color masking
    (LAB v4) → sanitize → output

Usage:
    from app.pipeline import WaterMeterPipeline

    pipeline = WaterMeterPipeline()
    result = pipeline.predict(bgr_image)
    print(result.reading, result.status, result.global_confidence)
"""

import re
import sys
import time
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from utils.cropping import OBB_CROP_PADDING, crop_obb_from_xywhr
from utils.orientation import resolve_orientation

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_YOLO_PATH = (
    PROJECT_ROOT / "models" / "odometer-detector" / "best.pt"
)
DEFAULT_OCR_MODEL_DIR = (
    PROJECT_ROOT / "models" / "ocr-reader" / "model"
)
DEFAULT_OCR_MODEL_NAME = "en_PP-OCRv4_mobile_rec"

_DIGITS_ONLY = re.compile(r"\D")

# YOLO detection confidence -detections below this are discarded
_YOLO_CONF_THRESHOLD = 0.5

# Default global validation threshold for auto-validation
_DEFAULT_VALIDATION_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Structured result from the water meter reading pipeline."""

    reading: str | None
    """Sanitized digit string (e.g. '09066') or None if unreadable."""

    global_confidence: float
    """detection_conf × recognition_conf (orientation excluded)."""

    orientation_confidence: float
    """Red-cue offset (red_position) or OCR conf delta (dual_ocr)."""

    orientation_method: str
    """How orientation was resolved: 'red_position' or 'dual_ocr'."""

    status: str
    """'valid' | 'needs_review' | 'no_detection'."""

    detection_confidence: float
    recognition_confidence: float

    timing_ms: dict = field(default_factory=dict)
    """Per-stage timing: total, detection, preprocessing, ocr."""

    intermediates: dict = field(default_factory=dict)
    """Debug images: annotated, crop, oriented, masked (BGR np.ndarray)."""

    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class WaterMeterPipeline:
    """Full inference pipeline: YOLO OBB detection → preprocessing → OCR."""

    def __init__(
        self,
        yolo_model_path: Path | str | None = None,
        ocr_model_dir: Path | str | None = None,
        ocr_model_name: str = DEFAULT_OCR_MODEL_NAME,
    ):

        yolo_path = Path(yolo_model_path) if yolo_model_path else DEFAULT_YOLO_PATH
        ocr_dir = Path(ocr_model_dir) if ocr_model_dir else DEFAULT_OCR_MODEL_DIR

        # --- Load YOLO OBB detector ---
        from ultralytics import YOLO

        print(f"Loading YOLO model: {yolo_path}", flush=True)
        self._yolo = YOLO(str(yolo_path))
        # Warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._yolo.predict(dummy, imgsz=640, verbose=False)
        print("YOLO model ready.", flush=True)

        # --- Load PaddleOCR rec-only model ---
        from paddlex import create_model

        print(f"Loading OCR model: {ocr_model_name} from {ocr_dir}", flush=True)
        self._rec = create_model(
            model_name=ocr_model_name,
            model_dir=str(ocr_dir),
        )
        # Warmup the OCR recognizer once so the first user request does not
        # pay the full backend initialization cost.
        self._run_rec(np.zeros((64, 256, 3), dtype=np.uint8))
        print("OCR model ready.", flush=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        image: np.ndarray,
        validation_threshold: float = _DEFAULT_VALIDATION_THRESHOLD,
        collect_intermediates: bool = False,
    ) -> PipelineResult:
        """Run the full pipeline on a BGR image.

        Args:
            image: BGR numpy array (OpenCV convention).
            validation_threshold: Minimum global confidence (det × rec) to
                mark a reading as 'valid'. Below this → 'needs_review'.
            collect_intermediates: Whether to generate debug images for each
                stage. Disable this in serving paths that do not display them.

        Returns:
            PipelineResult with reading, confidences, status, and debug images.
        """
        warnings: list[str] = []
        intermediates: dict = {}
        t_start = time.perf_counter()

        # 1. Detection --------------------------------------------------------
        t0 = time.perf_counter()
        results = self._yolo.predict(image, conf=_YOLO_CONF_THRESHOLD, imgsz=640, verbose=False)
        t_detection = time.perf_counter() - t0

        obb = results[0].obb
        if obb is None or len(obb) == 0:
            return PipelineResult(
                reading=None,
                global_confidence=0.0,
                orientation_confidence=0.0,
                orientation_method="none",
                status="no_detection",
                detection_confidence=0.0,
                recognition_confidence=0.0,
                timing_ms={"total": _ms(t_start), "detection": t_detection * 1000},
                intermediates={},
                warnings=warnings,
            )

        # 2. Select best detection (highest confidence) -----------------------
        n_dets = len(obb)
        if n_dets > 1:
            warnings.append(
                f"Multiple detections ({n_dets}), using highest confidence"
            )

        best_idx = int(obb.conf.argmax().cpu())
        det_conf = float(obb.conf[best_idx].cpu().numpy())
        xywhr = obb.xywhr[best_idx].cpu().numpy()
        corners = obb.xyxyxyxy[best_idx].cpu().numpy()

        # Annotated image for debug
        if collect_intermediates:
            intermediates["annotated"] = _draw_detection(image, corners, det_conf)

        # 3. OBB Crop ---------------------------------------------------------
        t0 = time.perf_counter()
        crop = crop_obb_from_xywhr(
            image, xywhr, padding=OBB_CROP_PADDING,
        )
        if crop is None or crop.size == 0:
            return PipelineResult(
                reading=None,
                global_confidence=0.0,
                orientation_confidence=0.0,
                orientation_method="none",
                status="needs_review",
                detection_confidence=det_conf,
                recognition_confidence=0.0,
                timing_ms={"total": _ms(t_start), "detection": t_detection * 1000},
                intermediates=intermediates,
                warnings=warnings + ["OBB crop failed (degenerate region)"],
            )

        if collect_intermediates:
            intermediates["crop"] = crop

        # 4. Hybrid orientation (landscape → red-position → dual-OCR) ----------
        ori = resolve_orientation(crop, self._run_rec)
        orientation_conf = ori.confidence
        if collect_intermediates:
            intermediates["oriented"] = ori.oriented
            intermediates["masked"] = ori.masked
        t_preprocess = time.perf_counter() - t0

        if ori.low_confidence:
            warnings.append(
                f"Orientation ambiguous (method={ori.method}, delta={ori.confidence:.4f})"
            )

        # 5. OCR rec-only -----------------------------------------------------
        t0 = time.perf_counter()
        if ori.method == "dual_ocr":
            # OCR already ran during orientation -reuse results
            raw_text, rec_conf = ori.raw_text, ori.recognition_confidence
        else:
            # Red-position resolved orientation without OCR -run it now
            raw_text, rec_conf = self._run_rec(ori.masked)
        t_ocr = time.perf_counter() - t0

        # Adjust timing: move OCR time from preprocessing to the OCR bucket
        t_ocr += ori.ocr_elapsed_s
        t_preprocess -= ori.ocr_elapsed_s

        # 6. Sanitization -----------------------------------------------------
        reading = _digits_only(raw_text) if raw_text else None
        if reading == "":
            reading = None

        # 7. Compute confidence and status ------------------------------------
        rec_conf_final = rec_conf if reading else 0.0
        global_conf = det_conf * rec_conf_final

        if reading and global_conf >= validation_threshold:
            status = "valid"
        else:
            status = "needs_review"

        return PipelineResult(
            reading=reading,
            global_confidence=round(global_conf, 4),
            orientation_confidence=round(orientation_conf, 4),
            orientation_method=ori.method,
            status=status,
            detection_confidence=round(det_conf, 4),
            recognition_confidence=round(rec_conf_final, 4),
            timing_ms={
                "total": _ms(t_start),
                "detection": round(t_detection * 1000, 1),
                "preprocessing": round(t_preprocess * 1000, 1),
                "ocr": round(t_ocr * 1000, 1),
                "orientation_ocr": round(ori.ocr_elapsed_s * 1000, 1),
                "ocr_passes": 2 if ori.method == "dual_ocr" else (1 if reading else 0),
            },
            intermediates=intermediates,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_rec(self, image_bgr: np.ndarray) -> tuple[str, float]:
        """Run PaddleOCR recognition on a BGR image.

        Returns:
            (raw_text, confidence) tuple.
        """
        output = self._rec.predict(input=image_bgr, batch_size=1)

        for res in output:
            rec_text = ""
            rec_score = 0.0

            # Handle nested result format (paddlex versions vary)
            if hasattr(res, "json"):
                data = res.json
                if "res" in data:
                    rec_text = data["res"].get("rec_text", "")
                    rec_score = data["res"].get("rec_score", 0.0)
                else:
                    rec_text = data.get("rec_text", "")
                    rec_score = data.get("rec_score", 0.0)
            elif isinstance(res, dict):
                rec_text = res.get("rec_text", "")
                rec_score = res.get("rec_score", 0.0)
            else:
                rec_text = getattr(res, "rec_text", "")
                rec_score = getattr(res, "rec_score", 0.0)

            return str(rec_text), float(rec_score)

        return "", 0.0


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _digits_only(text: str) -> str:
    """Strip all non-digit characters."""
    return _DIGITS_ONLY.sub("", text)


def _ms(t_start: float) -> float:
    """Milliseconds elapsed since t_start."""
    return round((time.perf_counter() - t_start) * 1000, 1)


def _draw_detection(
    image: np.ndarray, corners: np.ndarray, confidence: float,
) -> np.ndarray:
    """Draw a single OBB detection on a copy of the image."""
    vis = image.copy()
    pts = corners.astype(np.int32)

    # Semi-transparent fill
    overlay = vis.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 0))
    vis = cv2.addWeighted(overlay, 0.2, vis, 0.8, 0)

    # Outline + corner dots
    cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
    for corner in corners:
        cv2.circle(vis, (int(corner[0]), int(corner[1])), 5, (255, 0, 0), -1)

    # Label
    label = f"Odometer: {confidence:.1%}"
    label_pos = (int(corners[:, 0].min()), int(corners[:, 1].min()) - 10)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(
        vis,
        (label_pos[0], label_pos[1] - th - 5),
        (label_pos[0] + tw, label_pos[1] + 5),
        (0, 255, 0), -1,
    )
    cv2.putText(
        vis, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
    )
    return vis
