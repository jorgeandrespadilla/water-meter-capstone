"""FastAPI REST API for the water meter reading pipeline.

Exposes the full inference pipeline over HTTP:
    POST /predict       — single image → reading
    POST /predict/batch — multiple images → list of readings
    GET  /health        — readiness check

Usage:
    python -m app.api
    # or
    uvicorn app.api:app --host 0.0.0.0 --port 8000
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline import WaterMeterPipeline

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ReadingResponse(BaseModel):
    image_path: str
    reading: str | None
    global_confidence: float
    orientation_confidence: float
    orientation_method: str
    status: str
    detection_confidence: float
    recognition_confidence: float
    timing_ms: dict
    warnings: list[str]


class BatchResponse(BaseModel):
    results: list[ReadingResponse]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline on startup, release on shutdown."""
    print("Initializing pipeline...", flush=True)
    app.state.pipeline = WaterMeterPipeline()
    print("Pipeline ready.", flush=True)
    yield


app = FastAPI(
    title="Water Meter Reading API",
    description="End-to-end water meter odometer reading: YOLO OBB detection + PaddleOCR.",
    version="1.0.0",
    lifespan=lifespan,
)


def _decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode raw bytes to a BGR numpy array."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image file.")
    return image


def _result_to_response(result, filename: str) -> ReadingResponse:
    return ReadingResponse(
        image_path=filename,
        reading=result.reading,
        global_confidence=result.global_confidence,
        orientation_confidence=result.orientation_confidence,
        orientation_method=result.orientation_method,
        status=result.status,
        detection_confidence=result.detection_confidence,
        recognition_confidence=result.recognition_confidence,
        timing_ms=result.timing_ms,
        warnings=result.warnings,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health():
    """Readiness check."""
    loaded = hasattr(app.state, "pipeline") and app.state.pipeline is not None
    return HealthResponse(status="healthy", models_loaded=loaded)


@app.post("/predict", response_model=ReadingResponse)
def predict(
    file: UploadFile = File(...),
    validation_threshold: float = Query(0.7, ge=0.0, le=1.0),
):
    """Read a water meter from a single uploaded image."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix and suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}",
        )

    try:
        file_bytes = file.file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error reading file: {exc}")

    image = _decode_image(file_bytes)

    try:
        result = app.state.pipeline.predict(image, validation_threshold=validation_threshold)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    return _result_to_response(result, file.filename or "unknown")


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(
    files: list[UploadFile] = File(...),
    validation_threshold: float = Query(0.7, ge=0.0, le=1.0),
):
    """Read water meters from multiple uploaded images."""
    results = []
    for f in files:
        try:
            file_bytes = f.file.read()
            image = _decode_image(file_bytes)
            result = app.state.pipeline.predict(image, validation_threshold=validation_threshold)
            results.append(_result_to_response(result, f.filename or "unknown"))
        except HTTPException:
            raise
        except Exception as exc:
            results.append(ReadingResponse(
                image_path=f.filename or "unknown",
                reading=None,
                global_confidence=0.0,
                orientation_confidence=0.0,
                orientation_method="none",
                status="needs_review",
                detection_confidence=0.0,
                recognition_confidence=0.0,
                timing_ms={},
                warnings=[f"Processing error: {exc}"],
            ))
    return BatchResponse(results=results)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=False)
