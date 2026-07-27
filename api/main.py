import io
import time
import uuid
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from camscan.boundary import yolo26_doccorner_pose
from camscan.enhance import DEFAULT_STRENGTH
from camscan.pipeline import DEFAULT_METHOD, SUPPORTED_FORMATS, detect_boundary, save_output, warp_and_enhance
from camscan.preprocess import resize_for_detection

app = FastAPI(title="CamScanner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: this is a demo deployment (single process, no auth), not a
# production service, so a dict keyed by session id is enough -- no database needed.
# Each session holds everything needed to re-run warp_and_enhance as the user tweaks
# the crop/enhancement without re-uploading or re-running detection. Typed as
# dict[str, Any] rather than a stricter TypedDict/dataclass -- the value shape (numpy
# arrays, a float ratio, per-doc-index result cache) doesn't cross a serialization
# boundary the way the Pydantic models below do, so a precise type here wouldn't be
# checked against anything external and isn't worth the extra ceremony.
_SESSIONS: dict[str, dict[str, Any]] = {}
_SESSIONS_LOCK = Lock()
SESSION_TTL_SECONDS = 30 * 60


def _prune_sessions() -> None:
    cutoff = time.time() - SESSION_TTL_SECONDS
    expired = [sid for sid, s in _SESSIONS.items() if s["created_at"] < cutoff]
    for sid in expired:
        del _SESSIONS[sid]


def _get_session(session_id: str) -> dict[str, Any]:
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return session


class DetectResponse(BaseModel):
    session_id: str
    quad: list[list[float]]
    used_fallback: bool
    resized_width: int
    resized_height: int


class DetectBatchResponse(BaseModel):
    session_id: str
    quads: list[list[list[float]]]
    resized_width: int
    resized_height: int


class EnhanceRequest(BaseModel):
    session_id: str
    quad: list[list[float]] | None = None
    enhance_mode: str = "auto"
    enhance_strength: int = DEFAULT_STRENGTH
    manual_rotation: int = 0
    doc_index: int = 0


@app.post("/api/detect", response_model=DetectResponse)
async def detect_endpoint(file: UploadFile = File(...)) -> DetectResponse:
    data = await file.read()
    np_arr = np.frombuffer(data, dtype=np.uint8)
    original = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if original is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    resized, ratio = resize_for_detection(original)
    quad, used_fallback = detect_boundary(resized, method=DEFAULT_METHOD, original_image=original)

    with _SESSIONS_LOCK:
        _prune_sessions()
        session_id = uuid.uuid4().hex
        _SESSIONS[session_id] = {
            "original": original,
            "resized": resized,
            "ratio": ratio,
            "created_at": time.time(),
            "last_results": {},
        }

    h, w = resized.shape[:2]
    return DetectResponse(
        session_id=session_id,
        quad=quad.tolist(),
        used_fallback=used_fallback,
        resized_width=w,
        resized_height=h,
    )


@app.post("/api/detect_batch", response_model=DetectBatchResponse)
async def detect_batch_endpoint(file: UploadFile = File(...)) -> DetectBatchResponse:
    """Batch variant of /api/detect: returns every document quad found in one photo
    instead of a single best quad, for the experimental multi-document scan flow --
    see camscan.boundary.yolo26_doccorner_pose.find_document_contours' docstring for
    why this is flagged unverified rather than a trusted feature."""
    data = await file.read()
    np_arr = np.frombuffer(data, dtype=np.uint8)
    original = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if original is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    resized, ratio = resize_for_detection(original)
    quads = yolo26_doccorner_pose.find_document_contours(resized, detection_image=original)

    with _SESSIONS_LOCK:
        _prune_sessions()
        session_id = uuid.uuid4().hex
        _SESSIONS[session_id] = {
            "original": original,
            "resized": resized,
            "ratio": ratio,
            "created_at": time.time(),
            "last_results": {},
        }

    h, w = resized.shape[:2]
    return DetectBatchResponse(
        session_id=session_id,
        quads=[q.tolist() for q in quads],
        resized_width=w,
        resized_height=h,
    )


@app.get("/api/preview/{session_id}")
def preview_endpoint(session_id: str) -> StreamingResponse:
    """Serves the resized detection-space image the quad coordinates are relative
    to, so the frontend's crop editor can overlay draggable handles on the exact
    image the backend used for detection."""
    session = _get_session(session_id)
    ok, buf = cv2.imencode(".jpg", session["resized"])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode preview")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/jpeg")


@app.post("/api/enhance")
def enhance_endpoint(req: EnhanceRequest) -> StreamingResponse:
    session = _get_session(req.session_id)

    if req.quad is None:
        raise HTTPException(status_code=400, detail="quad is required")
    quad = np.array(req.quad, dtype=np.float32)
    if quad.shape != (4, 2):
        raise HTTPException(status_code=400, detail="quad must be a 4x2 array")

    if req.enhance_mode not in ("auto", "gray", "bw", "color"):
        raise HTTPException(status_code=400, detail="enhance_mode must be auto, gray, bw, or color")
    if not 0 <= req.enhance_strength <= 100:
        raise HTTPException(status_code=400, detail="enhance_strength must be 0-100")
    if req.manual_rotation not in (0, 90, 180, 270):
        raise HTTPException(status_code=400, detail="manual_rotation must be 0, 90, 180, or 270")

    result = warp_and_enhance(
        session["original"], quad, session["ratio"],
        enhance_mode=req.enhance_mode, enhance_strength=req.enhance_strength,
        manual_rotation=req.manual_rotation,
    )
    # Keyed by doc_index (default 0) so a batch-scan session can hold one enhanced
    # result per detected document independently, while the single-document flow
    # (which never sends doc_index) transparently keeps using slot 0.
    session["last_results"][req.doc_index] = result

    ok, buf = cv2.imencode(".png", result)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode preview")
    return StreamingResponse(io.BytesIO(buf.tobytes()), media_type="image/png")


@app.get("/api/export/{session_id}")
def export_endpoint(session_id: str, format: str = "png", doc_index: int = 0) -> StreamingResponse:
    session = _get_session(session_id)
    result = session["last_results"].get(doc_index)
    if result is None:
        raise HTTPException(status_code=400, detail="Call /api/enhance before exporting")

    fmt = format.lower().lstrip(".")
    if fmt not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported format, choose from {sorted(SUPPORTED_FORMATS)}")

    tmp_path = Path(f"/tmp/camscan_export_{session_id}_{doc_index}.{fmt}")
    save_output(result, tmp_path)
    data = tmp_path.read_bytes()
    tmp_path.unlink(missing_ok=True)

    media_types = {
        "pdf": "application/pdf", "png": "image/png", "jpg": "image/jpeg",
        "jpeg": "image/jpeg", "tiff": "image/tiff", "webp": "image/webp",
    }
    return StreamingResponse(
        io.BytesIO(data),
        media_type=media_types.get(fmt, "application/octet-stream"),
        headers={"Content-Disposition": f"attachment; filename=scan.{fmt}"},
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
