from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import csv
import hashlib
import json
import os
import time

import numpy as np
import torch
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from google.cloud import storage
from PIL import Image
from torchvision import transforms

from mlops_group54_project.model import ModelConfig, build_model


# ----------------------------
# Device + preprocessing
# ----------------------------
def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_infer_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _image_quality_features(img: Image.Image) -> tuple[float, float, float]:
    """
    Engineered image-quality features for drift monitoring:
    - brightness: mean grayscale intensity (0..255)
    - contrast: std grayscale intensity
    - sharpness: variance of Laplacian (focus measure)
    """
    gray = np.array(img.convert("L"), dtype=np.float32)  # [H,W] in 0..255
    brightness = float(gray.mean())
    contrast = float(gray.std())

    # Laplacian variance (simple focus metric)
    lap = (
        -4 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    sharpness = float(lap.var())
    return brightness, contrast, sharpness


# ----------------------------
# App + globals
# ----------------------------
app = FastAPI(title="Brain Tumor Inference API")

DEVICE = _device()
TRANSFORM = _build_infer_transform(image_size=224)

CHECKPOINT_PATH = Path("models/model.pth")
CLASS_MAPPING_PATH = Path("data/processed/class_mapping.pt")

MODEL: Optional[torch.nn.Module] = None
CLASS_NAMES: List[str] = []


# ----------------------------
# Monitoring config (Cloud Run friendly)
# ----------------------------
# If MONITORING_BUCKET is set, we log each request as a JSON object to GCS.
# Otherwise we fall back to local CSV (useful for local dev / VM).
MONITORING_BUCKET = os.getenv("MONITORING_BUCKET", "").strip()
MONITORING_PREFIX = os.getenv("MONITORING_PREFIX", "monitoring/requests").strip()

# Local fallback log (ephemeral on Cloud Run, persistent on VM with a mounted volume)
REQUEST_LOG_PATH = Path("data/monitoring/requests.csv")
REQUEST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

CSV_HEADER = [
    "time_utc",
    "image_sha256",
    "filename",
    "content_type",
    "pred_idx",
    "pred_label",
    "confidence",
    "latency_ms",
    # post-transform (normalized) per-channel statistics
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    # engineered “image quality” features (raw image)
    "brightness",
    "contrast",
    "sharpness",
]


def _append_csv_row(path: Path, header: list[str], row: list[Any]) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow(row)


_gcs_client: storage.Client | None = None


def _get_gcs_client() -> storage.Client:
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    return _gcs_client


def _upload_json_to_gcs(*, bucket_name: str, object_name: str, payload: dict) -> None:
    client = _get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_string(
        data=json.dumps(payload, ensure_ascii=False),
        content_type="application/json",
    )


def _log_request(
    *,
    image_bytes: bytes,
    filename: str,
    content_type: str,
    pred_idx: int,
    pred_label: str,
    confidence: float,
    latency_ms: float,
    ch_mean: list[float],
    ch_std: list[float],
    brightness: float,
    contrast: float,
    sharpness: float,
) -> None:
    """
    Logs a single prediction event.

    Cloud Run mode (preferred): one JSON object per request to GCS:
      gs://MONITORING_BUCKET/MONITORING_PREFIX/YYYY/MM/DD/<timestamp>_<sha>.json

    Local/dev fallback: append to data/monitoring/requests.csv
    """
    ts = datetime.now(timezone.utc).isoformat()
    sha = hashlib.sha256(image_bytes).hexdigest()

    payload = {
        "time_utc": ts,
        "image_sha256": sha,
        "filename": filename,
        "content_type": content_type,
        "pred_idx": int(pred_idx),
        "pred_label": pred_label,
        "confidence": float(confidence),
        "latency_ms": float(latency_ms),
        "mean_r": float(ch_mean[0]),
        "mean_g": float(ch_mean[1]),
        "mean_b": float(ch_mean[2]),
        "std_r": float(ch_std[0]),
        "std_g": float(ch_std[1]),
        "std_b": float(ch_std[2]),
        "brightness": float(brightness),
        "contrast": float(contrast),
        "sharpness": float(sharpness),
    }

    if MONITORING_BUCKET:
        # Create stable, date-partitioned object names
        date = ts[:10]  # YYYY-MM-DD
        yyyy, mm, dd = date.split("-")
        safe_ts = ts.replace(":", "").replace(".", "")
        object_name = f"{MONITORING_PREFIX}/{yyyy}/{mm}/{dd}/{safe_ts}_{sha}.json"

        _upload_json_to_gcs(
            bucket_name=MONITORING_BUCKET,
            object_name=object_name,
            payload=payload,
        )
        return

    # Fallback CSV (local dev / VM)
    _append_csv_row(
        REQUEST_LOG_PATH,
        CSV_HEADER,
        [
            ts,
            sha,
            filename,
            content_type,
            pred_idx,
            pred_label,
            float(confidence),
            float(latency_ms),
            float(ch_mean[0]),
            float(ch_mean[1]),
            float(ch_mean[2]),
            float(ch_std[0]),
            float(ch_std[1]),
            float(ch_std[2]),
            float(brightness),
            float(contrast),
            float(sharpness),
        ],
    )


# ----------------------------
# Metrics (simple Prometheus)
# ----------------------------
# This adds:
# - /metrics endpoint
# - request count / latency / status codes
#
# Add dependency: prometheus-fastapi-instrumentator
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
except Exception:
    # If dependency isn't installed, API still runs.
    pass


# ----------------------------
# Startup: load model + class mapping
# ----------------------------
@app.on_event("startup")
def _load_artifacts() -> None:
    global MODEL, CLASS_NAMES

    if not CLASS_MAPPING_PATH.exists():
        MODEL = None
        CLASS_NAMES = []
        return

    class_mapping = torch.load(CLASS_MAPPING_PATH, map_location="cpu")
    CLASS_NAMES = list(class_mapping["classes"])
    num_classes = len(CLASS_NAMES)

    model = build_model(ModelConfig(backbone="resnet50", pretrained=False, num_classes=num_classes)).to(DEVICE)
    model.eval()

    if not CHECKPOINT_PATH.exists():
        MODEL = None
        return

    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    MODEL = model


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "checkpoint": str(CHECKPOINT_PATH),
        "model_loaded": MODEL is not None,
        "num_classes": len(CLASS_NAMES),
        "monitoring_bucket_set": bool(MONITORING_BUCKET),
        "monitoring_prefix": MONITORING_PREFIX,
    }


@app.post("/predict")
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    if MODEL is None or not CLASS_NAMES:
        raise HTTPException(status_code=503, detail="Model not loaded. Check checkpoint and class_mapping paths.")

    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image")

    try:
        image_bytes = await file.read()
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    # Engineered features on raw image
    brightness, contrast, sharpness = _image_quality_features(img)

    # Post-transform features (normalized space)
    x_cpu = TRANSFORM(img)  # [3,224,224] on CPU
    ch_mean = x_cpu.mean(dim=(1, 2)).tolist()
    ch_std = x_cpu.std(dim=(1, 2)).tolist()

    # Inference tensor (same transform, then batch + device)
    x = x_cpu.unsqueeze(0).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred = int(torch.argmax(probs).item())
        confidence = float(probs[pred].item())
    latency_ms = (time.perf_counter() - t0) * 1000.0

    pred_label = CLASS_NAMES[pred]

    # Background logging (does not block response)
    background_tasks.add_task(
        _log_request,
        image_bytes=image_bytes,
        filename=file.filename or "",
        content_type=file.content_type or "",
        pred_idx=pred,
        pred_label=pred_label,
        confidence=confidence,
        latency_ms=latency_ms,
        ch_mean=ch_mean,
        ch_std=ch_std,
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
    )

    return {
        "pred_class": pred,
        "pred_label": pred_label,
        "latency_ms": float(latency_ms),
        "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
    }
