from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

from mlops_group54_project.model import ModelConfig, build_model


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


app = FastAPI(title="Brain Tumor Inference API")

DEVICE = _device()
TRANSFORM = _build_infer_transform(image_size=224)

CHECKPOINT_PATH = Path("models/model.pth")
CLASS_MAPPING_PATH = Path("data/processed/class_mapping.pt")

# These will be set on startup (or remain None if files are missing)
MODEL: Optional[torch.nn.Module] = None
CLASS_NAMES: List[str] = []


@app.on_event("startup")
def _load_artifacts() -> None:
    """
    Load class mapping + model weights at server startup.
    If files are missing (common in CI), keep MODEL=None and CLASS_NAMES=[].
    """
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


@app.get("/")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "checkpoint": str(CHECKPOINT_PATH),
        "model_loaded": MODEL is not None,
        "num_classes": len(CLASS_NAMES),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    if MODEL is None or not CLASS_NAMES:
        raise HTTPException(status_code=503, detail="Model not loaded. Check checkpoint and class_mapping paths.")

    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image")

    try:
        img = Image.open(file.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred = int(torch.argmax(probs).item())

    return {
        "pred_class": pred,
        "pred_label": CLASS_NAMES[pred],
        "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))},
    }
