from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from torchvision import transforms
from PIL import Image

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

# Defaults (you can later connect these to Hydra config)
CHECKPOINT_PATH = Path("models/model.pth")
NUM_CLASSES = 4

# Load model once at startup
MODEL = build_model(ModelConfig(backbone="resnet50", pretrained=False, num_classes=NUM_CLASSES)).to(DEVICE)
MODEL.eval()

if not CHECKPOINT_PATH.exists():
    # It's okay to raise here so you immediately see what's wrong when starting the server
    raise RuntimeError(f"Checkpoint not found at {CHECKPOINT_PATH.resolve()}")

state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
MODEL.load_state_dict(state)


@app.get("/")
def health() -> Dict[str, Any]:
    return {"status": "ok", "device": str(DEVICE), "checkpoint": str(CHECKPOINT_PATH)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    # Basic content-type check
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image")

    try:
        img = Image.open(file.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]

    with torch.no_grad():
        logits = MODEL(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)  # [C]
        pred = int(torch.argmax(probs).item())

    return {
        "pred_class": pred,
        "probs": [float(p) for p in probs.tolist()],
    }
