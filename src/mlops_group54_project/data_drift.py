from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import datasets, transforms

from evidently import Report
from evidently.presets import DataDriftPreset

# Keep this consistent with your API
IMAGE_SIZE = 224
TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Paths (adjust to your repo)
REQUESTS_CSV = Path("data/monitoring/requests.csv")
REFERENCE_CSV = Path("data/monitoring/reference.csv")
REPORT_HTML = Path("data/monitoring/drift_report.html")

# Adjust this to wherever your original training images live:
RAW_TRAIN_DIR = Path("data/raw/Training")  # <-- change if your structure differs


def image_quality_features(img: Image.Image) -> tuple[float, float, float]:
    gray = np.array(img.convert("L"), dtype=np.float32)
    brightness = float(gray.mean())
    contrast = float(gray.std())

    lap = (
        -4 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    sharpness = float(lap.var())
    return brightness, contrast, sharpness


def extract_features_for_image(img: Image.Image) -> dict[str, Any]:
    # engineered features on raw image
    brightness, contrast, sharpness = image_quality_features(img)

    # post-transform features (normalized space)
    x = TRANSFORM(img)  # [3,224,224]
    ch_mean = x.mean(dim=(1, 2)).tolist()
    ch_std = x.std(dim=(1, 2)).tolist()

    return {
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


def build_reference_csv(max_images: int = 300) -> pd.DataFrame:
    if not RAW_TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Could not find RAW_TRAIN_DIR={RAW_TRAIN_DIR}. "
            "Update RAW_TRAIN_DIR in data_drift.py to your training image folder."
        )

    ds = datasets.ImageFolder(root=str(RAW_TRAIN_DIR), transform=None)

    num_classes = len(ds.classes)
    max_per_class = max_images // num_classes
    remainder = max_images % num_classes

    # Collect indices per class
    indices_by_class: dict[int, list[int]] = {c: [] for c in range(num_classes)}
    for idx, (_path, label_idx) in enumerate(ds.samples):
        indices_by_class[int(label_idx)].append(idx)

    # Build a balanced list of indices
    selected_indices: list[int] = []
    for c in range(num_classes):
        take = max_per_class + (1 if c < remainder else 0)
        selected_indices.extend(indices_by_class[c][:take])

    rows: list[dict[str, Any]] = []
    for idx in selected_indices:
        path, label_idx = ds.samples[idx]
        img = Image.open(path).convert("RGB")
        feats = extract_features_for_image(img)
        feats["target"] = int(label_idx)  # optional
        feats["class_name"] = ds.classes[int(label_idx)]  # optional, helpful for debugging
        rows.append(feats)

    ref = pd.DataFrame(rows)
    REFERENCE_CSV.parent.mkdir(parents=True, exist_ok=True)
    ref.to_csv(REFERENCE_CSV, index=False)
    return ref



def load_current_csv(tail_n: int = 300) -> pd.DataFrame:
    if not REQUESTS_CSV.exists():
        raise FileNotFoundError(f"Could not find {REQUESTS_CSV}. Run the API and make some requests first.")
    cur = pd.read_csv(REQUESTS_CSV)

    # Keep only columns that exist in reference (plus any you want to monitor separately)
    # For drift, we'll focus on feature columns:
    feature_cols = [
        "mean_r", "mean_g", "mean_b",
        "std_r", "std_g", "std_b",
        "brightness", "contrast", "sharpness",
    ]
    missing = [c for c in feature_cols if c not in cur.columns]
    if missing:
        raise ValueError(
            f"Current CSV is missing columns: {missing}. "
            "Make sure your API is logging these features."
        )

    cur = cur.tail(tail_n).copy()
    return cur[feature_cols]


def run_evidently_report(reference: pd.DataFrame, current: pd.DataFrame) -> None:
    feature_cols = [
        "mean_r", "mean_g", "mean_b",
        "std_r", "std_g", "std_b",
        "brightness", "contrast", "sharpness",
    ]

    # Keep only columns that exist in BOTH (prevents crashes when current is missing cols)
    common = [c for c in feature_cols if c in reference.columns and c in current.columns]
    if not common:
        raise ValueError(
            "No common feature columns between reference and current. "
            "Check that requests.csv contains the engineered features."
        )

    reference = reference[common].copy()
    current = current[common].copy()

    report = Report(metrics=[DataDriftPreset()])

    # IMPORTANT: in some Evidently versions, run() returns a snapshot with export methods
    snapshot = report.run(reference_data=reference, current_data=current)

    REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)

    # Try to save HTML from snapshot first, then fallback to report
    for obj in (snapshot, report):
        if hasattr(obj, "save_html"):
            obj.save_html(str(REPORT_HTML))
            return
        if hasattr(obj, "save"):
            obj.save(str(REPORT_HTML))
            return
        if hasattr(obj, "as_html"):
            html = obj.as_html()
            REPORT_HTML.write_text(html, encoding="utf-8")
            return

    # Last resort: save JSON so you at least get an artifact
    for obj in (snapshot, report):
        if hasattr(obj, "save_json"):
            json_path = REPORT_HTML.with_suffix(".json")
            obj.save_json(str(json_path))
            raise RuntimeError(
                f"Could not export HTML in this Evidently version, but saved JSON to {json_path}."
            )

    raise RuntimeError("Don't know how to export Evidently output in this Evidently version.")



def main() -> None:
    # Create reference if it doesn't exist yet
    if not REFERENCE_CSV.exists():
        ref = build_reference_csv(max_images=300)
        print(f"Wrote reference to: {REFERENCE_CSV} with shape {ref.shape}")
    else:
        ref = pd.read_csv(REFERENCE_CSV)
        print(f"Loaded reference from: {REFERENCE_CSV} with shape {ref.shape}")

    cur = load_current_csv(tail_n=300)
    print(f"Loaded current from: {REQUESTS_CSV} (tail) with shape {cur.shape}")

    run_evidently_report(ref, cur)
    print(f"Saved drift report to: {REPORT_HTML}")


if __name__ == "__main__":
    main()
