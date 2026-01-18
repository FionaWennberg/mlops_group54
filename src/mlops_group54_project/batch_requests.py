from __future__ import annotations

from pathlib import Path
import time
import requests

API_URL = "http://127.0.0.1:8000/predict"

# Change this to a folder that contains MRI images
IMAGE_DIR = Path("data/raw/Testing")  # or data/raw/Training
EXTS = {".jpg", ".jpeg", ".png"}


def main(n: int = 200, sleep: float = 0.0) -> None:
    paths = [p for p in IMAGE_DIR.rglob("*") if p.suffix.lower() in EXTS]
    if not paths:
        raise SystemExit(f"No images found under {IMAGE_DIR}")

    paths = paths[:n]
    ok = 0

    for i, p in enumerate(paths, 1):
        with p.open("rb") as f:
            files = {"file": (p.name, f, "image/jpeg")}
            r = requests.post(API_URL, files=files, timeout=60)

        if r.ok:
            ok += 1
        else:
            print(f"[{i}/{len(paths)}] FAIL {p} -> {r.status_code} {r.text}")

        if sleep > 0:
            time.sleep(sleep)

        if i % 25 == 0:
            print(f"Sent {i}/{len(paths)} (ok={ok})")

    print(f"Done. Sent {len(paths)} images (ok={ok}).")


if __name__ == "__main__":
    main(n=200, sleep=0.0)
