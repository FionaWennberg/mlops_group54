from __future__ import annotations

import io
from typing import Any, Dict, List

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """
    Create a TestClient but monkeypatch the heavy globals in api.py
    so tests do not require real model weights or class_mapping.pt.
    """
    import mlops_group54_project.api as api

    class_names: List[str] = ["glioma", "meningioma", "no_tumor", "pituitary"]
    monkeypatch.setattr(api, "CLASS_NAMES", class_names, raising=False)
    monkeypatch.setattr(api, "NUM_CLASSES", len(class_names), raising=False)

    class DummyModel:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            batch = x.shape[0]
            logits = torch.zeros((batch, len(class_names)), dtype=torch.float32)
            logits[:, 0] = 10.0
            return logits

        def eval(self) -> "DummyModel":
            return self

        def to(self, device: torch.device) -> "DummyModel":
            return self

    monkeypatch.setattr(api, "MODEL", DummyModel(), raising=False)
    monkeypatch.setattr(api, "DEVICE", torch.device("cpu"), raising=False)

    return TestClient(api.app)


def _make_png_bytes(size: int = 224) -> bytes:
    img = Image.new("RGB", (size, size), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_ok(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    data: Dict[str, Any] = r.json()
    assert data["status"] == "ok"
    assert "device" in data
    assert "checkpoint" in data


def test_predict_png_ok(client: TestClient) -> None:
    img_bytes = _make_png_bytes()
    files = {"file": ("test.png", img_bytes, "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200

    data: Dict[str, Any] = r.json()
    assert data["pred_class"] == 0
    assert data["pred_label"] == "glioma"

    probs: Dict[str, float] = data["probs"]
    assert set(probs.keys()) == {"glioma", "meningioma", "no_tumor", "pituitary"}


def test_predict_rejects_wrong_content_type(client: TestClient) -> None:
    files = {"file": ("test.txt", b"hello", "text/plain")}
    r = client.post("/predict", files=files)
    assert r.status_code == 415


def test_predict_rejects_invalid_image_bytes(client: TestClient) -> None:
    files = {"file": ("bad.png", b"not-an-image", "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 400
    
def test_health_has_new_fields(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    data: Dict[str, Any] = r.json()

    # new fields added in updated api.py
    assert "model_loaded" in data
    assert "num_classes" in data
    assert "monitoring_bucket_set" in data
    assert "monitoring_prefix" in data


def test_predict_includes_latency_ms(client: TestClient) -> None:
    img_bytes = _make_png_bytes()
    files = {"file": ("test.png", img_bytes, "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200

    data: Dict[str, Any] = r.json()
    assert "latency_ms" in data
    assert isinstance(data["latency_ms"], float)
    assert data["latency_ms"] >= 0.0


def test_predict_returns_503_when_model_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import mlops_group54_project.api as api

    # simulate "not loaded"
    monkeypatch.setattr(api, "MODEL", None, raising=False)
    monkeypatch.setattr(api, "CLASS_NAMES", ["glioma"], raising=False)

    c = TestClient(api.app)
    img_bytes = _make_png_bytes()
    files = {"file": ("test.png", img_bytes, "image/png")}
    r = c.post("/predict", files=files)
    assert r.status_code == 503


def test_predict_returns_503_when_class_names_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import mlops_group54_project.api as api

    class DummyModel:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros((x.shape[0], 1), dtype=torch.float32)

        def eval(self) -> "DummyModel":
            return self

        def to(self, device: torch.device) -> "DummyModel":
            return self

    monkeypatch.setattr(api, "MODEL", DummyModel(), raising=False)
    monkeypatch.setattr(api, "CLASS_NAMES", [], raising=False)

    c = TestClient(api.app)
    img_bytes = _make_png_bytes()
    files = {"file": ("test.png", img_bytes, "image/png")}
    r = c.post("/predict", files=files)
    assert r.status_code == 503


def test_background_logging_is_scheduled(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Covers the background_tasks.add_task(_log_request, ...) path.
    We don't test GCS/CSV here, only that _log_request gets called with expected fields.
    """
    import mlops_group54_project.api as api

    class_names: List[str] = ["glioma", "meningioma", "no_tumor", "pituitary"]
    monkeypatch.setattr(api, "CLASS_NAMES", class_names, raising=False)

    class DummyModel:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros((x.shape[0], len(class_names)), dtype=torch.float32)
            logits[:, 0] = 10.0
            return logits

        def eval(self) -> "DummyModel":
            return self

        def to(self, device: torch.device) -> "DummyModel":
            return self

    monkeypatch.setattr(api, "MODEL", DummyModel(), raising=False)
    monkeypatch.setattr(api, "DEVICE", torch.device("cpu"), raising=False)

    called: Dict[str, Any] = {}

    def fake_log_request(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(api, "_log_request", fake_log_request, raising=True)

    c = TestClient(api.app)
    img_bytes = _make_png_bytes()
    files = {"file": ("test.png", img_bytes, "image/png")}
    r = c.post("/predict", files=files)
    assert r.status_code == 200

    # Background task should have executed and called fake_log_request
    assert "pred_label" in called
    assert "latency_ms" in called
    assert "image_bytes" in called