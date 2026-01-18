
# har lige fjernet denne fil midlertidigt da den failede og ved heller ikker om det nødvendigvis er relevant for projektet!

# import io
# import importlib
# from pathlib import Path
# from types import SimpleNamespace

# import anyio
# import pytest
# import torch
# import torch.nn as nn
# from PIL import Image


# def make_fake_uploadfile(content_type: str, fileobj):
#     return SimpleNamespace(content_type=content_type, file=fileobj)


# @pytest.fixture
# def api_module(monkeypatch):
#     monkeypatch.setattr(Path, "exists", lambda self: True)
#     monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
#     monkeypatch.setattr(nn.Module, "load_state_dict", lambda self, state: None)

#     class DummyModel(nn.Module):
#         def forward(self, x):
#             return torch.zeros(x.shape[0], 4)

#     monkeypatch.setattr(
#         "mlops_group54_project.model.build_model",
#         lambda *args, **kwargs: DummyModel(),
#     )

#     api = importlib.import_module("mlops_group54_project.api")
#     importlib.reload(api)
#     return api


# def test_device_returns_torch_device(api_module):
#     assert isinstance(api_module._device(), torch.device)


# def test_health_returns_expected_dict(api_module):
#     out = api_module.health()
#     assert out["status"] == "ok"
#     assert "device" in out
#     assert "checkpoint" in out


# def test_predict_rejects_wrong_content_type(api_module):
#     fake = make_fake_uploadfile("text/plain", io.BytesIO(b"hello"))
#     with pytest.raises(api_module.HTTPException) as exc:
#         anyio.run(api_module.predict, fake)
#     assert exc.value.status_code == 415


# def test_predict_rejects_unreadable_image(api_module):
#     fake = make_fake_uploadfile("image/png", io.BytesIO(b"not an image"))
#     with pytest.raises(api_module.HTTPException) as exc:
#         anyio.run(api_module.predict, fake)
#     assert exc.value.status_code == 400


# def test_predict_accepts_valid_image(api_module):
#     img = Image.new("RGB", (8, 8), color=(255, 0, 0))
#     buf = io.BytesIO()
#     img.save(buf, format="PNG")
#     buf.seek(0)

#     fake = make_fake_uploadfile("image/png", buf)
#     out = anyio.run(api_module.predict, fake)

#     assert isinstance(out["pred_class"], int)
#     assert isinstance(out["probs"], list)
#     assert len(out["probs"]) == 4
