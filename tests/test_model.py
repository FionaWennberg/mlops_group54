import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import mlops_group54_project.model as model_mod


# -------------------------
# Helpers for mocking resnet
# -------------------------
class DummyResNet(nn.Module):
    """
    Minimal stand-in for torchvision resnet model.
    Only needs .fc with .in_features so build_model can replace it.
    """

    def __init__(self, in_features: int = 2048):
        super().__init__()
        self.fc = nn.Linear(in_features, 1000)  # placeholder


def test_build_model_raises_on_unsupported_backbone():
    cfg = model_mod.ModelConfig(backbone="vgg16", pretrained=False, num_classes=4, dropout=0.0)
    with pytest.raises(ValueError, match="Unsupported backbone"):
        _ = model_mod.build_model(cfg)


def test_build_model_sets_linear_head_when_no_dropout(monkeypatch):
    """
    Unit test: mock torchvision.models.resnet50 so we don't build a huge model.
    Covers: happy path + dropout==0 branch (fc becomes nn.Linear).
    """

    def fake_resnet50(*args, **kwargs):
        return DummyResNet(in_features=2048)

    monkeypatch.setattr(model_mod.models, "resnet50", fake_resnet50)

    cfg = model_mod.ModelConfig(backbone="resnet50", pretrained=False, num_classes=7, dropout=0.0)
    m = model_mod.build_model(cfg)

    assert isinstance(m.fc, nn.Linear)
    assert m.fc.out_features == 7


def test_build_model_sets_dropout_head_when_dropout_positive(monkeypatch):
    """
    Unit test: covers dropout>0 branch (fc becomes Sequential(Dropout, Linear)).
    """

    def fake_resnet50(*args, **kwargs):
        return DummyResNet(in_features=2048)

    monkeypatch.setattr(model_mod.models, "resnet50", fake_resnet50)

    cfg = model_mod.ModelConfig(backbone="resnet50", pretrained=False, num_classes=3, dropout=0.25)
    m = model_mod.build_model(cfg)

    assert isinstance(m.fc, nn.Sequential)
    assert isinstance(m.fc[0], nn.Dropout)
    assert isinstance(m.fc[1], nn.Linear)
    assert m.fc[1].out_features == 3


def test_build_model_fallback_on_old_torchvision_api(monkeypatch):
    """
    Unit test: covers try/except TypeError fallback path in build_model.

    We simulate:
    - calling resnet50(weights=...) raises TypeError (old torchvision signature)
    - then fallback resnet50(pretrained=...) returns a model
    """

    calls = {"weights": 0, "pretrained": 0}

    def fake_resnet50(*args, **kwargs):
        if "weights" in kwargs:
            calls["weights"] += 1
            raise TypeError("old torchvision does not support weights kwarg")
        if "pretrained" in kwargs:
            calls["pretrained"] += 1
            return DummyResNet(in_features=2048)
        return DummyResNet(in_features=2048)

    monkeypatch.setattr(model_mod.models, "resnet50", fake_resnet50)

    cfg = model_mod.ModelConfig(backbone="resnet50", pretrained=False, num_classes=4, dropout=0.0)
    m = model_mod.build_model(cfg)

    assert isinstance(m.fc, nn.Linear)
    assert calls["weights"] == 1
    assert calls["pretrained"] == 1


def test_to_model_config_reads_fields_and_defaults():
    # matches your configs/model/model.yaml keys
    cfg = OmegaConf.create(
        {
            "model": {
                "backbone": "resnet50",
                "pretrained": True,
                "num_classes": 4,
                "dropout": 0.0,
            }
        }
    )
    mc = model_mod._to_model_config(cfg)
    assert mc.backbone == "resnet50"
    assert mc.pretrained is True
    assert mc.num_classes == 4
    assert mc.dropout == 0.0

    # without dropout -> default 0.0 (getattr fallback)
    cfg2 = OmegaConf.create({"model": {"backbone": "resnet50", "pretrained": False, "num_classes": 2}})
    mc2 = model_mod._to_model_config(cfg2)
    assert mc2.dropout == 0.0


@pytest.mark.parametrize("num_classes", [1, 4, 10])
def test_build_model_smoke_forward_shape_with_mock(monkeypatch, num_classes):
    """
    "Smoke" style, but still mocked:
    we give DummyResNet a forward so we can test output shape quickly.

    This doesn't use real torchvision or real ResNet.
    """

    class DummyResNetWithForward(DummyResNet):
        def __init__(self, in_features=8):
            super().__init__(in_features=in_features)

        def forward(self, x):
            # x: [B, 3, H, W] -> simple pooled feature: [B, in_features]
            b = x.shape[0]
            feats = torch.zeros(b, self.fc.in_features)
            return self.fc(feats)

    def fake_resnet50(*args, **kwargs):
        return DummyResNetWithForward(in_features=8)

    monkeypatch.setattr(model_mod.models, "resnet50", fake_resnet50)

    cfg = model_mod.ModelConfig(backbone="resnet50", pretrained=False, num_classes=num_classes, dropout=0.0)
    m = model_mod.build_model(cfg)

    x = torch.randn(2, 3, 16, 16)
    y = m(x)
    assert y.shape == (2, num_classes)
