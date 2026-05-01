from __future__ import annotations

import torch

from get.utils.compile import maybe_compile_model


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.lin(x)


def test_maybe_compile_model_disabled_returns_same_object():
    model = _Tiny()
    out = maybe_compile_model(model, {"enabled": False})
    assert out is model


def test_maybe_compile_model_failure_falls_back(monkeypatch):
    model = _Tiny()

    def _boom(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(torch, "compile", _boom)
    out = maybe_compile_model(model, {"enabled": True, "backend": "inductor"})
    assert out is model


def test_maybe_compile_model_skips_double_backward_models_by_default(monkeypatch):
    model = _Tiny()
    model.requires_double_backward = True
    called = {"n": 0}

    def _fake_compile(*args, **kwargs):
        called["n"] += 1
        return args[0]

    monkeypatch.setattr(torch, "compile", _fake_compile)
    out = maybe_compile_model(model, {"enabled": True})
    assert out is model
    assert called["n"] == 0
