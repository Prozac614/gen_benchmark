from __future__ import annotations

import importlib
from typing import Dict

from .base import BaseEngine

_ENGINE_CLASS_TO_MODULE: Dict[str, str] = {
    "DiffusersEngine": ".diffusers_engine",
    "FalEngine": ".fal_engine",
    "SGLangEngine": ".sglang_engine",
    "LightX2VEngine": ".lightx2v_engine",
    "VllmOmniEngine": ".vllm_omni_engine",
}

__all__ = ["BaseEngine", *_ENGINE_CLASS_TO_MODULE.keys()]


def __getattr__(name: str):
    module_rel = _ENGINE_CLASS_TO_MODULE.get(name)
    if module_rel is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_rel, __name__)
    return getattr(module, name)
