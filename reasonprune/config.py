"""Shared configuration: model paths, oMLX server, run defaults."""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"

# Direct-load analysis models (HF hub ids; mlx_lm resolves from local cache).
MODELS = {
    "qwen-0.8b": "Qwen/Qwen3.5-0.8B",
    "qwen-4b": "Qwen/Qwen3.5-4B",
    # MoE target: local path, standard MLX 8-bit quant of the A3B worker family.
    "qwen-35b-a3b": str(Path.home() / ".omlx/models/unsloth/Qwen3.6-35B-A3B-MLX-8bit"),
}


def omlx_config() -> dict:
    """Read the local oMLX server endpoint + key from ~/.omlx/settings.json."""
    settings = json.loads((Path.home() / ".omlx/settings.json").read_text())
    server = settings.get("server", settings)
    port = server.get("port", 5599)
    key = server.get("api_key") or settings.get("api_key", "")
    return {"base_url": f"http://localhost:{port}/v1", "api_key": key}


WORKER_MODEL = os.environ.get("REASONPRUNE_WORKER", "Qwen3.6-35B-A3B-oQ4e-mtp")
