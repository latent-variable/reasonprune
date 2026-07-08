"""Physically slice pruned MLP channels out of a checkpoint.

Produces a smaller, still-standard checkpoint: per-layer-uniform keep counts
mean `intermediate_size` stays one scalar in config.json, so the output loads
in mlx_lm / oMLX / transformers unchanged. Handles bf16 tensors and MLX
affine-quantized triplets (weight/scales/biases; dequantize -> slice ->
requantize, keep count snapped to a multiple of group_size).

MLP tensor keys matched: *.layers.{i}.mlp.{gate,up,down}_proj.*  — the `mtp.*`
(multi-token-prediction) tree and everything else pass through untouched.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np

MLP_KEY = re.compile(
    r"(?P<prefix>.*\.layers\.(?P<layer>\d+)\.mlp\.)"
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<part>weight|scales|biases)$")

AUX_KEEP = ("tokenizer", "chat_template", "merges", "vocab", "config",
            "generation_config", "preprocessor", "LICENSE")


def uniform_keep(mask: np.ndarray, scores: np.ndarray | None,
                 group_size: int | None) -> list[np.ndarray]:
    """Per-layer keep index lists with a UNIFORM count across layers.

    Uses the min keep count over layers (snapped to group_size); layers with
    more survivors keep their highest-scoring ones (scores = prunability,
    lower = keep first; None = keep lowest index).
    """
    n_layers, d = mask.shape
    counts = [(~mask[l]).sum() for l in range(n_layers)]
    k = min(counts)
    if group_size:
        k = (k // group_size) * group_size
    out = []
    for l in range(n_layers):
        keep = np.where(~mask[l])[0]
        if len(keep) > k:
            if scores is not None:
                keep = keep[np.argsort(scores[l, keep], kind="stable")][:k]
            else:
                keep = keep[:k]
        out.append(np.sort(keep))
    return out


def _dequant(w, s, b, group_size, bits):
    return mx.dequantize(w, s, b, group_size=group_size, bits=bits)


def slice_checkpoint(model_dir: Path, out_dir: Path, mask: np.ndarray,
                     scores: np.ndarray | None = None) -> dict:
    model_dir, out_dir = Path(model_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((model_dir / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    quant = cfg.get("quantization")
    group_size = quant.get("group_size") if quant else None
    bits = quant.get("bits") if quant else None

    keeps = uniform_keep(mask, scores, group_size)
    new_inter = len(keeps[0])
    old_inter = tc["intermediate_size"]

    shards = sorted(model_dir.glob("*.safetensors"))
    # Group quantized triplets: collect all tensors first (128 GB host: fine
    # for <=35 GB checkpoints), transform, then rewrite shard-per-shard.
    index = {"metadata": {}, "weight_map": {}}
    total_before = total_after = 0
    for shard in shards:
        tensors = mx.load(str(shard))
        out_tensors = {}
        # Pass 1: non-MLP and bf16 MLP tensors; collect quant triplet names.
        triplets = {}
        for name, t in tensors.items():
            m = MLP_KEY.match(name)
            total_before += t.nbytes
            if not m or name.startswith("mtp."):
                out_tensors[name] = t
                continue
            if m["part"] in ("scales", "biases") or (quant and m["part"] == "weight"):
                triplets.setdefault(m["prefix"] + m["proj"], {})[m["part"]] = t
                continue
            # bf16 path
            keep = mx.array(keeps[int(m["layer"])])
            if m["proj"] == "down_proj":
                out_tensors[name] = t[:, keep]
            else:
                out_tensors[name] = t[keep, :]
        # Pass 2: quantized triplets (dequantize, slice, requantize).
        for base, parts in triplets.items():
            lm = MLP_KEY.match(base + ".weight")
            layer, proj = int(lm["layer"]), lm["proj"]
            keep = mx.array(keeps[layer])
            dense = _dequant(parts["weight"], parts["scales"],
                             parts.get("biases"), group_size, bits)
            dense = dense[:, keep] if proj == "down_proj" else dense[keep, :]
            wq, sc, bi = mx.quantize(dense, group_size=group_size, bits=bits)
            out_tensors[base + ".weight"] = wq
            out_tensors[base + ".scales"] = sc
            out_tensors[base + ".biases"] = bi
        mx.eval(list(out_tensors.values()))
        for name, t in out_tensors.items():
            index["weight_map"][name] = shard.name
            total_after += t.nbytes
        mx.save_safetensors(str(out_dir / shard.name), out_tensors)
        del tensors, out_tensors
        mx.clear_cache()

    tc["intermediate_size"] = new_inter
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    idx_file = model_dir / "model.safetensors.index.json"
    if idx_file.exists():
        index["metadata"]["total_size"] = total_after
        (out_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2))
    for f in model_dir.iterdir():
        if f.name.startswith(AUX_KEEP) and f.name != "config.json" \
                and not f.name.endswith(".safetensors"):
            shutil.copy(f, out_dir / f.name)
    meta = {
        "old_intermediate_size": old_inter,
        "new_intermediate_size": new_inter,
        "mlp_reduction": 1 - new_inter / old_inter,
        "bytes_before": total_before,
        "bytes_after": total_after,
        "total_reduction": 1 - total_after / max(total_before, 1),
    }
    (out_dir / "reasonprune.json").write_text(json.dumps(meta, indent=2))
    return meta
