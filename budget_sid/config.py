"""Load the lightweight Beauty config and reject GRID's large-model defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import math

import yaml

from budget_sid.constants import (
    BEAM_WIDTHS,
    EMBEDDING_DIM,
    GRID_COMMIT,
    LARGE_BEAM,
    REFERENCE_BEAM,
    SMALL_BEAM,
    TEXT_ENCODER,
    TOP_K,
)
from budget_sid.errors import ConfigRejected

FORBIDDEN_STRATEGIES = frozenset(
    {
        "ddp",
        "ddp_spawn",
        "ddp_find_unused_parameters_true",
        "ddp_notebook",
        "deepspeed",
        "fsdp",
    }
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_config_path() -> Path:
    path = repo_root() / "configs" / "lightweight_beauty.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing lightweight config at {path}")
    return path


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else default_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigRejected("config root must be a mapping")
    validate_config(data)
    return data


def validate_config(data: dict[str, Any]) -> None:
    """Reject FLAN-T5-XL and DDP, then enforce the frozen Beauty contract."""
    _reject_forbidden(data)
    _require_lightweight_contract(data)


def _reject_forbidden(node: Any, key: str | None = None) -> None:
    if isinstance(node, dict):
        for child_key, child in node.items():
            _reject_forbidden(child, str(child_key))
        return
    if isinstance(node, (list, tuple)):
        for child in node:
            _reject_forbidden(child, key)
        return
    if isinstance(node, str) and "flan-t5-xl" in node.lower():
        raise ConfigRejected(
            "FLAN-T5-XL is the GRID embedding default and is rejected. "
            f"Use frozen {TEXT_ENCODER}."
        )
    if key == "strategy":
        strategy = str(node).strip().lower()
        if strategy in FORBIDDEN_STRATEGIES or strategy.startswith("ddp"):
            raise ConfigRejected(
                f"trainer strategy {strategy!r} is a distributed default. "
                "Use strategy: single_device with devices: 1."
            )
    if key == "devices" and str(node).strip() in {"-1"}:
        raise ConfigRejected(
            "devices: -1 is the GRID DDP default (every visible GPU). Use devices: 1."
        )


def _require(data: dict[str, Any], *keys: str) -> Any:
    cursor: Any = data
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        if not isinstance(cursor, dict) or key not in cursor:
            raise ConfigRejected(f"missing config field {'/'.join(walked)}")
        cursor = cursor[key]
    return cursor


def _require_lightweight_contract(data: dict[str, Any]) -> None:
    dataset = _require(data, "dataset", "name")
    if dataset != "beauty":
        raise ConfigRejected("this scaffold's frozen config is the Beauty path")

    encoder = _require(data, "text_encoder", "name")
    if encoder != TEXT_ENCODER:
        raise ConfigRejected(f"text encoder must be frozen {TEXT_ENCODER}")
    if int(_require(data, "text_encoder", "embedding_dim")) != EMBEDDING_DIM:
        raise ConfigRejected(f"MiniLM embedding dim must be {EMBEDDING_DIM}")
    if _require(data, "text_encoder", "frozen") is not True:
        raise ConfigRejected("text encoder must stay frozen")
    if _require(data, "text_encoder", "train_encoder") is not False:
        raise ConfigRejected("do not train the text encoder")

    if int(_require(data, "semantic_id", "num_codebooks")) != 3:
        raise ConfigRejected("RQ-KMeans uses 3 codebooks")
    if int(_require(data, "semantic_id", "codebook_width")) != 256:
        raise ConfigRejected("each RQ-KMeans codebook has width 256")
    if _require(data, "semantic_id", "dedup_digit") is not True:
        raise ConfigRejected("TIGER dedup digit is required")
    if int(_require(data, "semantic_id", "num_hierarchies")) != 4:
        raise ConfigRejected("num_hierarchies must be 3 codebooks plus the TIGER dedup digit")
    if str(_require(data, "semantic_id", "method")) != "rq_kmeans":
        raise ConfigRejected("semantic id method must be rq_kmeans")

    if int(_require(data, "model", "d_model")) != 128:
        raise ConfigRejected("lightweight transformer d_model must be 128")
    if int(_require(data, "model", "num_encoder_layers")) != 4:
        raise ConfigRejected("encoder depth must be 4")
    if int(_require(data, "model", "num_decoder_layers")) != 4:
        raise ConfigRejected("decoder depth must be 4")
    heads = int(_require(data, "model", "num_heads"))
    if 128 % heads != 0:
        raise ConfigRejected("d_model must be divisible by num_heads")
    backbone = str(_require(data, "model", "backbone")).lower()
    if backbone != "lightweight_t5":
        raise ConfigRejected("backbone must be lightweight_t5")

    if int(_require(data, "train", "batch_size")) != 16:
        raise ConfigRejected("per-device batch size must be 16")
    if int(_require(data, "train", "grad_accum")) != 4:
        raise ConfigRejected("gradient accumulation must be 4")
    if int(_require(data, "train", "devices")) != 1:
        raise ConfigRejected("single device only")
    if int(_require(data, "train", "num_nodes")) != 1:
        raise ConfigRejected("single node only")
    if str(_require(data, "train", "strategy")) != "single_device":
        raise ConfigRejected("strategy must be single_device")

    beams = tuple(int(width) for width in _require(data, "inference", "beam_widths"))
    if beams != BEAM_WIDTHS:
        raise ConfigRejected(f"beam widths must be {list(BEAM_WIDTHS)}")
    if int(_require(data, "inference", "small_beam")) != SMALL_BEAM:
        raise ConfigRejected("small beam must be 20")
    if int(_require(data, "inference", "large_beam")) != LARGE_BEAM:
        raise ConfigRejected("large beam must be 60")
    if int(_require(data, "inference", "reference_beam")) != REFERENCE_BEAM:
        raise ConfigRejected("uniform reference beam must be 40")
    if int(_require(data, "inference", "top_k")) != TOP_K:
        raise ConfigRejected("always return Top-20, regardless of beam width")
    if _require(data, "inference", "shared_first_step") is not True:
        raise ConfigRejected("routing must share the first decoding step")
    if _require(data, "inference", "forbid_rerun_large_after_small") is not True:
        raise ConfigRejected("do not finish a small beam and then rerun a large beam")

    if _require(data, "benefit", "target") != "delta_ndcg@20":
        raise ConfigRejected("benefit target must be delta NDCG@20 of large versus small beam")
    if _require(data, "benefit", "test_locked") is not True:
        raise ConfigRejected("the test split must stay locked")
    if list(_require(data, "benefit", "val_user_splits")) != ["A", "B"]:
        raise ConfigRejected("validation users are split into disjoint A and B")

    if _require(data, "eval", "sampled_negatives") is not False:
        raise ConfigRejected("evaluate full-catalog retrieval, not sampled negatives")
    metrics = list(_require(data, "eval", "metrics"))
    required_metrics = [
        "recall@20",
        "ndcg@20",
        "latency_mean",
        "latency_p95",
        "throughput",
        "peak_vram",
    ]
    if metrics != required_metrics:
        raise ConfigRejected(f"eval metrics must be {required_metrics}")
    if list(_require(data, "eval", "slices")) != ["long_tail", "short_history"]:
        raise ConfigRejected("eval slices must be long_tail and short_history")
    seeds = list(_require(data, "eval", "seeds"))
    if len(seeds) != 3:
        raise ConfigRejected("the chosen method is reported over 3 seeds")
    if not math.isclose(float(_require(data, "stop_gate", "top_fraction")), 0.2):
        raise ConfigRejected("stop-gate top_fraction must be 0.2")
    if not math.isclose(float(_require(data, "stop_gate", "min_positive_mass")), 0.5):
        raise ConfigRejected("stop-gate min_positive_mass must be 0.5")

    pinned = str(_require(data, "grid", "commit"))
    if pinned != GRID_COMMIT:
        raise ConfigRejected(f"GRID must stay pinned at {GRID_COMMIT}")
    if _require(data, "grid", "vendor") is not False:
        raise ConfigRejected("do not vendor GRID; pin the upstream checkout")
