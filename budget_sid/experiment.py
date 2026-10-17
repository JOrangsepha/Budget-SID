"""CPU smoke pipeline and the GPU command plan.

``--smoke`` fits the benefit model and scores beams with the scripted
recommender. The GPU path is a plan of GRID commands plus this package's
budget module; it does not download Beauty or call an external API.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from budget_sid.baselines import build_policies
from budget_sid.benefit import select_threshold, train_on_split_a
from budget_sid.constants import TOP_K
from budget_sid.grid_adapter import grid_commands, pin_status
from budget_sid.metrics import summarize_requests
from budget_sid.recommender import LookupRecommender, ScriptedRecommender
from budget_sid.routing import route_request
from budget_sid.semantic_ids import semantic_ids_from_embeddings
from budget_sid.splits import assert_test_unused
from budget_sid.stop_gate import run_stop_gate
from budget_sid.synthetic import CATALOG, make_population

GPU_PLAN_MESSAGE = (
    "GPU path is not executed in this environment. "
    "Fixed-beam Beauty inference already finished on SeetaCloud on 2026-10-17; "
    "see artifacts/BEAUTY_GPU_RESULTS.json. "
    "Request-level benefit-aware routing is not wired to those GRID outputs. "
    "On one GPU, run scripts/run_beauty_gpu.sh or scripts/run_after_tiger.sh. "
    "This process only prints the command plan."
)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return value


def _gpu_plan(command: str, config: dict) -> dict:
    status = pin_status()
    return {
        "command": command,
        "ok": False,
        "exit_code": 1,
        "status": "gpu_path_not_run",
        "scripted": False,
        "message": GPU_PLAN_MESSAGE,
        "pin": status,
        "grid_commands": grid_commands(config),
        "embedding_encoder": config["text_encoder"]["name"],
    }


def command_train_recommender(config: dict, smoke: bool) -> dict:
    if not smoke:
        return _gpu_plan("train_recommender", config)
    return {
        "command": "train_recommender",
        "ok": True,
        "exit_code": 0,
        "status": "scripted",
        "scripted": True,
        "device": "cpu",
        "dataset": config["dataset"]["name"],
        "d_model": int(config["model"]["d_model"]),
        "num_encoder_layers": int(config["model"]["num_encoder_layers"]),
        "num_decoder_layers": int(config["model"]["num_decoder_layers"]),
        "batch_size": int(config["train"]["batch_size"]),
        "grad_accum": int(config["train"]["grad_accum"]),
        "strategy": config["train"]["strategy"],
        "devices": int(config["train"]["devices"]),
        "text_encoder": config["text_encoder"]["name"],
        "text_encoder_frozen": True,
    }


def command_build_semantic_ids(config: dict, smoke: bool) -> dict:
    if not smoke:
        return _gpu_plan("build_semantic_ids", config)
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(12, int(config["text_encoder"]["embedding_dim"])))
    embeddings[1] = embeddings[0]
    codes = semantic_ids_from_embeddings(
        embeddings,
        n_codebooks=int(config["semantic_id"]["num_codebooks"]),
        codebook_width=int(config["semantic_id"]["codebook_width"]),
        seed=0,
    )
    return {
        "command": "build_semantic_ids",
        "ok": True,
        "exit_code": 0,
        "status": "scripted",
        "scripted": True,
        "embedding_dim": int(embeddings.shape[1]),
        "num_codebooks": int(config["semantic_id"]["num_codebooks"]),
        "dedup_digit": True,
        "code_shape": [int(codes.shape[0]), int(codes.shape[1])],
        "duplicate_prefix_matched": bool(np.array_equal(codes[0, :3], codes[1, :3])),
        "duplicate_dedup_differs": bool(int(codes[0, 3]) != int(codes[1, 3])),
    }


def _fit_smoke(config: dict):
    rows = make_population()
    trained = train_on_split_a(rows, seed=int(config["train"]["seed"]))
    test_ids = [row["user_id"] for row in rows if row["split"] == "test"]
    assert_test_unused(trained.split_a | trained.split_b, test_ids)
    val_rows = [row for row in rows if row["split"] == "val"]
    budget_rows = [row for row in val_rows if row["user_id"] in trained.split_b]
    entropy_threshold = select_threshold(
        np.array([row["features"]["first_step_entropy"] for row in budget_rows], dtype=np.float64)
    )
    return rows, trained, entropy_threshold


def command_train_benefit_predictor(config: dict, smoke: bool) -> dict:
    if not smoke:
        return _gpu_plan("train_benefit_predictor", config)
    _rows, trained, entropy_threshold = _fit_smoke(config)
    return {
        "command": "train_benefit_predictor",
        "ok": True,
        "exit_code": 0,
        "status": "scripted",
        "scripted": True,
        "target": config["benefit"]["target"],
        "n_train_rows": trained.n_train_rows,
        "n_split_a": len(trained.split_a),
        "n_split_b": len(trained.split_b),
        "splits_disjoint": len(trained.split_a & trained.split_b) == 0,
        "threshold": trained.threshold,
        "entropy_threshold": entropy_threshold,
        "test_locked": True,
    }


def _score_policy(policy, test_rows: list[dict]) -> dict:
    recommender = LookupRecommender(test_rows)
    scored = []
    for row in test_rows:
        routed = route_request(recommender, row["history"], policy, CATALOG, top_k=TOP_K)
        scored.append(
            {
                "predicted": routed.recommendations,
                "relevant": row["relevant"],
                "latency_s": routed.latency_s,
                "history_length": row["history_length"],
                "long_tail": row["long_tail"],
                "beam_width": routed.beam_width,
                "vram_mb": 0.0,
            }
        )
    if recommender.full_decode_calls != 0:
        raise RuntimeError("evaluation reran a full decode")
    summary = summarize_requests(scored)
    summary["mean_beam_width"] = float(np.mean([item["beam_width"] for item in scored]))
    summary["max_recommendations"] = max(len(item["predicted"]) for item in scored)
    return summary


def command_evaluate_budgets(config: dict, smoke: bool, *, records=None, load_test=None) -> dict:
    """Score fixed and routed beams on the test split only after the stop-gate passes.

    ``load_test`` is the only test-set read on this path. It is not called
    when the gate returns exit code 2. ``records`` replaces the scripted
    population for tests; the CLI leaves it unset.
    """
    if not smoke:
        plan = _gpu_plan("evaluate_budgets", config)
        plan["sampled_negatives"] = False
        return plan
    if records is None:
        rows, trained, entropy_threshold = _fit_smoke(config)
    else:
        rows = list(records)
        trained = train_on_split_a(rows, seed=int(config["train"]["seed"]))
        test_ids = [row["user_id"] for row in rows if row.get("split") == "test"]
        assert_test_unused(trained.split_a | trained.split_b, test_ids)
        val_rows_b = [
            row for row in rows if row.get("split") == "val" and str(row["user_id"]) in trained.split_b
        ]
        entropy_threshold = select_threshold(
            np.array([row["features"]["first_step_entropy"] for row in val_rows_b], dtype=np.float64)
        )
    val_rows = [row for row in rows if row.get("split", "val") == "val"]
    gate = run_stop_gate(
        val_rows,
        top_fraction=float(config["stop_gate"]["top_fraction"]),
        min_positive_mass=float(config["stop_gate"]["min_positive_mass"]),
    )
    payload = {
        "command": "evaluate_budgets",
        "ok": gate.ok,
        "exit_code": gate.exit_code,
        "status": "scripted" if gate.ok else "stopped",
        "scripted": True,
        "sampled_negatives": False,
        "test_locked": True,
        "test_scored": False,
        "stop_gate": {
            "exit_code": gate.exit_code,
            "concentrated_gain": gate.concentrated_gain,
            "oracle_beats_reference": gate.oracle_beats_reference,
            "mean_delta_ndcg": gate.mean_delta_ndcg,
            "oracle_ndcg": gate.oracle_ndcg,
            "reference_ndcg": gate.reference_ndcg,
            "reasons": gate.reasons,
        },
    }
    def default_load_test():
        return [row for row in rows if row.get("split") == "test"]

    if not gate.ok:
        return payload
    test_rows = list((load_test or default_load_test)())
    policies = build_policies(
        predictor=trained.predictor,
        threshold=trained.threshold,
        entropy_threshold=entropy_threshold,
        seed=int(config["train"]["seed"]),
    )
    payload["test_scored"] = True
    payload["test_metrics"] = {name: _score_policy(policy, test_rows) for name, policy in policies.items()}
    return payload


def parse_history(text: str) -> list:
    if not str(text).strip():
        return [1, 2, 3]
    parsed = []
    for part in str(text).split(","):
        token = part.strip()
        if not token:
            continue
        try:
            parsed.append(int(token))
        except ValueError:
            parsed.append(token)
    return parsed or [1, 2, 3]


def command_demo(config: dict, smoke: bool, history: str = "") -> dict:
    if not smoke:
        return _gpu_plan("demo", config)
    _rows, trained, entropy_threshold = _fit_smoke(config)
    policies = build_policies(
        predictor=trained.predictor,
        threshold=trained.threshold,
        entropy_threshold=entropy_threshold,
        seed=int(config["train"]["seed"]),
    )
    items = parse_history(history)
    recommender = ScriptedRecommender(catalog=CATALOG, seed=int(config["train"]["seed"]))
    routed = route_request(
        recommender,
        items,
        policies["benefit_aware"],
        CATALOG,
        top_k=int(config["inference"]["top_k"]),
    )
    if recommender.full_decode_calls != 0:
        raise RuntimeError("demo reran a full decode")
    return {
        "command": "demo",
        "ok": True,
        "exit_code": 0,
        "status": "scripted",
        "scripted": True,
        "history": items,
        "recommendations": routed.recommendations,
        "beam_width": routed.beam_width,
        "predicted_delta_ndcg": routed.predicted_delta,
        "policy": routed.policy,
        "top_k": int(config["inference"]["top_k"]),
        "latency_s": routed.latency_s,
    }
