"""Command line for the lightweight Beauty budget experiments."""

from __future__ import annotations

import argparse
import json
import sys

from budget_sid.analyze_benefit import command_analyze_benefit
from budget_sid.replay import command_replay_budgets
from budget_sid.beam_score import command_score_beam_ndcg
from budget_sid.config import default_config_path, load_config
from budget_sid.eval_audit import command_audit_eval_leakage
from budget_sid.eval_samples import command_export_eval_samples
from budget_sid.first_step_dump import command_dump_first_step_features
from budget_sid.first_step_logits import (
    command_dump_first_step_logits,
    command_extract_boundary_features,
)
from budget_sid.leak_safe_predict import command_build_leak_safe_predict, command_export_leak_safe_rankings
from budget_sid.experiment import (
    command_build_semantic_ids,
    command_demo,
    command_evaluate_budgets,
    command_train_benefit_predictor,
    command_train_recommender,
    to_builtin,
)

COMMANDS = (
    "train_recommender",
    "build_semantic_ids",
    "train_benefit_predictor",
    "evaluate_budgets",
    "demo",
    "export_eval_samples",
    "audit_eval_leakage",
    "score_beam_ndcg",
    "build_leak_safe_predict",
    "export_leak_safe_rankings",
    "analyze_benefit",
    "replay_budgets",
    "dump_first_step_features",
    "dump_first_step_logits",
    "extract_boundary_features",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="budget-sid",
        description=(
            "Lightweight TIGER-style recommender on GRID with request-level beam budgets. "
            "Pass --smoke to run the scripted CPU path."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        command = subparsers.add_parser(name)
        command.add_argument("--config", default=str(default_config_path()))
        command.add_argument(
            "--smoke",
            action="store_true",
            help="use the scripted CPU recommender; do not download data or call GRID",
        )
        if name == "demo":
            command.add_argument(
                "--history",
                default="",
                help="comma-separated history item ids; default is a short scripted history",
            )
        if name in {
            "export_eval_samples",
            "audit_eval_leakage",
            "score_beam_ndcg",
            "build_leak_safe_predict",
            "export_leak_safe_rankings",
            "dump_first_step_features",
            "dump_first_step_logits",
            "extract_boundary_features",
        }:
            command.add_argument(
                "--dry-run",
                action="store_true",
                help="check inputs and print the summary without writing files",
            )
        if name == "export_eval_samples":
            command.add_argument("--data-dir", default=None, help="Beauty root with training/evaluation/testing")
            command.add_argument("--sequences", default=None, help="JSONL, CSV, JSON, or a split directory")
            command.add_argument("--output-dir", default=None)
            command.add_argument("--checkpoint", default=None, help="checkpoint file to hash")
            command.add_argument("--semantic-id-path", default=None, help="semantic-id file to hash")
            command.add_argument("--checkpoint-hash", default=None)
            command.add_argument("--semantic-id-hash", default=None)
        if name == "audit_eval_leakage":
            command.add_argument("--samples", default=None, help="eval_samples.jsonl from export_eval_samples")
            command.add_argument("--output", default=None, help="audit report JSON path")
        if name == "score_beam_ndcg":
            command.add_argument("--samples", default=None, help="eval_samples.jsonl")
            command.add_argument("--beam20", default=None)
            command.add_argument("--beam40", default=None)
            command.add_argument("--beam60", default=None)
            command.add_argument(
                "--rankings-dir",
                default=None,
                help="directory of beam_20/beam_40/beam_60 JSONL, CSV, or pickle files",
            )
            command.add_argument("--output-dir", default=None)
            command.add_argument(
                "--ranking-source",
                default="unspecified",
                choices=("unspecified", "leak_safe_history", "grid_tiger_inference_flat"),
                help="grid_tiger_inference_flat rankings are scored but not treated as leak-safe ΔNDCG",
            )
            command.add_argument("--sid-map", default=None, help="JSON/JSONL/pt map from semantic id to item id")
        if name == "build_leak_safe_predict":
            command.add_argument("--samples", default=None, help="eval_samples.jsonl with history and target removed from the model input")
            command.add_argument("--output-dir", default=None, help="folder of history-only TFRecords for tiger_inference_flat")
            command.add_argument("--semantic-id-path", default=None, help="optional SID map; history item ids must occur in it")
            command.add_argument("--sequence-length", type=int, default=120)
            command.add_argument("--num-hierarchies", type=int, default=4)
        if name == "export_leak_safe_rankings":
            command.add_argument("--predictions", default=None, help="GRID merged_predictions.pkl (user_id is predict_key)")
            command.add_argument("--index", default=None, help="predict_index.jsonl from build_leak_safe_predict")
            command.add_argument("--sid-map", default=None, help="JSON/JSONL/pt map from semantic id to item id")
            command.add_argument("--beam", type=int, default=None)
            command.add_argument("--output", default=None, help="beam_<k>.jsonl with request_id and ranked_items")
        if name == "dump_first_step_features":
            command.add_argument(
                "--predict-dir",
                default=None,
                help="history-only folder from build_leak_safe_predict (not GRID's testing split)",
            )
            command.add_argument(
                "--output",
                default=None,
                help="first_step_features.jsonl joined on request_id in predict_index order",
            )
            command.add_argument("--ckpt", default=None, help="tiger_train_flat checkpoint; required on GPU")
            command.add_argument("--semantic-id-path", default=None)
            command.add_argument("--data-dir", default=None, help="Beauty data dir passed through to tiger_inference_flat")
            command.add_argument("--grid-dir", default=None, help="pinned GRID checkout (commit 2fe3475)")
            command.add_argument(
                "--per-request",
                default=None,
                help="optional per_request.jsonl; request_id order must match the predict index",
            )
        if name == "dump_first_step_logits":
            command.add_argument(
                "--predict-dir",
                default=None,
                help="history-only folder from build_leak_safe_predict (not GRID's testing split)",
            )
            command.add_argument(
                "--output",
                default=None,
                help="npz cache of hierarchy-0 logit vectors, joined on request_id and predict_key",
            )
            command.add_argument("--ckpt", default=None, help="tiger_train_flat checkpoint; required on GPU")
            command.add_argument("--semantic-id-path", default=None)
            command.add_argument("--data-dir", default=None, help="Beauty data dir passed through to tiger_inference_flat")
            command.add_argument("--grid-dir", default=None, help="pinned GRID checkout (commit 2fe3475)")
            command.add_argument(
                "--per-request",
                default=None,
                help="optional per_request.jsonl; every cached request_id must appear, and the split must be covered",
            )
            command.add_argument(
                "--split",
                default="val",
                help="val (default) caches validation rows; all caches every predict-index row",
            )
            command.add_argument(
                "--filtered-predict-dir",
                default=None,
                help="where to write the val-only history folder when predict_rows.jsonl exists",
            )
        if name == "extract_boundary_features":
            command.add_argument(
                "--logits",
                default=None,
                help="hierarchy-0 logits npz from dump_first_step_logits",
            )
            command.add_argument(
                "--output",
                default=None,
                help="boundary_features.jsonl joined by replay_budgets --features on request_id",
            )
            command.add_argument(
                "--split",
                default=None,
                help="optional val or all filter; default keeps every row in the cache",
            )
        if name == "analyze_benefit":
            command.add_argument(
                "--per-request",
                default=None,
                help="per_request.jsonl from score_beam_ndcg (delta_ndcg@20 and beams.20/40/60.ndcg@20)",
            )
            command.add_argument(
                "--summary",
                default=None,
                help="summary.json; required to confirm auditable_for_delta_ndcg before P2",
            )
            command.add_argument("--output", default=None, help="optional JSON report path")
        if name == "replay_budgets":
            command.add_argument(
                "--per-request",
                default=None,
                help="outputs/leak_safe_beams/per_request.jsonl (gitignored). Beam nDCG labels, not logits.",
            )
            command.add_argument(
                "--summary",
                default=None,
                help="summary.json beside the per-request file; a sibling summary.json is read when this is omitted",
            )
            command.add_argument(
                "--samples",
                default=None,
                help="eval_samples.jsonl joined on request_id for history_length only",
            )
            command.add_argument(
                "--features",
                default=None,
                help="JSONL of request_id plus first-step features or logits. Labels in this file are ignored.",
            )
            command.add_argument(
                "--predict-index",
                default=None,
                help="predict_index.jsonl; n_history is used as history_length when features omit it",
            )
            command.add_argument("--output-dir", default=None)
            command.add_argument("--seed", type=int, default=None)
            command.add_argument(
                "--seeds",
                default=None,
                help="comma-separated seeds, for example 42,43,44. Overrides --seed.",
            )
            command.add_argument(
                "--user-protocol",
                default="auto",
                choices=("auto", "disjoint_users", "temporal_loo"),
                help="auto uses temporal_loo when a validation user also appears in test",
            )
            command.add_argument(
                "--dry-run",
                action="store_true",
                help="score the replay and skip writing --output-dir",
            )
            command.add_argument(
                "--fractions",
                default=None,
                help=(
                    "comma-separated upgrade fractions. Default is "
                    "0,0.01,0.05,0.10,0.25,0.50 (0%%, 1%%, 5%%, 10%%, 25%%, 50%%). "
                    "Declared average beam is 20 + 40*fraction. "
                    "A trailing %% or any token above 1 reads the list as percents."
                ),
            )
            command.add_argument(
                "--quota-mode",
                default="fixed_fraction",
                choices=("fixed_fraction", "positive_count"),
                help=(
                    "fixed_fraction (default) is the P2.1 shared upgrade curve. "
                    "positive_count is the legacy narrow diagnostic "
                    "n_quota=min(n_positive, ceiling) and does not take --fractions."
                ),
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    smoke = bool(args.smoke)
    if args.command == "train_recommender":
        payload = command_train_recommender(config, smoke=smoke)
    elif args.command == "build_semantic_ids":
        payload = command_build_semantic_ids(config, smoke=smoke)
    elif args.command == "train_benefit_predictor":
        payload = command_train_benefit_predictor(config, smoke=smoke)
    elif args.command == "evaluate_budgets":
        payload = command_evaluate_budgets(config, smoke=smoke)
    elif args.command == "demo":
        payload = command_demo(config, smoke=smoke, history=args.history)
    elif args.command == "export_eval_samples":
        payload = command_export_eval_samples(
            smoke=smoke,
            data_dir=args.data_dir,
            sequences=args.sequences,
            output_dir=args.output_dir,
            checkpoint=args.checkpoint,
            semantic_id_path=args.semantic_id_path,
            checkpoint_hash=args.checkpoint_hash,
            semantic_id_hash=args.semantic_id_hash,
            dry_run=args.dry_run,
        )
    elif args.command == "audit_eval_leakage":
        payload = command_audit_eval_leakage(
            smoke=smoke,
            samples=args.samples,
            output=args.output,
            dry_run=args.dry_run,
        )
    elif args.command == "build_leak_safe_predict":
        payload = command_build_leak_safe_predict(
            smoke=smoke,
            samples=args.samples,
            output_dir=args.output_dir,
            semantic_id_path=args.semantic_id_path,
            sequence_length=args.sequence_length,
            num_hierarchies=args.num_hierarchies,
            dry_run=args.dry_run,
        )
    elif args.command == "export_leak_safe_rankings":
        payload = command_export_leak_safe_rankings(
            smoke=smoke,
            predictions=args.predictions,
            index=args.index,
            sid_map=args.sid_map,
            beam=args.beam,
            output=args.output,
            dry_run=args.dry_run,
        )
    elif args.command == "analyze_benefit":
        payload = command_analyze_benefit(
            per_request=args.per_request,
            summary=args.summary,
            smoke=smoke,
            output=args.output,
        )
    elif args.command == "dump_first_step_features":
        payload = command_dump_first_step_features(
            smoke=smoke,
            predict_dir=args.predict_dir,
            output=args.output,
            ckpt=args.ckpt,
            semantic_id_path=args.semantic_id_path,
            data_dir=args.data_dir,
            grid_dir=args.grid_dir,
            per_request=args.per_request,
            dry_run=args.dry_run,
        )
    elif args.command == "dump_first_step_logits":
        payload = command_dump_first_step_logits(
            smoke=smoke,
            predict_dir=args.predict_dir,
            output=args.output,
            ckpt=args.ckpt,
            semantic_id_path=args.semantic_id_path,
            data_dir=args.data_dir,
            grid_dir=args.grid_dir,
            per_request=args.per_request,
            split=args.split,
            filtered_predict_dir=args.filtered_predict_dir,
            dry_run=args.dry_run,
        )
    elif args.command == "extract_boundary_features":
        payload = command_extract_boundary_features(
            smoke=smoke,
            logits=args.logits,
            output=args.output,
            split=args.split,
            dry_run=args.dry_run,
        )
    elif args.command == "replay_budgets":
        payload = command_replay_budgets(
            config,
            per_request=args.per_request,
            summary=args.summary,
            samples=args.samples,
            features=args.features,
            predict_index=args.predict_index,
            output_dir=args.output_dir,
            seed=args.seed,
            seeds=args.seeds,
            user_protocol=args.user_protocol,
            smoke=smoke,
            dry_run=args.dry_run,
            fractions=args.fractions,
            quota_mode=args.quota_mode,
        )
    elif args.command == "score_beam_ndcg":
        payload = command_score_beam_ndcg(
            smoke=smoke,
            samples=args.samples,
            beam20=args.beam20,
            beam40=args.beam40,
            beam60=args.beam60,
            rankings_dir=args.rankings_dir,
            output_dir=args.output_dir,
            ranking_source=args.ranking_source,
            sid_map=args.sid_map,
            dry_run=args.dry_run,
        )
    else:
        raise SystemExit(f"unknown command {args.command}")
    json.dump(to_builtin(payload), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return int(payload["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
