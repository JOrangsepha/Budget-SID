"""Command line for the lightweight Beauty budget experiments."""

from __future__ import annotations

import argparse
import json
import sys

from budget_sid.beam_score import command_score_beam_ndcg
from budget_sid.config import default_config_path, load_config
from budget_sid.eval_audit import command_audit_eval_leakage
from budget_sid.eval_samples import command_export_eval_samples
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
        if name in {"export_eval_samples", "audit_eval_leakage", "score_beam_ndcg"}:
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
