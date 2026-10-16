"""Command line for the lightweight Beauty budget experiments."""

from __future__ import annotations

import argparse
import json
import sys

from budget_sid.config import default_config_path, load_config
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
    else:
        raise SystemExit(f"unknown command {args.command}")
    json.dump(to_builtin(payload), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return int(payload["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
