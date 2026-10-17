#!/usr/bin/env bash
# Leak-safe first-step features for the P2 replay.
#
# Reuses the history-only predict folder from scripts/run_leak_safe_beams.sh
# (build_leak_safe_predict). The target event is not in that history. This
# script does not read GRID's default testing split and does not score the
# 2026-10-17 tiger_inference_flat pickles.
#
# Pinned GRID generate() at 2fe3475 computes hierarchy-0 candidate_logits and
# then discards them. predict_step writes semantic ids only. The dump calls
# the same encoder/decoder first step and keeps the matrix. It does not run
# the rest of the beam search, so top_k does not change these features. One
# file serves beams 20, 40, and 60.
#
# Missing logits or a missing checkpoint is a non-zero exit. The dump does
# not fill zeros. It does not retrain, does not change the GRID pin, and does
# not start P3.
#
# SeetaCloud, from /root/autodl-tmp/Budget-SID, after the leak-safe predict
# folder exists (or let this script build it):
#
#   SID_PATH=/path/to/semantic_ids.pt bash scripts/dump_first_step_features.sh
#
# CKPT defaults to the 2026-10-17 Beauty checkpoint. DRY_RUN=1 checks the
# predict folder and the checkpoint file and does not load the model.
# SKIP_BUILD=1 reuses an existing predict folder.
#
# Then re-run P2 (CPU, no second decode). P3 stays gated until benefit-aware
# beats random and entropy on positive-Δ mass and nDCG@20 under the same quota:
#
#   python -m budget_sid replay_budgets \
#     --per-request outputs/leak_safe_beams/per_request.jsonl \
#     --summary outputs/leak_safe_beams/summary.json \
#     --features outputs/leak_safe_beams/first_step_features.jsonl \
#     --user-protocol temporal_loo \
#     --seeds 42,43,44 \
#     --output-dir outputs/p2_replay
#
# Hydra, same fixes as scripts/run_leak_safe_beams.sh:
#   ckpt_path='$CKPT'
#   data_loading.predict_dataloader_config.dataloader.num_workers=0
#   data_loading.predict_dataloader_config.dataloader.timeout=0
#   data_loading.predict_dataloader_config.dataloader.data_folder=<history-only dir>
# Do not override test_dataloader_config: tiger_inference_flat has no such key.
#
# This script prints SKIP_SHUTDOWN and does not power the machine off.

set -euo pipefail

log_instance_stays_on() {
  echo "SKIP_SHUTDOWN: instance stays ON. No autodl poweroff. No shutdown."
}
trap log_instance_stays_on EXIT

if [[ -n "${LEGACY_BEAM_DIR:-}${OLD_RANKINGS_DIR:-}" ]]; then
  echo "Refusing legacy tiger_inference_flat rankings." >&2
  echo "The 2026-10-17 beams kept the target event. They are not leak-safe first-step inputs." >&2
  echo "Unset LEGACY_BEAM_DIR and OLD_RANKINGS_DIR." >&2
  exit 1
fi

if [[ -n "${ROOT_DIR:-}" ]]; then
  ROOT="${ROOT_DIR}"
elif [[ -d /root/autodl-tmp/Budget-SID ]]; then
  ROOT=/root/autodl-tmp/Budget-SID
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ -n "${PYTHON:-}" ]]; then
  PY="${PYTHON}"
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  PY=python3
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

GRID_DIR="${GRID_DIR:-${ROOT}/third_party/GRID}"
DATA_DIR="${DATA_DIR:-${GRID_DIR}/data/amazon_data/beauty}"
SAMPLES="${SAMPLES:-${ROOT}/outputs/p0_eval/eval_samples.jsonl}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/leak_safe_beams}"
PREDICT_DIR="${PREDICT_DIR:-${OUT_DIR}/predict}"
FEATURES="${FEATURES:-${OUT_DIR}/first_step_features.jsonl}"
PER_REQUEST="${PER_REQUEST:-${OUT_DIR}/per_request.jsonl}"
SID_PATH="${SID_PATH:-${SEMANTIC_ID_PATH:-}}"
DEFAULT_CKPT="${GRID_DIR}/logs/train/runs/2026-10-17/00-35-10/checkpoints/checkpoint_epoch=000_step=018800.ckpt"
CKPT="${CKPT:-${DEFAULT_CKPT}}"

refuse_quote() {
  local label="$1"
  local value="$2"
  if [[ "${value}" == *"'"* ]]; then
    echo "${label} must not contain a single quote: ${value}" >&2
    exit 1
  fi
  # ckpt_path is passed as ckpt_path='$CKPT' so the filename may contain '='.
  # Other overrides are bare Hydra values and must not contain '='.
  if [[ "${label}" != "CKPT" && "${value}" == *"="* ]]; then
    echo "${label} must not contain '=': ${value}" >&2
    exit 1
  fi
}

refuse_quote CKPT "${CKPT}"
refuse_quote SID_PATH "${SID_PATH}"
refuse_quote PREDICT_DIR "${PREDICT_DIR}"
refuse_quote DATA_DIR "${DATA_DIR}"
refuse_quote FEATURES "${FEATURES}"

if [[ -z "${SID_PATH}" || ! -f "${SID_PATH}" ]]; then
  echo "Set SID_PATH to an existing semantic-id file (.pt or JSONL)." >&2
  exit 1
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  echo "The 2026-10-17 file lives on the GPU machine and is not in git:" >&2
  echo "  third_party/GRID/logs/train/runs/2026-10-17/00-35-10/checkpoints/checkpoint_epoch=000_step=018800.ckpt" >&2
  echo "Refusing to invent first-step logits." >&2
  exit 1
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  if [[ ! -f "${SAMPLES}" ]]; then
    echo "P0 samples not found: ${SAMPLES}" >&2
    echo "Run scripts/p0_eval_audit.sh first, or set SAMPLES, or SKIP_BUILD=1 with an existing predict folder." >&2
    exit 1
  fi
  "${PY}" -m budget_sid build_leak_safe_predict \
    --samples "${SAMPLES}" \
    --output-dir "${PREDICT_DIR}" \
    --semantic-id-path "${SID_PATH}"
fi

if [[ ! -f "${PREDICT_DIR}/history_only.tfrecord.gz" || ! -f "${PREDICT_DIR}/predict_index.jsonl" || ! -f "${PREDICT_DIR}/manifest.json" ]]; then
  echo "History-only predict set is missing under ${PREDICT_DIR}." >&2
  echo "Expected history_only.tfrecord.gz, predict_index.jsonl, and manifest.json." >&2
  exit 1
fi

echo "first-step dump on history-only ${PREDICT_DIR}"
echo "override ckpt_path='${CKPT}'"
echo "dataloader data_loading.predict_dataloader_config.dataloader.num_workers=0 timeout=0"
echo "data_folder data_loading.predict_dataloader_config.dataloader.data_folder=${PREDICT_DIR}"
echo "features ${FEATURES}"
echo "aligns with ${PER_REQUEST} on request_id in predict_index order"
echo "P3 stays gated until benefit beats random and entropy on positive-Δ mass and nDCG@20 under the same quota."

replay_line=(
  "${PY}" -m budget_sid replay_budgets
  --per-request "${PER_REQUEST}"
  --summary "${OUT_DIR}/summary.json"
  --features "${FEATURES}"
  --user-protocol temporal_loo
  --seeds 42,43,44
  --output-dir "${ROOT}/outputs/p2_replay"
)
echo "after the dump: ${replay_line[*]}"

dump_cmd=(
  "${PY}" -m budget_sid dump_first_step_features
  --predict-dir "${PREDICT_DIR}"
  --output "${FEATURES}"
  --ckpt "${CKPT}"
  --semantic-id-path "${SID_PATH}"
  --data-dir "${DATA_DIR}"
  --grid-dir "${GRID_DIR}"
)
if [[ -f "${PER_REQUEST}" ]]; then
  dump_cmd+=(--per-request "${PER_REQUEST}")
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  dump_cmd+=(--dry-run)
fi

"${dump_cmd[@]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; model was not loaded and ${FEATURES} was not written."
  exit 0
fi

if [[ ! -f "${FEATURES}" ]]; then
  echo "dump returned without ${FEATURES}. Refusing to invent a feature file." >&2
  exit 1
fi

echo "Wrote ${FEATURES}. Join it with replay_budgets --features. P3 is not started."
