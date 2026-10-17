#!/usr/bin/env bash
# Export leak-safe Beauty eval samples, audit them, and optionally score beams.
#
# Does not train TIGER or MiniLM. Does not power the machine off.
# On SeetaCloud the checkout is /root/autodl-tmp/Budget-SID.
#
#   DATA_DIR=/root/autodl-tmp/Budget-SID/third_party/GRID/data/amazon_data/beauty \
#   CKPT=/root/autodl-tmp/Budget-SID/third_party/GRID/logs/train/runs/2026-10-17/00-35-10/checkpoints/checkpoint_epoch=000_step=018800.ckpt \
#   SID_PATH=/path/to/semantic_ids.pt \
#   bash scripts/p0_eval_audit.sh
#
# Rankings from the 2026-10-17 tiger_inference_flat run are not leak-safe.
# Pass them only to inspect; the summary will set auditable_for_delta_ndcg=false
# unless RANKING_SOURCE=leak_safe_history.

set -euo pipefail

log_instance_stays_on() {
  echo "SKIP_SHUTDOWN: instance stays ON. No autodl poweroff. No shutdown."
}
trap log_instance_stays_on EXIT

if [[ -n "${ROOT_DIR:-}" ]]; then
  ROOT="${ROOT_DIR}"
elif [[ -d /root/autodl-tmp/Budget-SID ]]; then
  ROOT=/root/autodl-tmp/Budget-SID
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

DATA_DIR="${DATA_DIR:-${ROOT}/third_party/GRID/data/amazon_data/beauty}"
OUT_DIR="${OUT_DIR:-${ROOT}/outputs/p0_eval}"
CKPT="${CKPT:-${ROOT}/third_party/GRID/logs/train/runs/2026-10-17/00-35-10/checkpoints/checkpoint_epoch=000_step=018800.ckpt}"
SID_PATH="${SID_PATH:-}"
RANKINGS_DIR="${RANKINGS_DIR:-}"
RANKING_SOURCE="${RANKING_SOURCE:-grid_tiger_inference_flat}"
SID_MAP="${SID_MAP:-}"

mkdir -p "${OUT_DIR}"

hash_args=()
if [[ -n "${SID_PATH}" ]]; then
  hash_args+=(--semantic-id-path "${SID_PATH}")
elif [[ -n "${SEMANTIC_ID_HASH:-}" ]]; then
  hash_args+=(--semantic-id-hash "${SEMANTIC_ID_HASH}")
else
  echo "Set SID_PATH or SEMANTIC_ID_HASH." >&2
  exit 1
fi

python -m budget_sid export_eval_samples \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUT_DIR}" \
  --checkpoint "${CKPT}" \
  "${hash_args[@]}"

python -m budget_sid audit_eval_leakage \
  --samples "${OUT_DIR}/eval_samples.jsonl" \
  --output "${OUT_DIR}/audit_report.json"

if [[ -n "${RANKINGS_DIR}" ]]; then
  score_args=(
    --samples "${OUT_DIR}/eval_samples.jsonl"
    --rankings-dir "${RANKINGS_DIR}"
    --output-dir "${OUT_DIR}"
    --ranking-source "${RANKING_SOURCE}"
  )
  if [[ -n "${SID_MAP}" ]]; then
    score_args+=(--sid-map "${SID_MAP}")
  fi
  python -m budget_sid score_beam_ndcg "${score_args[@]}"
fi

echo "P0 eval files are in ${OUT_DIR}"
