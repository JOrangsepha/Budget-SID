#!/usr/bin/env bash
# Fixed-beam TIGER inference after tiger_train_flat.
#
# P0: collate_fn_inference_for_sequence keeps the last event. Rankings from
# this script are not leak-safe ΔNDCG inputs. Export history with
# scripts/p0_eval_audit.sh, then run scripts/run_leak_safe_beams.sh.
# See docs/P0_EVAL_AUDIT.md. Do not score the 2026-10-17 pickles as ΔNDCG.
#
# The 2026-10-17 SeetaCloud Beauty run already finished (top_k 20/40/60,
# exit 0). This script records that invocation. It does not power the
# machine off. A .done marker is written only when python returns 0.
#
# Hydra pitfalls baked in:
#   ckpt_path='$CKPT'   (filename contains "=")
#   data_loading.predict_dataloader_config.dataloader.num_workers=0
#   data_loading.predict_dataloader_config.dataloader.timeout=0
# Do not override test_dataloader_config: tiger_inference_flat has no such key.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRID_DIR="${GRID_DIR:-${ROOT}/third_party/GRID}"
DATA_DIR="${DATA_DIR:-${GRID_DIR}/data/amazon_data/beauty}"
SID_PATH="${SID_PATH:-${SEMANTIC_ID_PATH:-}}"
MARKER_DIR="${MARKER_DIR:-${ROOT}/artifacts/markers}"
DEFAULT_CKPT="${GRID_DIR}/logs/train/runs/2026-10-17/00-35-10/checkpoints/checkpoint_epoch=000_step=018800.ckpt"
CKPT="${CKPT:-${DEFAULT_CKPT}}"

log_instance_stays_on() {
  echo "SKIP_SHUTDOWN: instance stays ON. No autodl poweroff. No shutdown."
}
trap log_instance_stays_on EXIT

if [[ "${CKPT}" == *"'"* ]]; then
  echo "CKPT must not contain a single quote: ${CKPT}" >&2
  exit 1
fi
if [[ -z "${SID_PATH}" ]]; then
  echo "Set SID_PATH to the semantic-id tensor from RQ-KMeans inference." >&2
  exit 1
fi
if [[ ! -d "${GRID_DIR}" ]]; then
  echo "GRID checkout missing at ${GRID_DIR}. Run scripts/pin_grid.sh first." >&2
  exit 1
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  echo "Checkpoints are not in git. On the 2026-10-17 SeetaCloud box the file was:" >&2
  echo "  third_party/GRID/logs/train/runs/2026-10-17/00-35-10/checkpoints/checkpoint_epoch=000_step=018800.ckpt" >&2
  exit 1
fi

mkdir -p "${MARKER_DIR}"
IFS=' ' read -r -a TOP_K_LIST <<< "${TOP_KS:-20 40 60}"

run_one() {
  local k="$1"
  local marker="${MARKER_DIR}/tiger_infer_top${k}.done"
  local log="${MARKER_DIR}/tiger_infer_top${k}.log"
  local hydra_ckpt="ckpt_path='$CKPT'"
  local rc
  echo "fixed beam top_k=${k}"
  echo "override ${hydra_ckpt}"
  echo "dataloader data_loading.predict_dataloader_config.dataloader.num_workers=0 timeout=0"
  set +e
  (
    cd "${GRID_DIR}"
    python -m src.inference experiment=tiger_inference_flat \
      "data_dir=${DATA_DIR}" \
      "semantic_id_path=${SID_PATH}" \
      "${hydra_ckpt}" \
      num_hierarchies=4 \
      model.huggingface_model.config.d_model=128 \
      model.huggingface_model.config.num_layers=4 \
      model.decoder.config.num_layers=4 \
      "model.top_k_for_generation=${k}" \
      trainer.strategy=auto \
      trainer.devices=1 \
      trainer.num_nodes=1 \
      trainer.sync_batchnorm=false \
      trainer.accelerator=gpu \
      data_loading.predict_dataloader_config.dataloader.num_workers=0 \
      data_loading.predict_dataloader_config.dataloader.timeout=0
  ) >"${log}" 2>&1
  rc=$?
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    date -Iseconds > "${marker}"
    echo "marked ${marker}"
  else
    rm -f "${marker}"
    echo "top_k=${k} rc=${rc}; .done not written. See ${log}" >&2
    exit "${rc}"
  fi
}

for k in "${TOP_K_LIST[@]}"; do
  run_one "${k}"
done

echo "fixed-beam inference finished."
