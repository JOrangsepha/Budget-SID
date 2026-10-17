#!/usr/bin/env bash
# Leak-safe fixed beams 20/40/60 for offline ΔNDCG@20.
#
# P0 already wrote history that does not contain the target event
# (outputs/p0_eval/eval_samples.jsonl). This script encodes that history
# as a GRID predict folder, runs tiger_inference_flat on it, and scores
# with --ranking-source leak_safe_history.
#
# The 2026-10-17 tiger_inference_flat pickles are not inputs. Their collate
# kept the target event. Do not set LEGACY_BEAM_DIR or OLD_RANKINGS_DIR.
#
# SeetaCloud, from /root/autodl-tmp/Budget-SID, after the P0 export:
#
#   SID_PATH=/path/to/semantic_ids.pt bash scripts/run_leak_safe_beams.sh
#
# CKPT defaults to:
#   third_party/GRID/logs/train/runs/2026-10-17/00-35-10/checkpoints/checkpoint_epoch=000_step=018800.ckpt
#
# DRY_RUN=1 builds the history-only TFRecords and prints the GPU command.
# SKIP_BUILD=1 reuses an existing predict folder. This script prints
# SKIP_SHUTDOWN and does not power the machine off.
#
# Hydra, same fixes as scripts/run_after_tiger.sh:
#   ckpt_path='$CKPT'
#   data_loading.predict_dataloader_config.dataloader.num_workers=0
#   data_loading.predict_dataloader_config.dataloader.timeout=0
#   data_loading.predict_dataloader_config.dataloader.data_folder=<history-only dir>
# Do not override test_dataloader_config: tiger_inference_flat has no such key.

set -euo pipefail

log_instance_stays_on() {
  echo "SKIP_SHUTDOWN: instance stays ON. No autodl poweroff. No shutdown."
}
trap log_instance_stays_on EXIT

if [[ -n "${LEGACY_BEAM_DIR:-}${OLD_RANKINGS_DIR:-}" ]]; then
  echo "Refusing legacy tiger_inference_flat rankings." >&2
  echo "The 2026-10-17 beams kept the target event. They are not leak-safe ΔNDCG inputs." >&2
  echo "Unset LEGACY_BEAM_DIR and OLD_RANKINGS_DIR. This script builds new rankings from history." >&2
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
RANK_DIR="${RANK_DIR:-${OUT_DIR}/rankings}"
MARKER_DIR="${MARKER_DIR:-${ROOT}/artifacts/markers}"
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
refuse_quote OUT_DIR "${OUT_DIR}"

if [[ "${EXPORT:-0}" == "1" ]]; then
  if [[ -z "${SID_PATH}" || ! -f "${CKPT}" ]]; then
    echo "EXPORT=1 needs CKPT and SID_PATH." >&2
    exit 1
  fi
  sample_dir="$(dirname "${SAMPLES}")"
  mkdir -p "${sample_dir}"
  "${PY}" -m budget_sid export_eval_samples \
    --data-dir "${DATA_DIR}" \
    --output-dir "${sample_dir}" \
    --checkpoint "${CKPT}" \
    --semantic-id-path "${SID_PATH}"
  SAMPLES="${sample_dir}/eval_samples.jsonl"
fi

if [[ ! -f "${SAMPLES}" ]]; then
  echo "P0 samples not found: ${SAMPLES}" >&2
  echo "Run scripts/p0_eval_audit.sh first, or set SAMPLES=outputs/p0_eval/eval_samples.jsonl." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}" "${RANK_DIR}" "${MARKER_DIR}"

if [[ -z "${SID_PATH}" || ! -f "${SID_PATH}" ]]; then
  echo "Set SID_PATH to an existing semantic-id file (.pt or JSONL)." >&2
  exit 1
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  "${PY}" -m budget_sid build_leak_safe_predict \
    --samples "${SAMPLES}" \
    --output-dir "${PREDICT_DIR}" \
    --semantic-id-path "${SID_PATH}"
fi

if [[ ! -f "${PREDICT_DIR}/history_only.tfrecord.gz" || ! -f "${PREDICT_DIR}/predict_index.jsonl" ]]; then
  echo "History-only predict set is missing under ${PREDICT_DIR}." >&2
  echo "Expected history_only.tfrecord.gz and predict_index.jsonl." >&2
  exit 1
fi

print_infer_line() {
  local k="$1"
  local pickle_dir="$2"
  echo "fixed beam top_k=${k} on history-only ${PREDICT_DIR}"
  echo "override ckpt_path='${CKPT}'"
  echo "dataloader data_loading.predict_dataloader_config.dataloader.num_workers=0 timeout=0"
  echo "data_folder data_loading.predict_dataloader_config.dataloader.data_folder=${PREDICT_DIR}"
  echo "pickle callbacks.pickle_writer.output_dir=${pickle_dir}"
}

run_one() {
  local k="$1"
  local marker="${MARKER_DIR}/leak_safe_beam_${k}.done"
  local log="${MARKER_DIR}/leak_safe_beam_${k}.log"
  local pickle_dir="${OUT_DIR}/grid_raw/beam_${k}"
  local hydra_ckpt="ckpt_path='$CKPT'"
  local rc
  print_infer_line "${k}" "${pickle_dir}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1; not starting GPU inference for top_k=${k}"
    return 0
  fi
  if [[ ! -d "${GRID_DIR}" ]]; then
    echo "GRID checkout missing at ${GRID_DIR}. Run scripts/pin_grid.sh first." >&2
    exit 1
  fi
  if [[ ! -f "${CKPT}" ]]; then
    echo "Checkpoint not found: ${CKPT}" >&2
    echo "The 2026-10-17 file lives on the GPU machine and is not in git:" >&2
    echo "  third_party/GRID/logs/train/runs/2026-10-17/00-35-10/checkpoints/checkpoint_epoch=000_step=018800.ckpt" >&2
    exit 1
  fi
  if [[ -f "${marker}" && "${FORCE:-0}" != "1" ]]; then
    echo "already done: leak_safe_beam_${k}"
  else
    mkdir -p "${pickle_dir}"
    # Leave the pickle writer's tensor merge on. The pinned writer still
    # references that tensor when the flag is false, which raises after the
    # pickle is written. predict_key is 0..n-1, so the default merge is in range.
    # export_leak_safe_rankings reads merged_predictions.pkl, not the tensor.
    set +e
    (
      cd "${GRID_DIR}"
      "${PY}" -m src.inference experiment=tiger_inference_flat \
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
        data_loading.predict_dataloader_config.dataloader.timeout=0 \
        "data_loading.predict_dataloader_config.dataloader.data_folder=${PREDICT_DIR}" \
        "callbacks.pickle_writer.output_dir=${pickle_dir}"
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
  fi
  if [[ "${SKIP_EXPORT:-0}" == "1" ]]; then
    echo "SKIP_EXPORT=1; not writing beam_${k}.jsonl"
    return 0
  fi
  local pickle="${pickle_dir}/merged_predictions.pkl"
  if [[ ! -f "${pickle}" ]]; then
    echo "GRID did not write ${pickle}." >&2
    echo "tiger_inference_flat should merge rows keyed by user_id (the predict_key)." >&2
    exit 1
  fi
  "${PY}" -m budget_sid export_leak_safe_rankings \
    --predictions "${pickle}" \
    --index "${PREDICT_DIR}/predict_index.jsonl" \
    --sid-map "${SID_PATH}" \
    --beam "${k}" \
    --output "${RANK_DIR}/beam_${k}.jsonl"
}

IFS=' ' read -r -a TOP_K_LIST <<< "${TOP_KS:-20 40 60}"
for k in "${TOP_K_LIST[@]}"; do
  run_one "${k}"
done

score_cmd=(
  "${PY}" -m budget_sid score_beam_ndcg
  --samples "${SAMPLES}"
  --rankings-dir "${RANK_DIR}"
  --output-dir "${OUT_DIR}"
  --ranking-source leak_safe_history
)
echo "score ${score_cmd[*]}"

if [[ "${DRY_RUN:-0}" == "1" || "${SKIP_SCORE:-0}" == "1" ]]; then
  echo "Rankings directory: ${RANK_DIR}"
  echo "Leak-safe beams were not scored in this invocation."
  exit 0
fi

missing=0
for k in 20 40 60; do
  if [[ ! -f "${RANK_DIR}/beam_${k}.jsonl" ]]; then
    echo "missing ${RANK_DIR}/beam_${k}.jsonl" >&2
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  echo "score_beam_ndcg waits until beam 20, 40, and 60 have been exported." >&2
  exit 1
fi

"${score_cmd[@]}"
echo "Leak-safe beam scores are in ${OUT_DIR}. auditable_for_delta_ndcg requires ranking_source leak_safe_history."
