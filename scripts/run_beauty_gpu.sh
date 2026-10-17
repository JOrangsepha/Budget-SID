#!/usr/bin/env bash
# One-GPU Beauty pipeline.
#   frozen MiniLM cache (you provide EMBEDDING_PATH)
#   → RQ-KMeans train / inference
#   → tiger_train_flat
#   → fixed-beam infer at top_k 20/40/60
#
# Benefit-aware ΔNDCG@20 routing is not part of this script.
# The 2026-10-17 SeetaCloud run already finished; this file is the entry
# point, and running it here is a new job. It never powers the machine off.
# A .done marker is written only when that stage's command returns 0.
#
# Hydra pitfalls baked in (same as scripts/run_after_tiger.sh):
#   ckpt_path='$CKPT'   (filename contains "=")
#   data_loading.predict_dataloader_config — not test_dataloader_config
#   data_loading.predict_dataloader_config.dataloader.num_workers=0
#   data_loading.predict_dataloader_config.dataloader.timeout=0
#
# SKIP_TRAIN=1  reuse CKPT and SID_PATH and only run fixed-beam inference.
# SKIP_PIN=1    do not clone or fetch GRID (tests / already pinned).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRID_DIR="${GRID_DIR:-${ROOT}/third_party/GRID}"
MARKER_DIR="${MARKER_DIR:-${ROOT}/artifacts/markers}"
DATA_DIR="${DATA_DIR:-}"
EMBEDDING_PATH="${EMBEDDING_PATH:-}"
SID_PATH="${SID_PATH:-${SEMANTIC_ID_PATH:-}}"
CKPT="${CKPT:-}"

log_instance_stays_on() {
  echo "SKIP_SHUTDOWN: instance stays ON. No autodl poweroff. No shutdown."
}
trap log_instance_stays_on EXIT

mkdir -p "${MARKER_DIR}"

mark_if_ok() {
  local rc="$1"
  local marker="$2"
  if [[ "${rc}" -eq 0 ]]; then
    date -Iseconds > "${marker}"
    echo "marked ${marker}"
  else
    rm -f "${marker}"
    echo "$(basename "${marker}" .done) rc=${rc}; .done not written" >&2
    exit "${rc}"
  fi
}

run_in_grid() {
  local name="$1"
  shift
  local marker="${MARKER_DIR}/${name}.done"
  local log="${MARKER_DIR}/${name}.log"
  local rc
  if [[ -f "${marker}" && "${FORCE:-0}" != "1" ]]; then
    echo "already done: ${name}"
    return 0
  fi
  echo "stage ${name}"
  set +e
  (
    cd "${GRID_DIR}"
    "$@"
  ) >"${log}" 2>&1
  rc=$?
  set -e
  mark_if_ok "${rc}" "${marker}"
}

latest_ckpt() {
  local found=""
  if [[ -d "${GRID_DIR}/logs" || -d "${GRID_DIR}/outputs" ]]; then
    found="$(find "${GRID_DIR}/logs" "${GRID_DIR}/outputs" -type f -name '*.ckpt' -printf '%T@\t%p\n' 2>/dev/null | sort -n | tail -1 | cut -f2- || true)"
  fi
  printf '%s' "${found}"
}

refuse_flan() {
  local blob
  blob="$(printf '%s' "${EMBEDDING_PATH}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${blob}" == *flan-t5* ]]; then
    echo "Refusing FLAN-T5 embeddings. Cache frozen sentence-transformers/all-MiniLM-L6-v2 (384-d)." >&2
    exit 1
  fi
}

# P0: these beams use tiger_inference_flat, whose collate keeps the target
# event. Score them only after scripts/p0_eval_audit.sh. See docs/P0_EVAL_AUDIT.md.
run_fixed_beams() {
  local k hydra_ckpt rc marker log
  IFS=' ' read -r -a TOP_K_LIST <<< "${TOP_KS:-20 40 60}"
  for k in "${TOP_K_LIST[@]}"; do
    marker="${MARKER_DIR}/tiger_infer_top${k}.done"
    log="${MARKER_DIR}/tiger_infer_top${k}.log"
    hydra_ckpt="ckpt_path='$CKPT'"
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
  done
}

if [[ "${SKIP_PIN:-0}" == "1" ]]; then
  echo "SKIP_PIN=1; not cloning GRID"
else
  set +e
  bash "${ROOT}/scripts/pin_grid.sh"
  pin_rc=$?
  set -e
  mark_if_ok "${pin_rc}" "${MARKER_DIR}/pin_grid.done"
fi

if [[ ! -d "${GRID_DIR}" ]]; then
  echo "GRID checkout missing at ${GRID_DIR}." >&2
  exit 1
fi

if [[ -z "${DATA_DIR}" ]]; then
  if [[ -d "${ROOT}/data/amazon_data/beauty/training" ]]; then
    DATA_DIR="${ROOT}/data/amazon_data/beauty"
  else
    DATA_DIR="${GRID_DIR}/data/amazon_data/beauty"
  fi
fi

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  if [[ -z "${CKPT}" || -z "${SID_PATH}" ]]; then
    echo "SKIP_TRAIN=1 needs CKPT and SID_PATH." >&2
    exit 1
  fi
  if [[ ! -f "${CKPT}" ]]; then
    echo "Checkpoint not found: ${CKPT}" >&2
    exit 1
  fi
  run_fixed_beams
  echo "fixed-beam inference finished."
  exit 0
fi

if [[ -z "${EMBEDDING_PATH}" || ! -f "${EMBEDDING_PATH}" ]]; then
  echo "Set EMBEDDING_PATH to a frozen MiniLM 384-d cache file. Do not run FLAN-T5-XL." >&2
  exit 1
fi
refuse_flan

for split in training evaluation testing items; do
  if [[ ! -d "${DATA_DIR}/${split}" ]]; then
    echo "Missing ${DATA_DIR}/${split}. Beauty stays on the machine, not in git." >&2
    exit 1
  fi
done

if [[ "${CKPT}" == *"'"* || "${EMBEDDING_PATH}" == *"'"* ]]; then
  echo "Paths must not contain a single quote." >&2
  exit 1
fi

run_in_grid rkmeans_train \
  python -m src.train experiment=rkmeans_train_flat \
  "data_dir=${DATA_DIR}" \
  "embedding_path=${EMBEDDING_PATH}" \
  embedding_dim=384 \
  num_hierarchies=3 \
  codebook_width=256 \
  trainer.strategy=auto \
  trainer.devices=1 \
  trainer.num_nodes=1 \
  trainer.sync_batchnorm=false \
  trainer.accelerator=gpu

if [[ -n "${RKMEANS_CKPT:-}" ]]; then
  CKPT="${RKMEANS_CKPT}"
elif [[ -f "${MARKER_DIR}/rkmeans_train.ckptpath" && "${FORCE:-0}" != "1" ]]; then
  CKPT="$(cat "${MARKER_DIR}/rkmeans_train.ckptpath")"
else
  CKPT="$(latest_ckpt)"
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "RQ-KMeans checkpoint not found. Set RKMEANS_CKPT." >&2
  exit 1
fi
printf '%s\n' "${CKPT}" > "${MARKER_DIR}/rkmeans_train.ckptpath"
hydra_ckpt="ckpt_path='$CKPT'"

run_in_grid rkmeans_infer \
  python -m src.inference experiment=rkmeans_inference_flat \
  "data_dir=${DATA_DIR}" \
  "embedding_path=${EMBEDDING_PATH}" \
  "${hydra_ckpt}" \
  embedding_dim=384 \
  num_hierarchies=3 \
  codebook_width=256 \
  trainer.strategy=auto \
  trainer.devices=1 \
  trainer.num_nodes=1 \
  trainer.sync_batchnorm=false \
  trainer.accelerator=gpu \
  data_loading.datamodule.predict_dataloader_config.num_workers=0 \
  data_loading.datamodule.predict_dataloader_config.timeout=0 \
  data_loading.datamodule.predict_dataloader_config.persistent_workers=false

if [[ -z "${SID_PATH}" ]]; then
  echo "RQ-KMeans inference returned 0. Set SID_PATH to the semantic-id tensor, then re-run." >&2
  echo "tiger_train .done was not written." >&2
  exit 1
fi

run_in_grid tiger_train \
  python -m src.train experiment=tiger_train_flat \
  "data_dir=${DATA_DIR}" \
  "semantic_id_path=${SID_PATH}" \
  num_hierarchies=4 \
  model.huggingface_model.config.d_model=128 \
  model.huggingface_model.config.num_layers=4 \
  model.decoder.config.num_layers=4 \
  trainer.accumulate_grad_batches=4 \
  data_loading.train_dataloader_config.dataloader.batch_size_per_device=16 \
  trainer.strategy=auto \
  trainer.devices=1 \
  trainer.num_nodes=1 \
  trainer.sync_batchnorm=false \
  trainer.accelerator=gpu

if [[ -n "${TIGER_CKPT:-}" ]]; then
  CKPT="${TIGER_CKPT}"
elif [[ -f "${MARKER_DIR}/tiger_train.ckptpath" && "${FORCE:-0}" != "1" && -f "${MARKER_DIR}/tiger_train.done" ]]; then
  CKPT="$(cat "${MARKER_DIR}/tiger_train.ckptpath")"
else
  CKPT="$(latest_ckpt)"
  printf '%s\n' "${CKPT}" > "${MARKER_DIR}/tiger_train.ckptpath"
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "TIGER checkpoint not found. Set TIGER_CKPT or CKPT." >&2
  exit 1
fi

run_fixed_beams
echo "Beauty GPU stages finished. Benefit-aware stop-gate is still not wired."
