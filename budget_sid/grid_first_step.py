"""Hierarchy-0 logits from the pinned GRID generate path.

``SemanticIDEncoderDecoder.generate`` at commit ``2fe3475`` runs

    encoder_forward_pass
    decoder_forward_pass(future_ids=None)   # bos token, empty cache
    decoder.decoder_mlp[0](decoder_output[:, -1, :])

and passes that matrix to ``_beam_search_one_step``, which softmaxes it and
does not return it. ``predict_step`` writes semantic ids only. The pickle
files under ``outputs/leak_safe_beams`` therefore have no logits.

This module calls those three public pieces and keeps the matrix. It does not
edit the GRID checkout, does not change the pin, and does not run hierarchies
1..3. ``top_k_for_generation`` is unused here: the first-step scores are
computed before the beam is expanded, so one dump serves beams 20, 40, and 60.

If ``should_check_prefix`` is set, the same first-step mask ``generate`` applies
before its softmax is applied here, still before ``extract_features`` softmaxes.
Invalid codes become ``-inf``. They are not replaced with zeros.

``SequenceDataModule.setup`` reads ``self.trainer``. This dump does not call
``Trainer.predict`` (that path keeps semantic ids and drops the logits), so a
Trainer is attached to the module and the datamodule before ``setup``.
``predict_dataloader`` returns a 1-tuple; Lightning unpacks it, and this dump
does the same before iterating batches.
"""

from __future__ import annotations

from pathlib import Path

from budget_sid.constants import GRID_COMMIT
from budget_sid.errors import MissingFirstStepLogits
from budget_sid.first_step_dump import (
    LOGITS_SOURCE_GRID,
    feature_rows_from_keyed_logits,
    require_logit_matrix,
    require_logit_vector,
)

HYDRA_WORKERS = "data_loading.predict_dataloader_config.dataloader.num_workers=0"
HYDRA_TIMEOUT = "data_loading.predict_dataloader_config.dataloader.timeout=0"
HYDRA_DATA_FOLDER = "data_loading.predict_dataloader_config.dataloader.data_folder"


def hydra_overrides(*, data_dir: str, sid_path: str, ckpt: str, predict_dir: str) -> list[str]:
    """Same ``tiger_inference_flat`` overrides as ``scripts/run_leak_safe_beams.sh``.

    ``ckpt_path`` is quoted because the 2026-10-17 filename contains ``=``.
    The data folder is the history-only predict directory, not ``testing``.
    """
    for label, value in (
        ("data_dir", data_dir),
        ("semantic_id_path", sid_path),
        ("predict_dir", predict_dir),
        ("ckpt", ckpt),
    ):
        if value is None or value == "":
            raise MissingFirstStepLogits(f"{label} is empty. Refusing to invent logits.")
        if "'" in str(value):
            raise MissingFirstStepLogits(f"{label} must not contain a single quote: {value}")
        if label != "ckpt" and "=" in str(value):
            raise MissingFirstStepLogits(
                f"{label} must not contain '=': {value}. "
                "ckpt_path is the override that is quoted, because the checkpoint filename contains '='."
            )
    return [
        "experiment=tiger_inference_flat",
        f"data_dir={data_dir}",
        f"semantic_id_path={sid_path}",
        f"ckpt_path='{ckpt}'",
        "num_hierarchies=4",
        "model.huggingface_model.config.d_model=128",
        "model.huggingface_model.config.num_layers=4",
        "model.decoder.config.num_layers=4",
        "model.top_k_for_generation=20",
        "trainer.strategy=auto",
        "trainer.devices=1",
        "trainer.num_nodes=1",
        "trainer.sync_batchnorm=false",
        "trainer.accelerator=gpu",
        HYDRA_WORKERS,
        HYDRA_TIMEOUT,
        f"{HYDRA_DATA_FOLDER}={predict_dir}",
    ]


def _coerce_predict_key(value) -> int:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "reshape"):
        flat = value.reshape(-1)
        width = int(flat.shape[0])
        if width != 1:
            raise MissingFirstStepLogits(
                f"user_id entry has {width} values; expected the scalar predict_key. "
                "Refusing to guess which element is the request."
            )
        value = flat[0]
    if hasattr(value, "item"):
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise MissingFirstStepLogits(
            f"user_id {value!r} is not a predict_key. Refusing to align by batch position."
        ) from exc


def predict_keys_from_batch(batch) -> list[int]:
    """GRID stores the leak-safe ``predict_key`` on ``user_id_list``."""
    raw = getattr(batch, "user_id_list", None)
    if raw is None and isinstance(batch, dict):
        raw = batch.get("user_id_list")
    if raw is None:
        raise MissingFirstStepLogits(
            "predict batch has no user_id_list. "
            "tiger_inference_flat stores the leak-safe predict_key there. "
            "Refusing to align logits by batch position."
        )
    if hasattr(raw, "detach") and hasattr(raw, "ndim") and int(getattr(raw, "ndim", 1)) == 0:
        return [_coerce_predict_key(raw)]
    return [_coerce_predict_key(item) for item in raw]


def batch_model_inputs(module, batch):
    sequences = getattr(batch, "transformed_sequences", None)
    if sequences is None and isinstance(batch, dict):
        sequences = batch.get("transformed_sequences")
    if not sequences:
        raise MissingFirstStepLogits(
            "predict batch has no transformed_sequences. Refusing to invent logits."
        )
    mapping = getattr(module, "feature_to_model_input_map", None) or {
        "sequence_data": "input_ids",
        "user_id": "user_id",
    }
    mapped = {mapping.get(key, key): value for key, value in sequences.items()}
    if mapped.get("input_ids") is None:
        raise MissingFirstStepLogits(
            "batch is missing sequence_data/input_ids. Refusing to invent logits."
        )
    mask = getattr(batch, "mask", None)
    if mask is None and isinstance(batch, dict):
        mask = batch.get("mask")
    if mask is None:
        raise MissingFirstStepLogits(
            "batch is missing the attention mask. Refusing to invent logits."
        )
    return mask, mapped["input_ids"], mapped.get("user_id")


def _move(value, device):
    if value is None or device is None or not hasattr(value, "to"):
        return value
    return value.to(device)


def _apply_prefix_mask(module, logits):
    """Match ``_beam_search_one_step`` when ``generated_ids is None``.

    The mask runs before softmax. ``extract_features`` softmaxes afterwards.
    """
    if not getattr(module, "should_check_prefix", False):
        return logits
    checker = getattr(module, "_check_valid_prefix", None)
    if checker is None:
        raise MissingFirstStepLogits(
            "should_check_prefix is true but _check_valid_prefix is missing. "
            "Refusing to dump unmasked logits or zeros."
        )
    width = int(logits.shape[-1])
    if hasattr(logits, "device") and hasattr(logits, "new_tensor"):
        import torch

        prefix = torch.arange(width, device=logits.device).unsqueeze(1)
        valid = checker(prefix)
        if valid is None:
            raise MissingFirstStepLogits(
                "prefix check returned no mask. Refusing to invent logits."
            )
        masked = logits.clone()
        valid_flat = valid.reshape(-1)
        if int(valid_flat.shape[0]) != width:
            raise MissingFirstStepLogits(
                f"prefix mask width {int(valid_flat.shape[0])} != logit width {width}. "
                "Refusing to pad or truncate."
            )
        masked[:, ~valid_flat] = float("-inf")
        return masked
    import numpy as np

    prefix = np.arange(width).reshape(-1, 1)
    valid = checker(prefix)
    if valid is None:
        raise MissingFirstStepLogits(
            "prefix check returned no mask. Refusing to invent logits."
        )
    array = np.array(require_logit_matrix(logits), copy=True)
    valid_np = np.asarray(valid).reshape(-1).astype(bool)
    if valid_np.shape[0] != array.shape[1]:
        raise MissingFirstStepLogits(
            f"prefix mask width {valid_np.shape[0]} != logit width {array.shape[1]}. "
            "Refusing to pad or truncate."
        )
    array[:, ~valid_np] = -np.inf
    return array


def first_step_candidate_logits(module, *, attention_mask, input_ids, user_id):
    """Hierarchy-0 ``candidate_logits`` from the same calls ``generate`` makes.

    An empty KV cache and ``future_ids=None`` is the bos-token prompt.
    ``_is_kv_cache_valid(None)`` is false at pin ``2fe3475``, which is the
    same branch as the empty ``EncoderDecoderCache`` ``generate`` allocates
    before the loop.
    """
    if not hasattr(module, "encoder_forward_pass") or not hasattr(module, "decoder_forward_pass"):
        raise MissingFirstStepLogits(
            "The pinned SemanticIDEncoderDecoder does not expose encoder_forward_pass "
            "and decoder_forward_pass. predict_step returns semantic ids only. "
            "Refusing to invent first-step logits."
        )
    decoder = getattr(module, "decoder", None)
    mlp = getattr(decoder, "decoder_mlp", None) if decoder is not None else None
    if mlp is None or len(mlp) < 1 or not callable(mlp[0]):
        raise MissingFirstStepLogits(
            "decoder.decoder_mlp[0] is missing. "
            "generate() uses that linear map for hierarchy-0 candidate_logits and then discards it. "
            "Refusing to invent zeros."
        )
    if attention_mask is None or input_ids is None:
        raise MissingFirstStepLogits(
            "encoder inputs are missing. Refusing to invent logits."
        )
    encoded = module.encoder_forward_pass(
        attention_mask=attention_mask,
        input_ids=input_ids,
        user_id=user_id,
    )
    if not isinstance(encoded, tuple) or len(encoded) < 2 or encoded[0] is None:
        raise MissingFirstStepLogits(
            "encoder_forward_pass did not return an encoder state. Refusing to invent logits."
        )
    encoder_output, encoder_attention_mask = encoded[0], encoded[1]
    decoded = module.decoder_forward_pass(
        future_ids=None,
        encoder_output=encoder_output,
        attention_mask_for_encoder=encoder_attention_mask,
        use_cache=True,
        past_key_values=None,
    )
    if not isinstance(decoded, tuple) or decoded[0] is None:
        raise MissingFirstStepLogits(
            "decoder_forward_pass did not return a hidden state. Refusing to invent logits."
        )
    decoder_output = decoded[0]
    try:
        latest = decoder_output[:, -1, :]
    except (TypeError, IndexError, AttributeError) as exc:
        raise MissingFirstStepLogits(
            "decoder hidden state has no last position. Refusing to invent logits. " + str(exc)
        ) from exc
    candidate_logits = mlp[0](latest)
    if candidate_logits is None:
        raise MissingFirstStepLogits(
            "decoder.decoder_mlp[0] returned no logits. Refusing to invent zeros."
        )
    candidate_logits = _apply_prefix_mask(module, candidate_logits)
    matrix = require_logit_matrix(candidate_logits)
    expected = getattr(module, "num_embeddings_per_hierarchy", None)
    if expected is not None and int(matrix.shape[1]) != int(expected):
        raise MissingFirstStepLogits(
            f"first-step logit width {int(matrix.shape[1])} != "
            f"num_embeddings_per_hierarchy {int(expected)}. "
            "Refusing to pad or truncate."
        )
    return matrix


def collect_logits_by_key(module, batches, *, device=None) -> dict[int, np.ndarray]:
    """Run the hierarchy-0 first step and keep one vector per ``predict_key``.

    A missing tensor, an empty loader, or a duplicate key is an error. The
    returned vectors are NumPy copies, so the caller can downcast them after
    the GPU pass without another forward.
    """
    import numpy as np

    logits_by_key: dict[int, np.ndarray] = {}
    eval_fn = getattr(module, "eval", None)
    if callable(eval_fn):
        eval_fn()
    saw_batch = False
    for batch in batches:
        saw_batch = True
        keys = predict_keys_from_batch(batch)
        mask, input_ids, user_id = batch_model_inputs(module, batch)
        matrix = first_step_candidate_logits(
            module,
            attention_mask=_move(mask, device),
            input_ids=_move(input_ids, device),
            user_id=_move(user_id, device),
        )
        if int(matrix.shape[0]) != len(keys):
            raise MissingFirstStepLogits(
                f"logit batch {int(matrix.shape[0])} != user_id_list {len(keys)}. "
                "Refusing to zip by a guessed position."
            )
        for key, vector in zip(keys, matrix):
            if key in logits_by_key:
                raise MissingFirstStepLogits(
                    f"duplicate predict_key {key} in the dataloader. Refusing to keep one at random."
                )
            logits_by_key[key] = np.array(require_logit_vector(vector), dtype=np.float64, copy=True)
    if not saw_batch:
        raise MissingFirstStepLogits(
            "the history-only predict loader yielded no batches. Refusing to invent logits."
        )
    return logits_by_key


def rows_from_batches(module, batches, index_rows, *, device=None) -> list[dict]:
    """Run the first step on each batch and join on ``predict_key``."""
    logits_by_key = collect_logits_by_key(module, batches, device=device)
    return feature_rows_from_keyed_logits(
        index_rows,
        logits_by_key,
        logits_source=LOGITS_SOURCE_GRID,
    )


def _assert_grid_pin(grid_dir: Path) -> None:
    import subprocess

    if not (grid_dir / "src" / "models" / "modules" / "semantic_id" / "tiger_generation_model.py").is_file():
        raise MissingFirstStepLogits(
            f"GRID checkout missing at {grid_dir}. Run scripts/pin_grid.sh. "
            "Refusing to invent first-step logits."
        )
    result = subprocess.run(
        ["git", "-C", str(grid_dir), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    actual = (result.stdout or "").strip()
    if result.returncode != 0 or actual != GRID_COMMIT:
        raise MissingFirstStepLogits(
            f"GRID pin mismatch under {grid_dir}: expected {GRID_COMMIT}, got {actual or 'unreadable'}. "
            "This dump stays on 2fe3475. Refusing to call a different generate()."
        )


def _load_state_dict(model, ckpt: str) -> None:
    import torch

    try:
        payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise MissingFirstStepLogits(
            f"checkpoint {ckpt} has no state_dict. Refusing to invent logits."
        )
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except Exception as exc:
        raise MissingFirstStepLogits(
            "checkpoint state_dict does not match the pinned SemanticIDEncoderDecoder. "
            "Refusing a partial load, which would leave decoder.decoder_mlp at random weights "
            "and look like real logits. " + str(exc)
        ) from exc


def run_grid_first_step_dump(
    *,
    grid_dir: str,
    data_dir: str,
    sid_path: str,
    ckpt: str,
    predict_dir: str,
    index_rows,
) -> list[dict]:
    """Load ``tiger_inference_flat`` and dump hierarchy-0 features.

    Must run on a machine with the pinned checkout, the Beauty checkpoint, and
    CUDA. A missing dependency or a missing tensor raises
    ``MissingFirstStepLogits``.
    """
    logits_by_key = run_grid_first_step_logits_dump(
        grid_dir=grid_dir,
        data_dir=data_dir,
        sid_path=sid_path,
        ckpt=ckpt,
        predict_dir=predict_dir,
    )
    return feature_rows_from_keyed_logits(
        index_rows,
        logits_by_key,
        logits_source=LOGITS_SOURCE_GRID,
    )


def run_grid_first_step_logits_dump(
    *,
    grid_dir: str,
    data_dir: str,
    sid_path: str,
    ckpt: str,
    predict_dir: str,
) -> dict[int, np.ndarray]:
    """Same first step as the feature dump, returning the logit vectors.

    ``load_state_dict(..., strict=True)`` is the only checkpoint load. The
    caller decides whether the predict folder is the full history-only set or
    a validation subset. Keys are the GRID ``user_id`` values (``predict_key``).
    """
    root = Path(grid_dir)
    ckpt_path = Path(ckpt)
    if not ckpt_path.is_file():
        raise MissingFirstStepLogits(
            f"Checkpoint not found: {ckpt_path}. Refusing to invent first-step logits."
        )
    if not Path(sid_path).is_file():
        raise MissingFirstStepLogits(
            f"Semantic-id file not found: {sid_path}. Refusing to invent first-step logits."
        )
    _assert_grid_pin(root)
    import os
    import sys

    previous = Path.cwd()
    os.chdir(root)
    sys.path.insert(0, str(root))
    try:
        return _load_and_collect_logits(
            root=root,
            data_dir=data_dir,
            sid_path=sid_path,
            ckpt=str(ckpt_path),
            predict_dir=predict_dir,
        )
    finally:
        os.chdir(previous)


def attach_lightning_trainer(module, trainer) -> None:
    """Set ``module.trainer`` the way Lightning does before ``setup``.

    Pinned GRID ``SequenceDataModule.setup`` raises when ``self.trainer`` is
    missing. It reads ``world_size`` and ``global_rank`` to assign TFRecord
    files. ``Trainer.predict`` is what normally sets the attribute. This dump
    does not call ``predict``, because that hook drops the logits.
    """
    if module is None or trainer is None:
        raise MissingFirstStepLogits(
            "GRID setup requires a Lightning Trainer on self.trainer. "
            "Refusing to invent logits."
        )
    try:
        module.trainer = trainer
    except Exception as exc:
        raise MissingFirstStepLogits(
            "Could not attach a Lightning Trainer before datamodule.setup. "
            "GRID reads self.trainer. Refusing to invent logits. " + str(exc)
        ) from exc
    if getattr(module, "trainer", None) is None:
        raise MissingFirstStepLogits(
            "self.trainer is still unset after attach. "
            "GRID SequenceDataModule.setup will not assign predict files. "
            "Refusing to invent logits."
        )


def unwrap_predict_dataloader(loader):
    """Return the single loader inside GRID's 1-tuple.

    ``SequenceDataModule.get_dataloader`` returns ``(loader,)``. Lightning
    unpacks that before iteration. Iterating the tuple yields the loader
    itself, so the batch has no ``user_id_list`` and the join is wrong.
    """
    if loader is None:
        raise MissingFirstStepLogits(
            "predict_dataloader returned nothing. Refusing to invent logits."
        )
    if isinstance(loader, tuple):
        if len(loader) != 1 or loader[0] is None:
            raise MissingFirstStepLogits(
                f"predict_dataloader returned a {len(loader)}-tuple. "
                "Pinned GRID returns one loader. Refusing to iterate that tuple as batches."
            )
        return loader[0]
    return loader


def bind_trainer_and_predict_loader(module, datamodule, trainer):
    """Attach ``trainer``, then ``setup('predict')``, then unwrap the loader.

    Attachment happens before ``datamodule.setup``. GRID reads
    ``datamodule.trainer`` inside that hook. The module is attached first,
    matching Lightning's ``Trainer.predict`` order without calling ``predict``.
    """
    attach_lightning_trainer(module, trainer)
    attach_lightning_trainer(datamodule, trainer)
    if hasattr(datamodule, "setup"):
        datamodule.setup(stage="predict")
    if not hasattr(datamodule, "predict_dataloader"):
        raise MissingFirstStepLogits(
            "datamodule has no predict_dataloader(). Refusing to invent logits."
        )
    return unwrap_predict_dataloader(datamodule.predict_dataloader())


def _make_predict_trainer():
    """One-GPU Trainer. ``setup`` only needs ``world_size`` and ``global_rank``."""
    try:
        from lightning.pytorch import Trainer
    except ImportError as exc:
        raise MissingFirstStepLogits(
            "lightning is not importable. GRID SequenceDataModule.setup reads self.trainer. "
            "Refusing to invent logits. " + str(exc)
        ) from exc
    return Trainer(
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        strategy="auto",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def _load_and_collect_logits(*, root: Path, data_dir: str, sid_path: str, ckpt: str, predict_dir: str):
    try:
        import torch
    except ImportError as exc:
        raise MissingFirstStepLogits(
            "torch is not importable. Run scripts/dump_first_step_features.sh on the GPU machine. "
            "Refusing to invent logits. " + str(exc)
        ) from exc
    if not torch.cuda.is_available():
        raise MissingFirstStepLogits(
            "CUDA is not available. The first-step dump runs on GPU. Refusing to invent logits."
        )
    try:
        import hydra
        from hydra.core.global_hydra import GlobalHydra

        import src.utils.custom_hydra_resolvers  # noqa: F401  registers Hydra resolvers
    except ImportError as exc:
        raise MissingFirstStepLogits(
            "GRID dependencies are not importable (hydra or src). "
            "Run this from the pinned checkout via scripts/dump_first_step_features.sh. "
            "Refusing to invent logits. " + str(exc)
        ) from exc
    overrides = hydra_overrides(
        data_dir=data_dir,
        sid_path=sid_path,
        ckpt=ckpt,
        predict_dir=predict_dir,
    )
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = hydra.compose(config_name="inference", overrides=overrides)
    datamodule = hydra.utils.instantiate(cfg.data_loading.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    _load_state_dict(model, ckpt)
    loader = bind_trainer_and_predict_loader(model, datamodule, _make_predict_trainer())
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        return collect_logits_by_key(model, loader, device=device)
