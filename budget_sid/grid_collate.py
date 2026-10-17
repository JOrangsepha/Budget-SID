"""What the pinned GRID collates show the encoder, for one Beauty sequence.

``tiger_train_flat`` masks the last semantic-id group
(``NextKTokenMasking`` with ``next_k = num_hierarchies``) after
``pad_or_trim_sequence`` keeps the tail. The logged test nDCG@5 / nDCG@10
and Recall@5 / Recall@10 come from that path.

``tiger_inference_flat`` calls ``collate_fn_inference_for_sequence``, which
only pads or trims. The trim keeps the tail, so the last item's semantic-id
tokens stay in the beam-search input whenever the stored sequence ends with
the target event.

This module does not read the SeetaCloud rankings. It records the collate
behavior at GRID commit ``2fe3475b2d369580234093f35d52b1a2f54d0472``.
"""

from __future__ import annotations

from budget_sid.constants import GRID_COMMIT

NUM_HIERARCHIES = 4
SEQUENCE_LENGTH = 120
TRAIN_LABEL_FUNCTION = "NextKTokenMasking"
TRAIN_COLLATE = "collate_fn_train"
INFERENCE_COLLATE = "collate_fn_inference_for_sequence"
INFERENCE_EXPERIMENT = "tiger_inference_flat"
TRAIN_EXPERIMENT = "tiger_train_flat"

COLLATE_NOTE = (
    "tiger_train_flat masks the last semantic-id group via NextKTokenMasking, "
    "so the logged nDCG@5 and nDCG@10 are not this leak. "
    "tiger_inference_flat uses collate_fn_inference_for_sequence, which keeps the tail. "
    "When the stored sequence ends with the target event, that event's tokens are in the beam-search input."
)


def inference_input_includes_target_event(
    n_items: int,
    *,
    num_hierarchies: int = NUM_HIERARCHIES,
    sequence_length: int = SEQUENCE_LENGTH,
) -> bool:
    """True when any token of the final item survives the inference trim.

    ``pad_or_trim_sequence`` drops a prefix when the token sequence is longer
    than ``sequence_length`` and always retains the right edge. The final item
    occupies that edge, so a positive window includes it.
    """
    if n_items < 1 or num_hierarchies < 1 or sequence_length < 1:
        return False
    n_tokens = int(n_items) * int(num_hierarchies)
    window_start = 0 if n_tokens <= sequence_length else n_tokens - int(sequence_length)
    target_start = (int(n_items) - 1) * int(num_hierarchies)
    return target_start < n_tokens and n_tokens > window_start


def describe_collate_inputs(
    n_items: int,
    *,
    num_hierarchies: int = NUM_HIERARCHIES,
    sequence_length: int = SEQUENCE_LENGTH,
) -> dict:
    """Compare the train/val label mask with the inference collate.

    ``n_items`` is the length of the stored item sequence, which includes the
    target as its last event on the GRID leave-one-out files.
    """
    if n_items < 1:
        raise ValueError("sequence needs the target event")
    includes = inference_input_includes_target_event(
        n_items,
        num_hierarchies=num_hierarchies,
        sequence_length=sequence_length,
    )
    return {
        "grid_commit": GRID_COMMIT,
        "n_items": int(n_items),
        "target_position": int(n_items) - 1,
        "num_hierarchies": int(num_hierarchies),
        "sequence_length_tokens": int(sequence_length),
        "train_experiment": TRAIN_EXPERIMENT,
        "train_collate": TRAIN_COLLATE,
        "train_label_function": TRAIN_LABEL_FUNCTION,
        "train_input_includes_target_event": False,
        "reported_at5_at10_uses_masked_target": True,
        "inference_experiment": INFERENCE_EXPERIMENT,
        "inference_collate": INFERENCE_COLLATE,
        "inference_input_includes_target_event": includes,
        "pad_or_trim_keeps_tail": True,
        "note": COLLATE_NOTE,
    }
