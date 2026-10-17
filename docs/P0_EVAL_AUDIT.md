# P0 eval audit

P0 makes the Beauty fixed-beam rankings auditable for offline ΔNDCG@20. It does not retrain TIGER or MiniLM, and it does not claim a benefit-aware gain. The benefit-aware stop-gate is still not wired.

## What is label leakage

Label leakage is the target **interaction event** appearing in the history used to predict it. A repeated purchase of the same item id at an earlier position is allowed.

Beauty is temporal leave-one-out, the P5 split shipped with GRID:

- Interactions for a user are in time order.
- `testing` stores the full sequence. The last event is the test target.
- `evaluation` is that sequence without the test event. Its last event is the validation target.
- `training` is the evaluation sequence without the validation event.

The same `user_id` in more than one of those folders is expected. Shared user ids are not label leakage. `assert_test_unused` still rejects shared user ids, and that lock stays on the scripted benefit smoke, where each user belongs to one split. Beauty benefit fitting, when P1 does it, must use `user_protocol="temporal_loo"`, which rejects a reused test `request_id` and allows the shared user.

## GRID collate, pinned at `2fe3475b2d369580234093f35d52b1a2f54d0472`

`tiger_train_flat` runs `collate_fn_train` and `NextKTokenMasking(next_k=num_hierarchies)`. `pad_or_trim_sequence` keeps the tail, then the label function masks the last semantic-id group. Those tokens are the last item. The logged test nDCG@5, nDCG@10, Recall@5, and Recall@10 come from that masked input. This audit does not treat those numbers as leaked, and it does not recompute them.

`tiger_inference_flat` runs `collate_fn_inference_for_sequence`. That collate pads or trims and does not mask. The trim keeps the tail, so if the stored testing sequence ends with the target event, the beam search sees that event. The 2026-10-17 beam 20/40/60 files were produced by `scripts/run_after_tiger.sh` on that collate. They are not leak-safe ΔNDCG inputs. This VM does not have those files; the audit confirms the file layout when it is run on the data.

## Files

`export_eval_samples` writes:

| file | meaning |
| --- | --- |
| `eval_samples.jsonl` | one request per line |
| `eval_samples.csv` | the same rows; list fields are JSON |

Each record has at least `request_id`, `user_id`, `split`, `history`, `target`, `target_position`, `checkpoint_hash`, `semantic_id_hash`. `history` is the strict prefix before `target_position`. `source_sequence` is `history + [target]` and is what the cross-split prefix check uses. `repeated_target_item` is true when the target item id already occurs in history.

`request_id` is `{split}:{user_id}:{target_position}` unless the input row sets one.

`audit_eval_leakage` writes `audit_report.json`:

- every row is checked; `n_target_event_in_history` must be 0
- a spot-check list of request ids (the check itself is not a sample)
- user-overlap counts, with overlap marked as not label leakage
- whether train / val / test sequences are leave-one-out prefixes
- the collate finding above
- `tiger_inference_flat_rankings_leak_safe` (false when the prefixes match, because the stored tail is the target and the inference collate keeps it)

`score_beam_ndcg` writes:

| file | meaning |
| --- | --- |
| `per_request.jsonl` | NDCG and recall at 5, 10, and 20 for beams 20, 40, and 60, plus `delta_ndcg@20` |
| `per_request.csv` | the same rows, flattened |
| `summary.json` | means, and whether the delta is auditable |
| `summary.csv` | the summary table |

`delta_ndcg@20` is NDCG@20(beam 60) minus NDCG@20(beam 20). Beam width is the search width. The metric still cuts the ranked list at 20 (and also reports @5 and @10 on that list). Duplicate items in a ranking are dropped, first occurrence kept, so a repeated hit cannot push NDCG above 1.

`auditable_for_delta_ndcg` is true only when `--ranking-source leak_safe_history`. `grid_tiger_inference_flat` still writes the numbers, and marks them as not a leak-safe delta. `benefit_aware_claimed` is false.

## How to run

CPU smoke, no Beauty download and no GRID call:

```bash
python -m budget_sid export_eval_samples --smoke
python -m budget_sid audit_eval_leakage --smoke
python -m budget_sid score_beam_ndcg --smoke
```

`--dry-run` checks and prints the summary without writing files.

On SeetaCloud, from `/root/autodl-tmp/Budget-SID`, against the existing Beauty TFRecords (TensorFlow is in the GRID env; this CI image does not have it):

```bash
DATA_DIR=/root/autodl-tmp/Budget-SID/third_party/GRID/data/amazon_data/beauty \
CKPT=/root/autodl-tmp/Budget-SID/third_party/GRID/logs/train/runs/2026-10-17/00-35-10/checkpoints/checkpoint_epoch=000_step=018800.ckpt \
SID_PATH=/path/to/semantic_ids.pt \
bash scripts/p0_eval_audit.sh
```

`CKPT` defaults to the 2026-10-17 early-stopped checkpoint relative to the repo. `SID_PATH` or `SEMANTIC_ID_HASH` is required. The script prints `SKIP_SHUTDOWN` and does not power the machine off.

JSONL also works if TensorFlow is absent. A directory of `training` / `evaluation` / `testing` (or `train` / `validation` / `test`) is read per split. A row is either `{"user_id", "sequence"}` with the target included as the last event, or `{"user_id", "history", "target", "target_position"}`.

To score rankings, pass a directory of `beam_20.jsonl`, `beam_40.jsonl`, and `beam_60.jsonl` (CSV and pickle of plain rows also work):

```bash
python -m budget_sid score_beam_ndcg \
  --samples outputs/p0_eval/eval_samples.jsonl \
  --rankings-dir /path/to/rankings \
  --output-dir outputs/p0_eval \
  --ranking-source leak_safe_history
```

Each ranking row needs `request_id` or a `user_id` that matches one sample, and `ranked_items`. Semantic-id rankings need `--sid-map` (JSONL of `item_id` and `semantic_id`, or a `.pt` tensor in the GRID env). A `.pt` map of shape `[4, n_items]` or `[n_items, 4]` is the GRID layout.

`--ranking-source grid_tiger_inference_flat` is what `scripts/p0_eval_audit.sh` uses when `RANKINGS_DIR` points at the existing beam exports. Read the numbers as contaminated until the beams are re-run on the exported history.

## P1

P1 should consume these files and still not start from the 2026-10-17 beam exports:

1. Encode `history` from `eval_samples.jsonl` (the target event is already removed) and re-run fixed beams 20, 40, and 60.
2. Score with `--ranking-source leak_safe_history`.
3. Use `per_request.jsonl` field `delta_ndcg@20` as the oracle label on validation rows: fit on user split A, choose the threshold on user split B, leave test events out. Shared user ids are allowed.
4. Only then run the stop-gate. A gain on the old `tiger_inference_flat` rankings is not a benefit-aware result.
