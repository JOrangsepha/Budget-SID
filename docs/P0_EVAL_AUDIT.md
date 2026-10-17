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
- `protocol_exact` is that strict check. `protocol_verified` is also true when every user who has all three splits is either exact or a one-item tail window (see below)
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

## Truncation and `protocol_verified`

On the SeetaCloud Beauty export (`n=67089`, 22363 per split) the event check passed: `n_target_event_in_history=0`. The strict prefix check did not, for about 1019 users. Those stored sequences are still usable.

GRID's inference collate keeps at most `sequence_length=120` tokens, which is 30 items when `num_hierarchies=4`. The P5 files can also already be a tail of a longer interaction list. After that cap, val is not byte-for-byte `test[:-1]`: both sequences are length L, and `val[1:] == test[:-1]` (the window slid by the new target). The same slide applies to train versus val. `protocol_exact` stays false. `protocol_verified` is true when every all-split user is either an exact prefix or that one-item window. `status` is `verified_with_truncation`. A mismatch that is not that window still fails the audit.

The exported `history` is the strict prefix of the stored sequence. It does not contain the target event. A long history is trimmed again at inference time, from the left, and that trim still does not see the target. Do not treat `protocol_exact=false` as label leakage when `n_target_event_in_history=0`.

## Leak-safe beams

`scripts/run_leak_safe_beams.sh` is the GPU step after the export. It does not retrain. It does not read the 2026-10-17 `tiger_inference_flat` pickles. Those files were produced by `collate_fn_inference_for_sequence` on sequences that end with the target event. Scoring them with `--ranking-source leak_safe_history` would mark a contaminated delta as auditable. Do not do that.

What the script runs:

1. `build_leak_safe_predict` reads `eval_samples.jsonl` and writes `outputs/leak_safe_beams/predict/`.
2. `history_only.tfrecord.gz` has `sequence_data` (history item ids only) and `user_id` (a dense `predict_key` in `0 .. n-1`). Raw user ids collide across splits, and GRID's merged prediction tensor indexes by `user_id`, so the key cannot be the raw id. `predict_index.jsonl` maps `predict_key` to `request_id`.
3. `tiger_inference_flat` is pointed at that folder with the existing Beauty checkpoint. The override that exists on the experiment is `data_loading.predict_dataloader_config.dataloader.data_folder`. Workers and timeout stay 0. `ckpt_path='$CKPT'`. No `test_dataloader_config`. No autodl poweroff.
4. `export_leak_safe_rankings` reads `merged_predictions.pkl` (`user_id`, `semantic_ids`), decodes codes with the SID map, and writes `rankings/beam_20.jsonl`, `beam_40.jsonl`, and `beam_60.jsonl`. Each row has `request_id` and `ranked_items`. `predict_key` is `0 .. n-1` with no gaps, which is what GRID's merged tensor index expects. The scorer reads the pickle list, joined through `predict_index.jsonl`.
5. `score_beam_ndcg --ranking-source leak_safe_history` writes `per_request.jsonl` and `summary.json` under `outputs/leak_safe_beams/`. `auditable_for_delta_ndcg` is true only for that source.

CPU smoke, no GRID call:

```bash
python -m budget_sid build_leak_safe_predict --smoke
python -m budget_sid export_leak_safe_rankings --smoke
```

`DRY_RUN=1` builds the TFRecords and prints the GPU command.

SeetaCloud one-liner, from `/root/autodl-tmp/Budget-SID`, after `scripts/p0_eval_audit.sh` has written the samples. `CKPT` defaults to the 2026-10-17 early-stopped checkpoint. `SID_PATH` is the RQ-KMeans semantic-id tensor:

```bash
SID_PATH=/path/to/semantic_ids.pt bash scripts/run_leak_safe_beams.sh
```

`SKIP_BUILD=1` reuses `outputs/leak_safe_beams/predict/`. The script prints `SKIP_SHUTDOWN` and leaves the instance on.

## P1

P1 should consume these files and still not start from the 2026-10-17 beam exports:

1. Run `scripts/run_leak_safe_beams.sh` so beams 20, 40, and 60 are encoded from `history` (the target event is already removed). Do not start from the 2026-10-17 `tiger_inference_flat` pickles.
2. Score with `--ranking-source leak_safe_history` (the script does this).
3. Use `per_request.jsonl` field `delta_ndcg@20` as the oracle label on validation rows: fit on user split A, choose the threshold on user split B, leave test events out. Shared user ids are allowed.
4. Only then run the stop-gate. A gain on the old `tiger_inference_flat` rankings is not a benefit-aware result.

The offline reading of the SeetaCloud summary is `docs/P1_BENEFIT_ANALYSIS.md`. That summary does not clear P2, and it does not claim a benefit-aware gain. On the machine that has the jsonl, run `python -m budget_sid analyze_benefit --per-request outputs/leak_safe_beams/per_request.jsonl --summary outputs/leak_safe_beams/summary.json`. The validation pass that set `p2_worth_doing=true` is replayed offline by `docs/P2_OFFLINE_REPLAY.md`.
