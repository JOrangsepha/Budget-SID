# P2 offline replay

P2.1 measures whether a model can spend a **pre-declared upgrade fraction** better than random and entropy. A fraction `f` upgrades that share of validation split B from beam 20 to beam 60. The declared average beam is `20 + 40 * f`. It is an offline allocation check on leak-safe validation labels.

It is not a latency result. It does not run GridRecommender, it does not time decoding, and it does not claim that benefit-aware routing will raise Beauty nDCG online. P3 stays gated.

## Recorded validation pass

SeetaCloud validation rows of `outputs/leak_safe_beams/per_request.jsonl` were reported as `p2_worth_doing=true`. That file is gitignored. This commit does not contain it and does not recompute the table.

| | validation |
| --- | ---: |
| n | 22363 |
| positive Δ | 0.926% (207) |
| zero Δ | 97.50% |
| negative Δ | 1.57% |
| oracle nDCG@20 | 0.05911 |
| uniform beam 40 nDCG@20 | 0.05750 |
| oracle mean beam | ~20.37 |

Uniform beam 60 does not beat uniform beam 40. The oracle above is the corrected P1 assignment: upgrade a request only when ΔnDCG@20 is positive, highest deltas first, and stop at the beam-40 ceiling. 207 positives is under that ceiling (`floor(0.5 * 22363) = 11181`), so the oracle spends 207 large beams:

```text
(207 * 60 + (22363 - 207) * 20) / 22363 = 20.370
```

These are validation means. They are not the pooled n = 67089 summary in `artifacts/LEAK_SAFE_BEAMS_SUMMARY.json` (beam 40 nDCG@20 0.06425 on all splits). The summary means still do not, by themselves, clear P2. `docs/P1_BENEFIT_ANALYSIS.md` explains why.

## Protocol

1. Read `per_request.jsonl` from `score_beam_ndcg` (`ranking_source=leak_safe_history`). A summary with `auditable_for_delta_ndcg: false` is refused. A sibling `summary.json` is read when `--summary` is omitted.
2. Keep test events out. `user_protocol=auto` uses `temporal_loo` when a validation user also appears in test (Beauty leave-one-out), and `disjoint_users` otherwise. `temporal_loo` rejects a reused test `request_id` and allows the shared user. Training-split rows are ignored.
3. Split validation **users** into disjoint A and B with `split_val_users` (seeded shuffle, independent of row order). Default seed is config `train.seed` (42). `--seeds 42,43,44` repeats the replay. Those seeds reshuffle the A/B split and the random-policy scores. They are not independent recommender retrainings. The Beauty checkpoint stays the 2026-10-17 early stop.
4. Fit `fit_benefit_predictor` on A: ridge on standardized features, intercept unpenalized. The target is ΔnDCG@20 = nDCG@20(beam 60) − nDCG@20(beam 20). When scikit-learn is installed (`pip install -e ".[hgb]"`), the same columns also fit `HistGradientBoostingRegressor` with `max_depth=3` and `max_iter=50`. That tree is another benefit policy under the same quota. Boundary columns are in that shared list only when the joined file has them. A missing scikit-learn install leaves ridge in place and marks `benefit_aware_hgb` unavailable.
5. On B, sweep pre-declared upgrade fractions. The default list is 0%, 1%, 5%, 10%, 25%, and 50%:

```text
n_quota = floor(fraction * n_B)
declared average beam = 20 + 40 * fraction
realized average beam = 20 + 40 * (n_quota / n_B)
```

`floor` keeps the realized average beam at or under the declaration. At 50%, `n_quota` matches `n_large_for_budget` (the beam-40 ceiling). Random, entropy, ridge, and the tree (when it runs) each upgrade that same count. One score ranking is drawn per seed; a larger fraction is a prefix of that ranking, so the policies share one quota at every point. The count is a fraction of requests. `quota_derived_from_positive_count` is false. It is not `min(n_positive on B, ceiling)`.

Tie-break is `request_id`, so equal scores still spend exactly `n_quota` slots. `select_threshold` is not used here: it targets the ceiling and can drop ties below that count, so policies would not share one quota.

The oracle upper bound is reported beside each point, with the compute it actually used. It upgrades positive-Δ requests only, highest delta first, and stops at `min(n_positive, n_quota)`. Leftover slots stay at beam 20. It does not upgrade a negative-Δ or zero-Δ request to fill the fraction. When positives are scarce, the oracle mean beam stays near 20 while the policies sit on the declared budget. `same_compute_as_policies` records that difference.

`--quota-mode positive_count` restores the older narrow diagnostic, `n_quota = min(n_positive on B, ceiling)`, as a single point. That flag does not take `--fractions`. It is not the default.

## Metrics

Primary metrics, on B, for every policy that runs:

- `mean_ndcg@20` of the chosen beam (beam 60 if selected, otherwise beam 20)
- `delta_vs_all_beam20`, that mean minus the all-beam-20 mean

They live under `primary` in the summary JSON. `primary_metrics` lists those two names.

Explanatory metrics live under `explanatory` and are not the allocation result:

- `n_positive_captured`, how many selected requests have positive Δ
- `negative_loss`, the sum of `-Δ` over selected requests whose Δ is negative

`explanatory_metrics_are_not_primary` is true. Split-B MSE is a fit diagnostic in the same way. Uniform beam 20 / 40 / 60 and the rejected force-fill are context on `split_b`. They are not points on the quota curve.

The curve summary repeats the primary block and the explanatory block as separate objects, and it repeats the oracle's own `n_large` and `mean_beam`.

## Features

`score_beam_ndcg` writes this schema and nothing about the first decoding step:

| field | role |
| --- | --- |
| `request_id`, `user_id`, `split` | identity and the train / val / test cut |
| `target_position` | P0 sets this equal to `len(history)` |
| `beams["20"\|"40"\|"60"]["ndcg@20"]` | label, read only after a policy has chosen |
| `delta_ndcg@20` | label, beam 60 minus beam 20 |

There is no `history_length`, no entropy, no probability gap, and no logits. Rank, recall, the target id, and `repeated_target_item` are labels or label-derived. They are dropped if a feature file contains them. They are never model inputs.

The ridge features, from `budget_sid.features`, are:

| feature | where it can come from |
| --- | --- |
| `history_length` | `--features`, else `--samples` history length, else a `history` field on the row, else `--predict-index` `n_history`, else `target_position` |
| `first_step_entropy` | `--features`, or `extract_features(history_length, logits)` when the feature file supplies logits |
| `candidate_prob_gap` | same |
| `top1_prob` | same |
| `top20_cum_prob` | `--features` from `extract_boundary_features`, or logits with at least 60 candidates |
| `prob_gap_p20_p21` | same. p(rank 20) − p(rank 21) after sorting first-step probabilities descending |
| `prob_sum_ranks_21_60` | same. Mass on ranks 21–60, the codes beam 60 can still reach after beam 20 |

Priority for history length is the list above. A disagreement between samples and `target_position` is counted. Samples win. Incomplete columns are dropped for every validation row. They are not imputed. A logit vector shorter than 60 candidates does not grow the boundary columns. It is not padded with zeros.

Ranks are 1-based. Rank 1 is the highest first-step probability. Ties keep the earlier logit index (stable mergesort). `top20_cum_prob` sums ranks 1–20. `prob_gap_p20_p21` is rank 20 minus rank 21. `prob_sum_ranks_21_60` sums ranks 21–60. Invalid prefix codes that the pinned generate path masks to `-inf` contribute zero probability. They are not rewritten as zeros before the softmax.

### Recorded narrow-quota result

Both SeetaCloud replays below used the old narrow quota, `n_quota = min(n_positive on B, ceiling)`. Neither beat random and entropy. This checkout does not recompute them. P3 stays gated. P2.1 replaces that quota with the fixed-fraction curve. The four-feature file is still a valid replay. Boundary columns are an extra arm on the same curve. They are used when `boundary_features.jsonl` is joined. This repo does not invent a table that says they beat random or entropy.

| replay | what was joined | result under the narrow quota |
| --- | --- | --- |
| history length only | `history_length` from `target_position` | Ridge captured about 0% of the positive-Δ mass and tied or lost to random (nDCG@20 about 0.0566 versus about 0.0567). Entropy was skipped. |
| four features | `history_length`, `first_step_entropy`, `candidate_prob_gap`, `top1_prob` | Benefit-aware did not beat random or entropy. This repo has no separate nDCG table for that run, and this change does not invent one. |

Until `first_step_features.jsonl` is joined, a replay still behaves like the history-length row:

- entropy is skipped
- ridge uses whatever non-label columns were actually joined
- `features.full_p2_ready` is false and `features.full_p2_blockers` lists the missing names
- `invented_logits` is false

That run is an honest partial replay. It is not the four-feature model. Do not fill the columns with placeholders.

### First-step dump

`dump_first_step_features` writes `first_step_features.jsonl`. Each row has `request_id`, `history_length`, `first_step_entropy`, `candidate_prob_gap`, and `top1_prob`. The four feature columns are `extract_features` on the hierarchy-0 candidate logits. The raw vector is not stored, and it is not reconstructed later.

Alignment with `outputs/leak_safe_beams/per_request.jsonl`:

1. `build_leak_safe_predict` writes `predict_index.jsonl` in eval-sample order. `predict_key` is `0 .. n-1`. `sequence_data` is the exported history. The target event is not appended. `n_history` is that length.
2. `scripts/run_leak_safe_beams.sh` points `tiger_inference_flat` at that folder. GRID's `user_id` field is the `predict_key`. `score_beam_ndcg` writes `per_request.jsonl` in the same eval-sample order.
3. The dump reads the same folder. It joins logits on `user_id_list` (the `predict_key`), then emits rows in index order. `history_length` is `n_history`. When `--per-request` is passed, the request-id sequence must match exactly. A dataloader that yields keys out of order still writes index order. Batch position is not a join key.

Pinned GRID `SemanticIDEncoderDecoder.generate` (commit `2fe3475`) computes hierarchy-0 `candidate_logits` with `encoder_forward_pass`, `decoder_forward_pass(future_ids=None)`, and `decoder.decoder_mlp[0]`, then drops them inside `_beam_search_one_step`. `predict_step` returns semantic ids only, so the beam pickles have no logits. The dump calls those same three methods and keeps the matrix. It does not run hierarchies 1..3, so `top_k` does not change the file: one dump covers beams 20, 40, and 60. The pin stays `2fe3475`. The checkout is not edited.

`SequenceDataModule.setup` reads `self.trainer`. The dump attaches a Lightning Trainer to the module and the datamodule before that call, and leaves `Trainer.predict` unused because `predict_step` drops the logits. `predict_dataloader` returns a 1-tuple; the dump unwraps it before iterating batches. Iterating the tuple yields the loader, so the batch loop never sees `user_id_list`.

If the logit tensor, the MLP, the checkpoint `state_dict`, or `user_id_list` is missing, the process exits non-zero and does not write a zero-filled file. A partial `load_state_dict` is refused, because a randomly initialized `decoder_mlp` would look like real logits.

P3 stays gated until benefit-aware routing beats **both** random and entropy on primary `mean_ndcg@20` and `delta_vs_all_beam20` under the pre-declared fractions, and that ordering holds across seeds 42, 43, and 44. The narrow-quota replays above did not clear it. The four-feature fraction curve is the same gate. This dump does not start P3, does not time decoding, and does not claim an online win.

### Hierarchy-0 logits cache and boundary features

`first_step_features.jsonl` stores the four scalars and drops the vector, so a new cut of that vector needs another GPU pass. `dump_first_step_logits` stores the hierarchy-0 logit vector instead. Float16 is used when every finite score stays finite after the cast. Overflow to inf falls back to float32, or the process exits. NaN and a row with no finite score are errors. `load_state_dict` is strict, the same loader as the four-scalar dump. The GRID pin stays `2fe3475`.

Default scope is validation (about 22k Beauty requests). Two passes are allowed:

- **val-only.** `predict_rows.jsonl` is present. The dump writes a sibling folder (default `outputs/leak_safe_beams/predict_val`) whose TFRecord contains only validation histories. `user_id` stays the original `predict_key`, so the forward matches the full-folder forward for those rows. Train and test are not scored.
- **full-then-filter.** `predict_rows.jsonl` is missing, or `--split all` was not requested and the sequences file is absent. The existing history-only folder is scored. Every index key must come back. Non-validation rows are dropped only after that check. A validation key with no vector is still an error.

`--split all` caches every predict-index row. The cache does not renumber `predict_key`.

Join:

| key | where it lines up |
| --- | --- |
| `request_id` | `replay_budgets --features`, `per_request.jsonl`, `boundary_features.jsonl` |
| `predict_key` | GRID `user_id` on the history-only TFRecord, and `predict_index.jsonl` |

Batch position is not a join key. The npz columns are `logits`, `predict_key`, `request_id`, `history_length`, and `split`, plus a `logits_dtype` label. `first_step_logits_val_manifest.json` (or the `_manifest.json` beside whatever path `--output` used) records the same join, the pass (`val_only`, `full_then_filter`, or `all`), and `invented_logits: false`.

`extract_boundary_features` reads that npz on CPU and writes `boundary_features.jsonl`. Each row has the four scalars plus `top20_cum_prob`, `prob_gap_p20_p21`, and `prob_sum_ranks_21_60`. The jsonl does not embed the vector. Replay joins it with `--features`. `features.boundary_features_ready` is true when all three boundary columns are present on every validation row. `boundary_features_dumped` on the replay summary is that same bit. Missing boundary columns are left out. They are not blockers for the four-scalar replay, and they are not filled in.

The target event is not a routing feature. An offline attribution check can ask, after the fact, whether the target item's hierarchy-0 semantic-id code was among the top-20 first-step codes. That bit reads the target, so it is a label. This change does not compute it and does not write it into the feature file.

## How to run

CPU smoke, no Beauty file and no GRID call. The synthetic tail is concentrated: uniform beam 60 loses to beam 40, high entropy is mostly decoys, and the ridge features separate the positive cluster.

```bash
python -m budget_sid replay_budgets --smoke
python -m budget_sid replay_budgets --smoke --seeds 42,43,44
```

The default sweep is `0,0.01,0.05,0.10,0.25,0.50`. One run writes `replay_summary.json`, `curve_summary.json`, and `assignments_seed<seed>.jsonl`. Declared average beam at those fractions is 20, 20.4, 22, 24, 30, and 40.

On the machine that already has `per_request.jsonl` and `first_step_features.jsonl`, the curve is a CPU replay. It does not start a GPU job and does not power the machine off. Seeds 42, 43, and 44 are A/B split and replay seeds.

```bash
python -m budget_sid replay_budgets \
  --per-request outputs/leak_safe_beams/per_request.jsonl \
  --summary outputs/leak_safe_beams/summary.json \
  --features outputs/leak_safe_beams/first_step_features.jsonl \
  --user-protocol temporal_loo \
  --seeds 42,43,44 \
  --fractions 0,0.01,0.05,0.10,0.25,0.50 \
  --output-dir outputs/p2_replay
```

`--fractions` can be omitted; that list is the default. `0%,1%,5%,10%,25%,50%` and `0,1,5,10,25,50` parse as the same percents. The legacy narrow quota is explicit:

```bash
python -m budget_sid replay_budgets \
  --per-request outputs/leak_safe_beams/per_request.jsonl \
  --features outputs/leak_safe_beams/first_step_features.jsonl \
  --user-protocol temporal_loo \
  --seeds 42,43,44 \
  --quota-mode positive_count \
  --output-dir outputs/p2_replay_legacy
```

Produce the feature file on the GPU machine, from the same history-only folder the beams used. `DRY_RUN=1` checks that folder and the checkpoint file and does not load the model. A missing checkpoint exits non-zero. The script prints `SKIP_SHUTDOWN` and does not power the machine off.

```bash
SID_PATH=/path/to/semantic_ids.pt bash scripts/dump_first_step_features.sh
```

That writes `outputs/leak_safe_beams/first_step_features.jsonl` next to `per_request.jsonl`. Then re-run the replay. This is still offline allocation. It is not P3.

`--samples` is optional if `target_position` is present. The feature file's `history_length` wins over it. `--dry-run` prints the summary and does not write `--output-dir`.

The four-scalar fraction sweep needs the existing `per_request.jsonl` plus `first_step_features.jsonl` (or `history_length` from samples / `target_position`). It does not need a new GPU pass.

The boundary arm needs one GPU pass, then CPU extract and the same sweep. From the repo on SeetaCloud (`/root/autodl-tmp/Budget-SID` when that checkout exists). `DRY_RUN=1` checks the predict folder and the checkpoint and does not load the model. `SKIP_BUILD=1` reuses the existing predict folder. `EXTRACT_ONLY=1` rebuilds the jsonl from the npz with no second GPU pass. The script prints `SKIP_SHUTDOWN` and does not power the machine off.

```bash
SID_PATH=/path/to/semantic_ids.pt bash scripts/dump_first_step_logits.sh
```

That writes `outputs/leak_safe_beams/first_step_logits_val.npz` and `outputs/leak_safe_beams/boundary_features.jsonl`. Equivalent steps, if the cache already exists:

```bash
python -m budget_sid extract_boundary_features \
  --logits outputs/leak_safe_beams/first_step_logits_val.npz \
  --output outputs/leak_safe_beams/boundary_features.jsonl \
  --split val

python -m budget_sid replay_budgets \
  --per-request outputs/leak_safe_beams/per_request.jsonl \
  --summary outputs/leak_safe_beams/summary.json \
  --features outputs/leak_safe_beams/boundary_features.jsonl \
  --user-protocol temporal_loo \
  --seeds 42,43,44 \
  --fractions 0,0.01,0.05,0.10,0.25,0.50 \
  --output-dir outputs/p2_replay_boundary
```

`--fractions` can be omitted; that list is the default. P3 stays gated until benefit beats both random and entropy on `mean_ndcg@20` and `delta_vs_all_beam20` at those fractions, stably across seeds 42, 43, and 44. This command does not decide that. It does not retrain the recommender.

`outputs/` is gitignored. The replay command writes `replay_summary.json`, `curve_summary.json`, and `assignments_seed<seed>.jsonl`. The dump also writes `first_step_features_manifest.json` beside the jsonl. CPU smoke, with the scripted recommender rather than GRID logits:

```bash
python -m budget_sid dump_first_step_features --smoke
python -m budget_sid dump_first_step_logits --smoke
python -m budget_sid extract_boundary_features --smoke
```

The logits smoke is a scripted CPU vector of width 64. It is not a Beauty GRID dump.

## What this is not

No benefit-aware online result. No latency, throughput, or memory claim. No GridRecommender. No second decode. No P3. No invented logits, and no target-derived routing feature. The 2026-10-17 `tiger_inference_flat` rankings are not these labels. A ridge model cannot beat the positive-Δ oracle that already sees Δ on B; that oracle row is a bound, and its mean beam is the compute it used. The logits cache is produced on the GPU machine. This checkout does not contain it and does not claim the boundary arm beat random or entropy.
