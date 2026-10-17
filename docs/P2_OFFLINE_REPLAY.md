# P2 offline replay

P2 measures whether a ridge model can spend a **shared large-beam quota** on the requests that actually gain from beam 60. It is an offline allocation check on leak-safe validation labels.

It is not a latency result. It does not run GridRecommender, it does not time decoding, and it does not claim that benefit-aware routing will raise Beauty nDCG online. Those questions are P3.

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
3. Split validation **users** into disjoint A and B with `split_val_users` (seeded shuffle, independent of row order). Default seed is config `train.seed` (42). `--seeds 42,43,44` repeats the replay.
4. Fit `fit_benefit_predictor` on A: ridge on standardized features, intercept unpenalized. The target is ΔnDCG@20 = nDCG@20(beam 60) − nDCG@20(beam 20).
5. On B, give every non-oracle policy the **same** number of beam-60 slots:

```text
n_quota = min(n_positive on B, n_large_for_budget(n_B))
```

`n_large_for_budget` is the beam-40 ceiling (at most half the requests on beam 60). The quota does not fill unused ceiling with non-positive requests. P1 rejects that force-fill. The quota is also not the Bernoulli draw in `RandomBudgetPolicy` (`p = 0.5`), which only matches mean beam 40 in expectation.

The quota **count** is the label marginal on B (how many positive requests the corrected oracle would upgrade). Random, entropy, and benefit-aware do not read which request has which delta when they pick. The oracle ranks B by true delta and is the upper bound, not a deployable policy.

Tie-break is `request_id`, so equal scores still spend exactly `n_quota` slots. `select_threshold` is not used here: it targets the ceiling and can drop ties below that count, so policies would not share one quota.

## Metrics

On B, for random, entropy (when `first_step_entropy` is joined), benefit-aware, and the oracle:

- mean nDCG@20 of the chosen beam (beam 60 if selected, otherwise beam 20)
- mean beam, which matches across those policies because `n_quota` matches
- positive-Δ mass captured, as a fraction of the oracle's captured mass under that quota

Uniform beam 20 / 40 / 60 and the rejected force-fill are context. They are not quota-matched policies. Split-B MSE of the ridge model is a fit diagnostic. It is not the allocation result.

`beats_uniform_beam40` is not required for the quota to be well defined. A policy can capture positive mass and still lose to uniform beam 40 when beam 40 is ahead of beam 20 on the requests left at beam 20. The mean nDCG is the number that shows that.

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

Priority for history length is the list above. A disagreement between samples and `target_position` is counted. Samples win. Incomplete columns are dropped for every validation row. They are not imputed.

### Follow-up blocker

Full P2 needs a first-step dump joined on `request_id`:

```text
history_length, first_step_entropy, candidate_prob_gap, top1_prob
```

or `history_length` plus the first-step logit vector. Produce those with `extract_features` after the shared first decoding step. Do not fill the columns with placeholders. Until that file exists, the SeetaCloud path still runs:

- entropy is skipped
- ridge uses whatever non-label columns were actually joined (often only `history_length` via `target_position`)
- `features.full_p2_ready` is false and `features.full_p2_blockers` lists the missing names
- `invented_logits` is false

That run is an honest partial replay. It is not the four-feature model.

## How to run

CPU smoke, no Beauty file and no GRID call. The synthetic tail is concentrated: uniform beam 60 loses to beam 40, high entropy is mostly decoys, and the ridge features separate the positive cluster.

```bash
python -m budget_sid replay_budgets --smoke
python -m budget_sid replay_budgets --smoke --seeds 42,43,44
```

On SeetaCloud, from the repo that already has the leak-safe beams. This does not start a GPU job and does not power the machine off.

```bash
python -m budget_sid replay_budgets \
  --per-request outputs/leak_safe_beams/per_request.jsonl \
  --summary outputs/leak_safe_beams/summary.json \
  --samples outputs/p0_eval/eval_samples.jsonl \
  --user-protocol temporal_loo \
  --seeds 42,43,44 \
  --output-dir outputs/p2_replay
```

Add `--features /path/to/first_step_features.jsonl` when the logit dump exists. Each row needs `request_id` and either the four feature fields or `history_length` plus `logits`. `--samples` is optional if `target_position` is present. `--dry-run` prints the summary and does not write `--output-dir`.

`outputs/` is gitignored. The command writes `replay_summary.json` and `assignments_seed<seed>.jsonl`.

## What this is not

No benefit-aware online result. No latency, throughput, or memory claim. No GridRecommender. No second decode. The 2026-10-17 `tiger_inference_flat` rankings are not these labels. A ridge model cannot beat the oracle that already sees Δ on B; the oracle row is a bound.
