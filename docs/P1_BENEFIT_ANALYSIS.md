# P1 benefit analysis

## Update

The aggregates in this document still do not clear P2. A later SeetaCloud pass on the validation rows of `per_request.jsonl` did: `p2_worth_doing=true` (n=22363, positive Δ 0.926%, oracle nDCG@20 0.05911 > uniform beam 40 0.05750, mean beam about 20.37; uniform beam 60 does not beat beam 40). Those rows are not in git. The offline replay of that decision is `docs/P2_OFFLINE_REPLAY.md`. It is allocation quality, not a latency or online benefit-aware claim.

0.05750 is the full validation set (n=22363), not validation split B. Do not set it next to the A/B-averaged routing table. That table, including fixed beam 20/40/60 on the same split-B requests (seeds 42/43/44), is `docs/FINAL_RESULTS.md`. Archived 2026-10-20: Ridge and HGB did not stably beat random or entropy, online routing stopped, and P3 was not started. The oracle row reads true Δ. It is a non-deployable upper bound (mean beam about 20.4 on split B), not a Budget-SID algorithm gain.

## Decision

**The summary means do not clear P2.** The leak-safe Beauty summary does not show that a concentrated positive tail is large enough to continue.

Uniform beam 60 loses to uniform beam 40. Mean ΔnDCG@20 (beam 60 − beam 20) is 0.000618, which is smaller than the nDCG@20 gap from beam 20 to beam 40 (0.06425 − 0.06356 = 0.00069). A selective oracle beats beam 40 only when the positive deltas it keeps average more than that gap. `summary.json` does not count how many requests are positive, zero, or negative, so that mass is unknown.

`p2_worth_doing` from the summary alone is null. Fit the offline Ridge model only after `analyze_benefit` on the validation rows of `per_request.jsonl` returns true. This document does not claim a benefit-aware online gain. GridRecommender is not started.

A near-zero mean is not, by itself, a reason to kill the question. It is also not a reason to start the Ridge fit. The per-request file is the measurement that decides.

## SeetaCloud aggregates

Transcribed from `outputs/leak_safe_beams/summary.json` (2026-10-17, leak-safe history, `auditable_for_delta_ndcg` true). `per_request.jsonl` is not in this checkout. `outputs/` is gitignored. This commit did not re-run the beams and does not add metrics beyond these aggregates. The copy in git is `artifacts/LEAK_SAFE_BEAMS_SUMMARY.json`.

| | beam 20 | beam 40 | beam 60 |
| --- | ---: | ---: | ---: |
| nDCG@20 | 0.06356 | 0.06425 | 0.06418 |
| Recall@20 | ≈0.1388 | ≈0.1416 | ≈0.1414 |

- n = 67089
- mean ΔnDCG@20 (60 − 20) = 0.000618
- min Δ = -0.369, max Δ = 1.0
- `benefit_aware_claimed` is false

Beam 60 is below beam 40 on both nDCG@20 (0.06418 < 0.06425) and recall@20 (≈0.1414 < ≈0.1416).

0.06418 − 0.06356 = 0.00062, which is 0.000002 away from the reported mean 0.000618. That is rounding of the published beam means, not a second measurement.

One request at the observed max (+1.0) moves the mean by 1/67089 ≈ 0.0000149. One request at the observed min (-0.369) moves it by about -0.0000055. The range shows the delta is not uniform. It does not show that the positive tail clears the 0.00069 gap.

## Why the old gate was the wrong test

Selective allocation can help when the mean Δ over all requests is near zero or negative. The requests that lose on a wide beam stay on beam 20. Two bugs hid that:

1. `has_concentrated_gain` used to fail whenever the mean Δ was ≤ 0, even if a minority of requests held the positive gain.
2. `oracle_allocation` used to spend the entire beam-40 budget, including requests with negative Δ. The cap is an average beam of at most 40 (at most half the requests on beam 60). It is a ceiling, not a quota.

The corrected oracle upgrades a request only when its ΔnDCG@20 is positive, highest deltas first, and stops at the budget. `evaluate_budgets` loads and scores the test split only after the gate returns exit code 0. Exit code 2 does not call the test loader.

## Synthetic check, not Beauty

Ten validation requests. Two gain from beam 60. Eight are hurt by it. Mean Δ = −0.6.

| request | nDCG@20 beam 20 | beam 40 | beam 60 | Δ (60−20) |
| --- | ---: | ---: | ---: | ---: |
| 2 requests | 0 | 0 | 1 | +1 |
| 8 requests | 1 | 1 | 0 | −1 |

| policy | nDCG@20 | mean beam |
| --- | ---: | ---: |
| corrected oracle (2 large beams) | 1.0 | 28 |
| rejected force-fill (5 large beams) | 0.7 | 40 |
| all beam 20 | 0.8 | 20 |
| all beam 40 | 0.8 | 40 |
| all beam 60 | 0.2 | 60 |

The corrected oracle beats force-fill (1.0 > 0.7) and beats uniform beam 40 (1.0 > 0.8). Force-fill loses to beam 40 (0.7 < 0.8). The old oracle would have reported a loss on a case where selective allocation wins. These numbers are the unit-test counterexample. They are not Beauty.

`--smoke` prints this case. `population` is `synthetic_counterexample`.

## What would make P2 worth doing

P2 is an offline Ridge model of ΔNDCG@20. It cannot beat a budgeted oracle that already sees the label. P2 is worth doing only when that oracle, on validation rows, beats uniform beam 40 and the positive tail is concentrated.

```bash
python -m budget_sid analyze_benefit \
  --per-request outputs/leak_safe_beams/per_request.jsonl \
  --summary outputs/leak_safe_beams/summary.json
```

The file schema is the `score_beam_ndcg` schema: `delta_ndcg@20`, and `beams["20"|"40"|"60"]["ndcg@20"]`. `split` is `train`, `val`, or `test`.

`p2_worth_doing` is true only when all of these hold:

- `--summary` has `auditable_for_delta_ndcg: true`. A `grid_tiger_inference_flat` summary cannot clear P2.
- The decision population is the validation split when `split` is present. Test rows cannot clear P2. The all-rows table is descriptive and is what should match `summary.json` (n = 67089).
- Positive Δ is uneven. Either a minority of requests have Δ > 0, or the top 20% of requests hold at least half of the positive mass. The mean over all requests is not part of this check.
- The corrected oracle's nDCG@20 is strictly above uniform beam 40, and its average beam is at most 40.

If the oracle loses, stop. A Ridge model will not recover a gain the label does not contain under the budget. If the summary was not passed and the oracle would have won, `p2_worth_doing` is null: the labels are not confirmed leak-safe. Null means do not start P2.

Summary means alone never clear P2. `p2_identified_by_summary_means` is false even when someone later finds a large tail, because the tail is not in the means.

## What this is not

No benefit-aware online result. No claim that routing will raise Beauty nDCG@20. No GridRecommender. No Ridge fit in this change. The 2026-10-17 `tiger_inference_flat` rankings are not these labels.
