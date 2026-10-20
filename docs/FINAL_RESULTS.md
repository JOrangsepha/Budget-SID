# Budget-SID Beauty 归档最终结果（同验证集 B）

生成时间：2026-10-20（Asia/Shanghai）。仅整理已有逐请求缓存，**未再调用 GPU**。机器可读的同一张表在 `docs/FINAL_RESULTS_TABLE.json`。

**人口**：与 P2.1 策略相同的 validation split B（每 seed ≈11182 条；seeds 42/43/44）。
**不要**把全体 leak_safe（n=67089，beam20≈0.06356）或 `validation_premise.all_40≈0.05750` 直接和本表比。那两个数不是这批 A/B 划分后的请求。

Checkpoint：`checkpoint_epoch=000_step=018800.ckpt`（sha256 `e28e34166123e2b6c85366d761dfc291056fc7aef26bae124ba3dc1bd6a56d03`）。不重训推荐模型。

## 主表：固定 beam + P2.1 四特征策略

特征：`history_length`, `first_step_entropy`, `candidate_prob_gap`, `top1_prob`。Ridge=`benefit_aware`，HGB=`benefit_aware_hgb`。

固定 beam 20/40/60 与下面的策略用同一批 validation-B 请求。固定 beam 没有按策略去记正收益捕获和负收益损失，这两列标 **未测**。净收益是相对这批请求上的全体 beam 20。

| 方法 | 升级比例 | 实际平均 beam | NDCG@20（均值±跨seed std） | 正收益捕获 n_pos | 负收益损失 | 净收益 Δ vs all-beam20 |
|---|---:|---:|---:|---:|---:|---:|
| fixed_beam20 | — | 20.0 | 0.05664±0.00126 | 未测 | 未测 | +0.00000 |
| fixed_beam40 | — | 40.0 | 0.05755±0.00100 | 未测 | 未测 | +0.00091 |
| fixed_beam60 | — | 60.0 | 0.05746±0.00092 | 未测 | 未测 | +0.00082 |
| random | 0 | 20.000±0.000 | 0.05664±0.00126 | 0.0±0.0 | 0.0000±0.0000 | +0.00000±0.00000 |
| entropy | 0 | 20.000±0.000 | 0.05664±0.00126 | 0.0±0.0 | 0.0000±0.0000 | +0.00000±0.00000 |
| ridge | 0 | 20.000±0.000 | 0.05664±0.00126 | 0.0±0.0 | 0.0000±0.0000 | +0.00000±0.00000 |
| hgb | 0 | 20.000±0.000 | 0.05664±0.00126 | 0.0±0.0 | 0.0000±0.0000 | +0.00000±0.00000 |
| random | 0.01 | 20.397±0.000 | 0.05665±0.00124 | 1.3±0.6 | 0.2352±0.3522 | +0.00001±0.00002 |
| entropy | 0.01 | 20.397±0.000 | 0.05665±0.00123 | 1.0±1.0 | 0.1391±0.0701 | +0.00001±0.00003 |
| ridge | 0.01 | 20.397±0.000 | 0.05664±0.00128 | 0.3±0.6 | 0.1192±0.1300 | -0.00000±0.00002 |
| hgb | 0.01 | 20.397±0.000 | 0.05665±0.00126 | 1.3±0.6 | 0.1872±0.1637 | +0.00001±0.00001 |
| random | 0.05 | 22.000±0.000 | 0.05672±0.00123 | 6.7±0.6 | 0.8493±0.2665 | +0.00007±0.00004 |
| entropy | 0.05 | 22.000±0.000 | 0.05669±0.00119 | 5.7±3.5 | 1.0208±0.1013 | +0.00005±0.00007 |
| ridge | 0.05 | 22.000±0.000 | 0.05667±0.00124 | 4.3±1.5 | 0.8309±0.5424 | +0.00003±0.00002 |
| hgb | 0.05 | 22.000±0.000 | 0.05668±0.00125 | 4.7±0.6 | 0.7757±0.2913 | +0.00004±0.00003 |
| random | 0.1 | 23.999±0.000 | 0.05679±0.00120 | 12.3±4.2 | 1.5724±0.5686 | +0.00015±0.00009 |
| entropy | 0.1 | 23.999±0.000 | 0.05672±0.00116 | 11.7±1.5 | 2.4296±0.6980 | +0.00007±0.00011 |
| ridge | 0.1 | 23.999±0.000 | 0.05673±0.00124 | 10.7±3.5 | 1.8503±0.5689 | +0.00008±0.00005 |
| hgb | 0.1 | 23.999±0.000 | 0.05678±0.00120 | 11.3±1.5 | 1.6480±0.5574 | +0.00014±0.00007 |
| random | 0.25 | 29.998±0.000 | 0.05686±0.00114 | 26.0±4.4 | 4.1359±1.5123 | +0.00022±0.00015 |
| entropy | 0.25 | 29.998±0.000 | 0.05694±0.00113 | 29.3±3.2 | 4.9172±1.2768 | +0.00029±0.00014 |
| ridge | 0.25 | 29.998±0.000 | 0.05677±0.00121 | 26.3±7.6 | 5.5849±0.7931 | +0.00013±0.00020 |
| hgb | 0.25 | 29.998±0.000 | 0.05694±0.00118 | 31.3±6.7 | 5.2443±0.8011 | +0.00030±0.00009 |
| random | 0.5 | 40.000±0.000 | 0.05720±0.00104 | 58.0±7.2 | 8.9229±1.7782 | +0.00055±0.00026 |
| entropy | 0.5 | 40.000±0.000 | 0.05713±0.00108 | 55.7±3.1 | 9.6903±1.5180 | +0.00049±0.00019 |
| ridge | 0.5 | 40.000±0.000 | 0.05710±0.00105 | 58.0±3.6 | 10.3534±1.6175 | +0.00046±0.00021 |
| hgb | 0.5 | 40.000±0.000 | 0.05709±0.00102 | 55.7±6.4 | 9.7725±1.3506 | +0.00044±0.00027 |

## Oracle（单列，不参与方法排名）

> **警告**：Oracle **读取真实 ΔNDCG@20**，属于**不可部署上界**，不是 Budget-SID 算法提升。不要写入普通方法排名，也不要把 oracle 的提升写成算法结果。

典型实际平均 beam ≈ **20.4**（只升级正 Δ，剩余配额不拿负/零 Δ 填满；因此与声明 25%/50% 的策略不算同算力）。

| 声明升级比例 | 实际平均 beam | NDCG@20 | 正收益捕获 | 负收益损失 | 净收益 Δ vs all-beam20 |
|---:|---:|---:|---:|---:|---:|
| 0 | 20.000±0.000 | 0.05664±0.00126 | 0.0 | 0.0000 | +0.00000 |
| 0.01 | 20.376±0.020 | 0.05908±0.00113 | 105.0 | 0.0000 | +0.00244 |
| 0.05 | 20.382±0.030 | 0.05912±0.00109 | 106.7 | 0.0000 | +0.00247 |
| 0.1 | 20.382±0.030 | 0.05912±0.00109 | 106.7 | 0.0000 | +0.00247 |
| 0.25 | 20.382±0.030 | 0.05912±0.00109 | 106.7 | 0.0000 | +0.00247 |
| 0.5 | 20.382±0.030 | 0.05912±0.00109 | 106.7 | 0.0000 | +0.00247 |

## 边界特征臂（附录，非主排名）

加入 `top20_cum_prob` / `prob_gap_p20_p21` / `prob_sum_ranks_21_60` 后，Ridge/HGB 仍未稳定优于 random/entropy。数字在 `docs/FINAL_RESULTS_TABLE.json` 的 `boundary_arm_brief_not_main`。原始曲线在数据盘的 `outputs/p2_replay_boundary/`，不在 git 里。

## 结论

On this Beauty checkpoint, tested features, and declared budget fractions, benefit prediction (Ridge/HGB) did not stably beat random or entropy on primary metrics. Online routing development stopped. No P3.

- 不启动 P3，不重训推荐模型，不扩展 Toys，不再增加特征或预测器。
- 线上收益路由开发停止。

## 未测缺口

- Latency / throughput / peak VRAM under routing: 未测
- Retrain recommender under budget policy: 未测 (explicitly out of archive scope)
- Toys / other datasets: 未测
- Online A/B deploy: 未做
- P3: 未启动

## 归档位置

大体积文件不进 git：checkpoint（`.ckpt`）、logits npz、`per_request.jsonl`、SID 张量（`.pt`）。

- 实例数据盘（关机后保留）：`/root/autodl-tmp/Budget-SID-archive-20261020/`（约 855M，含 ckpt / per_request / logits / 全阶段 P2 输出）
- 工作副本仍在：`/root/autodl-tmp/Budget-SID/`（同盘）
- 完整性：`BACKUP_OK.txt`、`HASHES_sha256.txt`、`SPOTCHECK_json.txt`、`FILE_COUNTS.txt`（都在上述归档目录里）

`artifacts/PROTOCOL_SUMMARY.json` 是 2026-10-17 固定 beam 跑完时的进度，里面的 `still_open` 不是这份归档结论。

## 公开复现（小文件）

公开仓库里能直接看的是本文件和 `docs/FINAL_RESULTS_TABLE.json`。下面的命令要在数据盘上的工作副本里跑，读的是已经写好的缓存，不下载 Beauty，不调用 GPU，不重训。

```bash
# 仓库：/root/autodl-tmp/Budget-SID
# 大文件也可从归档副本取：/root/autodl-tmp/Budget-SID-archive-20261020/

# 四特征比例曲线（CPU）。seeds 只重抽 A/B，不重训推荐模型。
python -m budget_sid replay_budgets \
  --per-request outputs/leak_safe_beams/per_request.jsonl \
  --summary outputs/leak_safe_beams/summary.json \
  --features outputs/leak_safe_beams/first_step_features.jsonl \
  --user-protocol temporal_loo \
  --seeds 42,43,44 \
  --fractions 0,0.01,0.05,0.10,0.25,0.50 \
  --output-dir outputs/p2_replay

# 边界特征是另一条臂，同样是 CPU 回放。
python -m budget_sid replay_budgets \
  --per-request outputs/leak_safe_beams/per_request.jsonl \
  --summary outputs/leak_safe_beams/summary.json \
  --features outputs/leak_safe_beams/boundary_features.jsonl \
  --user-protocol temporal_loo \
  --seeds 42,43,44 \
  --fractions 0,0.01,0.05,0.10,0.25,0.50 \
  --output-dir outputs/p2_replay_boundary
```

固定 beam 20/40/60 不是曲线上的策略点。它们是同一批 split-B 的 `request_id` 上，对 `per_request.jsonl` 里 beam 20/40/60 的 nDCG@20 做的跨 seed 平均。回放写在 `split_b` 里的 `all_20` / `all_40` / `all_60` 是同一次划分上的全体 beam，可以对上这三行。没有 `per_request.jsonl` 时，`python -m budget_sid replay_budgets --smoke` 只跑合成烟雾测试，不会重算出上表。字段说明见 `docs/P2_OFFLINE_REPLAY.md`。
