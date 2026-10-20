# Budget-SID

Budget-SID 在 Amazon Beauty 上做两件事：用轻量 Semantic ID 做生成式召回，以及按请求分配搜索预算。

生成式召回先把商品编成一串短码（Semantic ID），再用一个小 Transformer 根据用户历史把下一件商品的码写出来。搜索时 beam 越宽越不容易漏掉目标，也越慢。预算分配要猜的是：这条请求值不值得把 beam 从 20 加到 60。返回的推荐始终是 Top-20。

公共仓库：<https://github.com/JOrangsepha/Budget-SID>

GRID 不放进本仓库，只用 `scripts/pin_grid.sh` 锁在 commit `2fe3475b2d369580234093f35d52b1a2f54d0472`。

范围就是预算分配，加上这条轻量 SID 生成式召回实验。TIGER 论文里的模型创新不归本仓库，这里的数字也不是 TIGER、Semantic ID 或 GRID 论文的复现结果。

2026-10-17 在 SeetaCloud RTX 4090 上跑完前半段：冻结 MiniLM、RQ-KMeans、Semantic ID、`tiger_train_flat`、固定 beam 推理。泄漏安全评测、P1 前提和 P2.1 离线回放后来也做完了。2026-10-20 的结论是收益预测没有稳定打赢随机或熵，线上路由停止，不启动 P3。写这份说明时没有再跑 GPU，也没有重训。

## English

- Budget allocation plus a light SID generative-recall experiment on Amazon Beauty. Online benefit routing is archived.
- Pinned GRID commit `2fe3475b2d369580234093f35d52b1a2f54d0472`. GRID is not vendored.
- 2026-10-17 SeetaCloud RTX 4090: fixed beams 20/40/60 finished (exit 0). That day's run did not wire a benefit-aware stop-gate to GRID outputs. The later offline replay did test Ridge and HGB on leak-safe labels. They did not stably beat random or entropy, so the online gate was not wired and P3 was not started.
- P1 reads leak-safe `per_request.jsonl` when that file is present. The transcribed SeetaCloud summary does not clear P2. The validation pass did (`p2_worth_doing=true`). See `docs/P1_BENEFIT_ANALYSIS.md` and `docs/P2_OFFLINE_REPLAY.md`. The routing decision after that pass is the archive section below.
- Archive (2026-10-20): on `checkpoint_epoch=000_step=018800.ckpt`, the tested features, and the declared upgrade fractions, Ridge and HGB did not stably beat random or entropy on mean nDCG@20 or net Δ versus all-beam 20. Online routing development is stopped. No P3, no recommender retrain, no Toys, no new features or predictors. The main table is validation split B, seeds 42/43/44, and it includes fixed beam 20/40/60 on that same set. Oracle reads true Δ and stays in its own section (mean beam ≈20.4). Numbers: `docs/FINAL_RESULTS.md` and `docs/FINAL_RESULTS_TABLE.json`. Large artifacts stay on the SeetaCloud data disk.
- P0 eval audit does not retrain and does not claim a benefit-aware gain. It exports leak-safe history/target samples and can score NDCG@5, NDCG@10, and NDCG@20 per request. See `docs/P0_EVAL_AUDIT.md`. The 2026-10-17 `tiger_inference_flat` rankings keep the target event in the encoder input, so they are not leak-safe ΔNDCG. Beams were re-run with `scripts/run_leak_safe_beams.sh` on the exported history; those files are on the data disk, not in git. The logged test nDCG@5 and nDCG@10 came from `NextKTokenMasking` and are not that leak.
- Checkpoint stays on the GPU machine. It is not in git.
- Scripts log `SKIP_SHUTDOWN` and do not power the instance off.

## 归档结论（2026-10-20）

收益路由到这里停。不启动 P3，不重训推荐模型，不扩展 Toys，不再加特征或预测器。

用的还是当前 Beauty checkpoint `checkpoint_epoch=000_step=018800.ckpt`。测过的特征是四特征（`history_length`、`first_step_entropy`、`candidate_prob_gap`、`top1_prob`）和边界特征（`top20_cum_prob`、`prob_gap_p20_p21`、`prob_sum_ranks_21_60`）。升级比例是事先声明的 0%、1%、5%、10%、25%、50%。在这个范围内，Ridge 和 HGB 没有在主指标上稳定优于 random 或 entropy。主指标是平均 nDCG@20，以及相对全体 beam 20 的净 Δ。线上路由不再做。

主表和预算策略用同一批 validation split B（seeds 42、43、44，每 seed 大约 11182 条）。固定 beam 20/40/60 也是这批请求。不要拿全体验证集的 beam 40（0.05750）或全体 leak_safe（n=67089，beam 20 约 0.06356）和这张表比，那是另一批请求。完整主表在 `docs/FINAL_RESULTS.md`，同一份数字的 JSON 在 `docs/FINAL_RESULTS_TABLE.json`。

同一 validation-B 上的固定 beam（跨 seed 的均值 ± 标准差）。正收益捕获和负收益损失对固定 beam 没有按策略去记，标成未测。净 Δ 是相对这批请求上的全体 beam 20。

| 方法 | 升级比例 | 实际平均 beam | NDCG@20 | 正收益捕获 | 负收益损失 | 净收益 Δ vs all-beam20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed_beam20 | — | 20.0 | 0.05664±0.00126 | 未测 | 未测 | +0.00000 |
| fixed_beam40 | — | 40.0 | 0.05755±0.00100 | 未测 | 未测 | +0.00091 |
| fixed_beam60 | — | 60.0 | 0.05746±0.00092 | 未测 | 未测 | +0.00082 |

random、entropy、ridge、hgb 在同一张表里，不在上表重复。ridge / hgb 没有稳定高于 random 或 entropy。升级比例 0.5 时，random 的 nDCG@20 是 0.05720±0.00104，entropy 是 0.05713±0.00108，ridge 是 0.05710±0.00105，hgb 是 0.05709±0.00102。其余比例见 `docs/FINAL_RESULTS.md`。

Oracle 单独放，不进方法排名。它读的是真实 ΔNDCG@20，只升级正 Δ，剩下的名额不用负 Δ 或零 Δ 填满，所以实际平均 beam 大约 20.4。这是不可部署的上界。它的提升不是 Budget-SID 的算法提升。

大文件不进 git：checkpoint、logits 的 npz、`per_request.jsonl`、SID 张量。它们在 SeetaCloud 数据盘：

```text
/root/autodl-tmp/Budget-SID-archive-20261020/
/root/autodl-tmp/Budget-SID/
```

第一处是 2026-10-20 的归档副本（约 855M，含 P0 审计、leak-safe 逐请求、首步和边界特征与 logits、P1、各阶段 P2、ckpt、SID 映射、生效配置和日志）。第二处是同一块盘上的工作目录。完整性记在归档目录里的 `BACKUP_OK.txt`、`HASHES_sha256.txt`、`SPOTCHECK_json.txt`、`FILE_COUNTS.txt`。公开仓库只放复现命令和这两份小汇总。

在数据盘上、已经有逐请求缓存时，用 CPU 重放比例曲线（不启 GPU，不重训）：

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

边界特征把 `--features` 换成 `outputs/leak_safe_beams/boundary_features.jsonl`，输出目录用 `outputs/p2_replay_boundary`。固定 beam 20/40/60 是同一批 split-B 请求上 `per_request.jsonl` 的 beam 列。没有这些大文件时，`python -m budget_sid replay_budgets --smoke` 只跑合成烟雾测试，不会重算出上表。`artifacts/PROTOCOL_SUMMARY.json` 记的是 2026-10-17 当天的进度，里面的 `still_open` 不是这次归档结论。

## Beauty 流水线

数据用 GRID 已提供的 Amazon P5 Beauty。锁定配置认的目录名是：

```text
data/amazon_data/beauty/{training,evaluation,testing,items}
```

GRID README 示意图里有时写成 `train` / `validation` / `test`。以 pinned 配置为准，用 `training` / `evaluation` / `testing` / `items`。数据不进 git（`data/*` 已忽略）。

训练和评测用的历史不能包含正在预测的那一次交互。测试用户不参与收益模型，也不参与选阈值。

文本侧用冻结的 `sentence-transformers/all-MiniLM-L6-v2`，384 维。不要跑 GRID 默认的 FLAN-T5-XL。

Semantic ID 用 RQ-KMeans：3 个 codebook，宽度 256，再加一位用来消解冲突。RQ-KMeans 的 `num_hierarchies=3`，生成模型的 `num_hierarchies=4`。

生成模型的目标配置在 `configs/lightweight_beauty.yaml`：`d_model=128`，encoder / decoder 各 4 层，`num_heads=4`，每卡 batch 16，梯度累积 4，单卡，不做 DDP。配置加载器会拒绝 FLAN-T5-XL 和 DDP。GPU 脚本传给 GRID 的覆盖和 `budget_sid.grid_adapter` 一样，改的是 `d_model`、层数、batch 和梯度累积。`num_heads` 没有单独覆盖，GRID flat 配置里的默认值是 6；这次 4090 日志没有记下实际用了几头。

## 怎么跑

### CPU 脚手架

不下载 Amazon，不调用 GRID，不访问付费 API。

```bash
pip install -e ".[dev]"
pytest
python -m budget_sid train_recommender --config configs/lightweight_beauty.yaml --smoke
python -m budget_sid build_semantic_ids --config configs/lightweight_beauty.yaml --smoke
python -m budget_sid train_benefit_predictor --config configs/lightweight_beauty.yaml --smoke
python -m budget_sid evaluate_budgets --config configs/lightweight_beauty.yaml --smoke
python -m budget_sid replay_budgets --config configs/lightweight_beauty.yaml --smoke
python -m budget_sid demo --config configs/lightweight_beauty.yaml --smoke --history 1,2,3
```

不带 `--smoke` 时，Python 入口只打印命令计划并返回退出码 1。它不会读测试标签，也不会把下面的 4090 结果当成刚刚算出来的。

### GPU 脚本（单卡）

GRID 命令在 `third_party/GRID` 里执行。

```bash
bash scripts/pin_grid.sh
# EMBEDDING_PATH：冻结 MiniLM 的 384 维缓存。Beauty 目录在机器上，不在 git 里。
EMBEDDING_PATH=/path/to/minilm_beauty.pt bash scripts/run_beauty_gpu.sh
```

`run_beauty_gpu.sh` 的顺序：检查 MiniLM 缓存 → RQ-KMeans 训练和推理 → 你把 SID 张量路径设为 `SID_PATH` 后再跑 → `tiger_train_flat` → 固定 beam 20/40/60。RQ-KMeans 成功但还没有 `SID_PATH` 时，脚本退出，不会写 `tiger_train.done`。

训练已经结束、只重跑固定 beam：

```bash
CKPT=/path/to/checkpoint_epoch=000_step=018800.ckpt \
  SID_PATH=/path/to/semantic_ids.pt \
  bash scripts/run_after_tiger.sh
```

也可以 `SKIP_TRAIN=1 CKPT=... SID_PATH=... bash scripts/run_beauty_gpu.sh`。

两支脚本都只在该阶段命令退出码为 0 时写 `.done`。可选：`scripts/push_github.sh` 推送当前分支，并拒绝提交 `third_party/GRID/`。

## Beauty 结果（SeetaCloud，2026-10-17）

机器：SeetaCloud RTX 4090。全部结束：2026-10-17 12:09（Asia/Shanghai）。实例按 Josepha 的要求保持开机。

最好的 checkpoint（提前停止）：`checkpoint_epoch=000_step=018800.ckpt`。早停看的是 `val/recall@5`，最好大约 **0.059**。

文件在 GPU 上，相对 `third_party/GRID/`：

```text
logs/train/runs/2026-10-17/00-35-10/checkpoints/checkpoint_epoch=000_step=018800.ckpt
```

checkpoint 不进 git，太大。

GRID 这次评测打的是 @5 和 @10。协议里的 Recall@20、NDCG@20、延迟和显存这次没有记，表里就不补。后来的泄漏安全打分才有 nDCG@20。路由结论用的是 validation split B 上的那组，见上面的归档结论，不要用这里的 @5 / @10 代替。

| 切分 | nDCG@5 | nDCG@10 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| 测试 | 0.0293 | 0.0366 | 0.0456 | 0.0682 |
| 验证 | 0.0361 | 0.0453 | ≈0.059（早停监控） | 未记录 |

固定 beam 推理（`top_k` 20 / 40 / 60），每个大约 22363 行：

| top_k | 退出码 | 大约行数 | 完成（北京时间） |
| --- | ---: | ---: | --- |
| 20 | 0 | 22363 | 12:06 |
| 40 | 0 | 22363 | 12:07 |
| 60 | 0 | 22363 | 12:09 |

原文记在 `artifacts/BEAUTY_GPU_RESULTS.json`，协议进度在 `artifacts/PROTOCOL_SUMMARY.json`。来源是 SeetaCloud 2026-10-17 的日志。

## Hydra 两个坑

checkpoint 文件名里有 `=`：`checkpoint_epoch=000_step=018800.ckpt`。Hydra 会把没有引号的 `=` 拆成新参数。覆盖要写成：

```bash
ckpt_path='$CKPT'
```

脚本用双引号包住整段（`ckpt_path='$CKPT'` 这一整段），shell 展开 `$CKPT`，单引号留给 Hydra。

推理读的是 `predict_dataloader_config`，不是 `test_dataloader_config`。`tiger_inference_flat` 没有 test dataloader 这个键。`num_workers=0` 时 PyTorch 要求 `timeout` 也是 0：

```bash
data_loading.predict_dataloader_config.dataloader.num_workers=0
data_loading.predict_dataloader_config.dataloader.timeout=0
```

单卡还要带上 `trainer.strategy=auto`、`trainer.devices=1`、`trainer.accelerator=gpu`。不要把 `strategy` 设回 `ddp`，也不要 `devices=-1`。

## 跑完不要关机

脚本在退出时打印 `SKIP_SHUTDOWN`，不调用 autodl poweroff，也不调用 shutdown。2026-10-17 那台机器是故意留着开机的。

## 预算分配（离线测完，线上已停）

搜索宽度只在 {20, 40, 60} 里选，返回值始终是 Top-20。Beam 是搜索宽度，不是截断长度。

离线回放按下面这套规则测过。Ridge / HGB 没有稳定打赢 random 或 entropy，所以没有把收益预测接到线上的 GRID 推理里，P3 也没有启动。数字在 `docs/FINAL_RESULTS.md`。

当时定下的做法：

1. 所有请求共用同一次首步解码。
2. 从首步取出历史长度、熵、top1 与 top2 的概率差、top1 概率。
3. 预测 ΔNDCG@20 = NDCG@20(beam 60) − NDCG@20(beam 20)。
4. 只按选定宽度继续搜一次。不要先跑完小 beam，再整段重跑大 beam。过滤、补齐和路由都算进延迟。
5. 验证用户分成不相交的 A / B。A 训练收益回归。B 在平均 beam 不超过 40 的前提下选阈值，所以最多一半请求走 beam 60。测试集不参与训练、选阈值或调参。
6. 同一 checkpoint 上比较固定 20 / 40 / 60、随机预算、按熵、收益感知。种子 42、43、44。全目录计算 Recall@20 和 NDCG@20，不用随机负例。切片包括长尾商品（频次最低 20%）和短历史用户（历史长度 ≤ 5）。
7. 止损在验证集上做。大 beam 没有集中收益，或者理想预算分配打不赢全体 beam 40，进程退出码为 2。退出码 2 表示实验前提不成立，不是程序崩溃。

2026-10-17 那次 4090 只跑了固定 beam，当时还没把请求级止损接到 GRID `tiger_inference` 上。后来的离线回放用泄漏安全标签把分配质量测完了，结论是停止，不再去接线上推理。`evaluate_budgets` 不带 `--smoke` 时仍然只给命令计划（退出码 1），不会自己重跑这张表。

## P0 评测审计

固定 beam 的排序要能离线算 ΔNDCG@20，得先证明每条请求的历史里没有这次要预测的交互，并且 beam 20/40/60 用的是同一批样本。这一步不重新训练，也不宣称收益感知有收益。

GRID 的 `tiger_train_flat` 用 `NextKTokenMasking` 把最后一个 Semantic ID 组遮住，所以日志里的测试 nDCG@5、nDCG@10 不是「模型看见了目标」这种泄漏。`tiger_inference_flat` 用的 `collate_fn_inference_for_sequence` 只做补齐和截断，截断留的是序列尾部，目标事件还在编码器输入里。2026-10-17 那三份 beam 排序因此不能直接当泄漏安全的 ΔNDCG。

Beauty 是按时间的 leave-one-out：同一 `user_id` 会出现在 training、evaluation、testing。用户重叠不是标签泄漏。标签泄漏指的是目标那一次交互还在历史里。更早买过同一个商品可以留下。`assert_test_unused` 仍只用于脚本里「每个用户只属于一个切分」的烟雾测试；Beauty 走 `temporal_loo`，锁的是测试事件，不是用户 id。

```bash
python -m budget_sid export_eval_samples --smoke
python -m budget_sid audit_eval_leakage --smoke
python -m budget_sid score_beam_ndcg --smoke
```

SeetaCloud 上对已有 Beauty 文件（不在 git 里）：

```bash
# 仓库在 /root/autodl-tmp/Budget-SID
SID_PATH=/path/to/semantic_ids.pt bash scripts/p0_eval_audit.sh
```

P0 导出之后，用历史（不含目标事件）重跑固定 beam 20/40/60，再按 `leak_safe_history` 打分。不要把 2026-10-17 的 `tiger_inference_flat` pickle 拿来算 ΔNDCG。

```bash
SID_PATH=/path/to/semantic_ids.pt bash scripts/run_leak_safe_beams.sh
```

字段、审计结论、截断窗口，以及上面这条命令，写在 `docs/P0_EVAL_AUDIT.md`。

## P1 止损

泄漏安全汇总（`outputs/leak_safe_beams/summary.json`，n=67089）里，beam 60 的 nDCG@20 是 0.06418，beam 40 是 0.06425，平均 ΔnDCG@20（60−20）是 0.000618。这个均值本身不能放行 P2。逐条标签不在 git 里。SeetaCloud 上全体验证集的后续结果是 `p2_worth_doing=true`：n=22363，正 Δ 0.926%（207），零 97.50%，负 1.57%，oracle nDCG@20 0.05911，高于全体验证 beam 40 的 0.05750，平均 beam 约 20.37。全体 beam 60 不高于 beam 40。0.05750 是全体验证行，不是归档主表里的 validation split B。主表见 `docs/FINAL_RESULTS.md`。这组全体验证数是报告值，本仓库没有重算。机器上有 `per_request.jsonl` 时可以再核对：

```bash
python -m budget_sid analyze_benefit \
  --per-request outputs/leak_safe_beams/per_request.jsonl \
  --summary outputs/leak_safe_beams/summary.json
```

汇总均值不能把 `p2_worth_doing` 设成 true。GridRecommender 没有开始，这里也不声称线上收益。说明在 `docs/P1_BENEFIT_ANALYSIS.md`。

## P2 离线回放

P2.1 只回答分配质量。验证用户分成不相交的 A / B。A 上用已有的标准化 Ridge 拟合 ΔNDCG@20；装了 scikit-learn 时，同一批列再拟合一个 `max_depth=3`、`max_iter=50` 的 HistGradientBoosting，和 Ridge 花同一笔名额。B 上按预先声明的升级比例扫一条曲线，默认 0%、1%、5%、10%、25%、50%。比例 `f` 把这么多请求从 beam 20 升到 60，声明的平均 beam 是 `20 + 40×f`。随机、熵（特征里有熵才比较）、收益感知花掉的名额一样大。名额是 `floor(f × n_B)`，不从 B 上正 Δ 的条数推出来。

主指标是最终 nDCG@20，以及相对全体 beam 20 的净 Δ。抓到多少条正 Δ、负 Δ 亏了多少，写在 `explanatory` 里，不拿来当主结果。MSE 只是拟合诊断。选择请求时不读该请求的 Δ。测试事件不参与。种子 42、43、44 只重抽 A/B 和随机策略，不是把推荐模型重训三遍。

Oracle 上界单独报，只升级正 Δ，按 Δ 从高到低，花到正例用完或名额用完就停。剩下的名额留在 beam 20，不用负 Δ 去填满。它的平均 beam 是实际花掉的计算量。

旧的窄名额是 `n_quota = min(B 上正 Δ 条数, 平均 beam 40 的上限)`。SeetaCloud 上两次回放都没能在这个名额下打赢随机和熵：只接 `history_length` 时，Ridge 大约抓到 0% 的正 Δ 质量，nDCG@20 约 0.0566，和随机的约 0.0567 打平或更差，熵比较被跳过；四特征在那条窄名额下同样没有打赢随机和熵。`--quota-mode positive_count` 还能复现那条窄名额，但默认已经换成固定比例。

固定比例曲线后来在同一 checkpoint 上跑完了。四特征（history length、熵、概率差、top1）和边界特征都没有在 seeds 42、43、44 上稳定打赢 random 和 entropy。主表包括同一 validation-B 上的固定 beam 20/40/60，见 `docs/FINAL_RESULTS.md`。P3 不启动。线上路由停止。

`per_request.jsonl` 本身没有首步 logits。缺列写在 `full_p2_blockers` 里，不补假 logits。四特征曲线是在已有的 `per_request.jsonl` 和 `first_step_features.jsonl` 上用 CPU 跑的。边界特征用的是验证集第 0 层 logits：`scripts/dump_first_step_logits.sh` 把向量缓存成 npz（默认只跑验证集；没有 `predict_rows.jsonl` 时先跑全量再丢掉非验证行），`extract_boundary_features` 在 CPU 上写出 `top20_cum_prob`、`prob_gap_p20_p21`、`prob_sum_ranks_21_60`，再用 `replay_budgets --features` 接上。`request_id` 对回放，`predict_key` 对 GRID 的 `user_id`。目标是否落在 top-20 里是事后归因，不进路由特征。这些大文件在数据盘归档里，不在 git 里。边界臂也没有打赢 random 或 entropy。

首步特征用泄漏安全的历史目录（目标事件不在历史里），按 `predict_index.jsonl` 的 `request_id` 顺序写出。GRID 的 `predict_step` 不返回 logits；dump 调用的是 pinned `generate()` 在第 0 层算出 `candidate_logits` 的那三步，然后停住。checkpoint 或张量缺失就非零退出，不写零向量。这不是延迟，也不是线上收益感知。

```bash
python -m budget_sid replay_budgets --smoke
python -m budget_sid dump_first_step_features --smoke
python -m budget_sid dump_first_step_logits --smoke
SID_PATH=/path/to/semantic_ids.pt bash scripts/dump_first_step_logits.sh
python -m budget_sid replay_budgets \
  --per-request outputs/leak_safe_beams/per_request.jsonl \
  --summary outputs/leak_safe_beams/summary.json \
  --features outputs/leak_safe_beams/boundary_features.jsonl \
  --user-protocol temporal_loo \
  --seeds 42,43,44 \
  --fractions 0,0.01,0.05,0.10,0.25,0.50 \
  --output-dir outputs/p2_replay_boundary
```

字段、对齐方式和 SeetaCloud 命令在 `docs/P2_OFFLINE_REPLAY.md`。

## TODO

收益路由在 2026-10-20 归档。下面已完成的是已经跑过的步骤。未测的不再排进下一次实验：不启动 P3，不重训，不扩展 Toys。

已完成（SeetaCloud RTX 4090，单卡，无 DDP）：

- [x] `bash scripts/pin_grid.sh`，HEAD 为 `2fe3475b2d369580234093f35d52b1a2f54d0472`
- [x] 机器上准备了 `data/amazon_data/beauty/{training,evaluation,testing,items}`（不进 git）
- [x] 冻结 MiniLM，384 维；没有加载 FLAN-T5-XL
- [x] RQ-KMeans（3 个 codebook）+ SID，再训练 4+4 层、`d_model=128` 的生成模型；早停，最好的 `val/recall@5` ≈ 0.059
- [x] 固定 beam / `top_k` 20、40、60 推理，退出码都是 0（2026-10-17）
- [x] 离线 P2 窄名额诊断：history-length-only 和四特征都没有在 `min(正 Δ 条数, ceiling)` 下打赢随机和熵。见 `docs/P2_OFFLINE_REPLAY.md`。
- [x] P2.1 固定比例曲线（0/1/5/10/25/50%，seeds 42/43/44，validation split B）：四特征上的 Ridge / HGB 没有稳定优于 random 或 entropy。同一批请求上的固定 beam 20/40/60 写在 `docs/FINAL_RESULTS.md`。
- [x] 边界特征臂（`top20_cum_prob`、`prob_gap_p20_p21`、`prob_sum_ranks_21_60`）同样没有稳定打赢 random 或 entropy。logits npz 在数据盘，不在 git。
- [x] 止损结论：停止线上路由，不启动 P3。

未测（归档范围之外，不再排期）：

- [ ] 路由下的延迟、吞吐、峰值显存：未测
- [ ] 按预算策略重训推荐模型：未测
- [ ] Toys 或其他数据集：未测
- [ ] 线上 A/B：未做
- [ ] P3：未启动
- [ ] 把收益预测接到 GRID `tiger_inference` 的在线输出上：未做。离线门已经没过，不再接。
- [ ] 长尾商品和短历史切片：未测
- [ ] 复查 P0 `verified_with_truncation`：本归档不重跑 GPU 审计。见 `docs/P0_EVAL_AUDIT.md`。

## 引用

- GRID / Semantic IDs handbook: <https://research.snap.com/publications/semantic-ids-in-generative-recommendation-a-practitioners-handbook.html>
- GRID repo: <https://github.com/snap-research/GRID>
- TIGER: <https://arxiv.org/abs/2305.05065>
- GR4AD（beam / budget）: <https://arxiv.org/html/2602.22732v2>
- MiniLM: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
