# Budget-SID

Budget-SID 在 Amazon Beauty 上做两件事：用轻量 Semantic ID 做生成式召回，以及按请求分配搜索预算。

生成式召回先把商品编成一串短码（Semantic ID），再用一个小 Transformer 根据用户历史把下一件商品的码写出来。搜索时 beam 越宽越不容易漏掉目标，也越慢。预算分配要猜的是：这条请求值不值得把 beam 从 20 加到 60。返回的推荐始终是 Top-20。

公共仓库：<https://github.com/JOrangsepha/Budget-SID>

GRID 不放进本仓库，只用 `scripts/pin_grid.sh` 锁在 commit `2fe3475b2d369580234093f35d52b1a2f54d0472`。

范围就是预算分配，加上这条轻量 SID 生成式召回实验。TIGER 论文里的模型创新不归本仓库，这里的数字也不是 TIGER、Semantic ID 或 GRID 论文的复现结果。

2026-10-17 在 SeetaCloud RTX 4090 上跑完的是前半段：冻结 MiniLM、RQ-KMeans、Semantic ID、`tiger_train_flat`、固定 beam 推理。按请求估计 ΔNDCG@20 的止损和路由还没接到 GRID 的推理输出上。写这份说明时没有再跑 GPU。

## English

- Budget allocation plus a light SID generative-recall experiment on Amazon Beauty.
- Pinned GRID commit `2fe3475b2d369580234093f35d52b1a2f54d0472`. GRID is not vendored.
- 2026-10-17 SeetaCloud RTX 4090: fixed beams 20/40/60 finished (exit 0). Benefit-aware stop-gate is not wired.
- P0 eval audit does not retrain and does not claim a benefit-aware gain. It exports leak-safe history/target samples and can score NDCG@5, NDCG@10, and NDCG@20 per request. See `docs/P0_EVAL_AUDIT.md`. The 2026-10-17 `tiger_inference_flat` rankings keep the target event in the encoder input, so they are not leak-safe ΔNDCG. Re-run beams with `scripts/run_leak_safe_beams.sh` on the exported history. The logged test nDCG@5 and nDCG@10 came from `NextKTokenMasking` and are not that leak.
- Checkpoint stays on the GPU machine. It is not in git.
- Scripts log `SKIP_SHUTDOWN` and do not power the instance off.

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

GRID 这次评测打的是 @5 和 @10。协议里的 Recall@20、NDCG@20、延迟和显存这次没有记，表里就不补。

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

## 预算分配（还没接上）

搜索宽度只在 {20, 40, 60} 里选，返回值始终是 Top-20。Beam 是搜索宽度，不是截断长度。

计划中的做法：

1. 所有请求共用同一次首步解码。
2. 从首步取出历史长度、熵、top1 与 top2 的概率差、top1 概率。
3. 预测 ΔNDCG@20 = NDCG@20(beam 60) − NDCG@20(beam 20)。
4. 只按选定宽度继续搜一次。不要先跑完小 beam，再整段重跑大 beam。过滤、补齐和路由都算进延迟。
5. 验证用户分成不相交的 A / B。A 训练收益回归。B 在平均 beam 不超过 40 的前提下选阈值，所以最多一半请求走 beam 60。测试集不参与训练、选阈值或调参。
6. 同一 checkpoint 上比较固定 20 / 40 / 60、随机预算、按熵、收益感知。种子 42、43、44。全目录计算 Recall@20 和 NDCG@20，不用随机负例。切片包括长尾商品（频次最低 20%）和短历史用户（历史长度 ≤ 5）。
7. 止损在验证集上做。大 beam 没有集中收益，或者理想预算分配打不赢全体 beam 40，进程退出码为 2。退出码 2 表示实验前提不成立，不是程序崩溃。

这次 4090 只跑了固定 beam。请求级收益感知止损还没接到 GRID `tiger_inference` 的输出上。`evaluate_budgets` 不带 `--smoke` 时仍然只给命令计划（退出码 1）。

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

## TODO

原来的 TODO(gpu) 分成两截。固定 beam 那几步已经在 SeetaCloud 上做完；收益止损还开着。

已完成（SeetaCloud RTX 4090，2026-10-17，单卡，无 DDP）：

- [x] `bash scripts/pin_grid.sh`，HEAD 为 `2fe3475b2d369580234093f35d52b1a2f54d0472`
- [x] 机器上准备了 `data/amazon_data/beauty/{training,evaluation,testing,items}`（不进 git）
- [x] 冻结 MiniLM，384 维；没有加载 FLAN-T5-XL
- [x] RQ-KMeans（3 个 codebook）+ SID，再训练 4+4 层、`d_model=128` 的生成模型；早停，最好的 `val/recall@5` ≈ 0.059
- [x] 固定 beam / `top_k` 20、40、60 推理，退出码都是 0

未完成：

- [ ] 把请求级收益预测接到 GRID `tiger_inference` 的输出上（ΔNDCG@20）
- [ ] 验证集 A 训练收益模型，B 上选平均 beam ≤ 40 的阈值
- [ ] 在验证集上跑止损。退出码 2 就停，不要再加路由
- [ ] 同一 checkpoint 上比较固定 20/40/60、随机、熵、收益感知
- [ ] 三个种子，记录 Recall@20、NDCG@20、延迟均值 / P95、吞吐、峰值显存，以及长尾和短历史切片

## 引用

- GRID / Semantic IDs handbook: <https://research.snap.com/publications/semantic-ids-in-generative-recommendation-a-practitioners-handbook.html>
- GRID repo: <https://github.com/snap-research/GRID>
- TIGER: <https://arxiv.org/abs/2305.05065>
- GR4AD（beam / budget）: <https://arxiv.org/html/2602.22732v2>
- MiniLM: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
