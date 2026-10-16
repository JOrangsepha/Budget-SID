# Budget-SID

基于 [Snap Research GRID](https://github.com/snap-research/GRID) 的轻量 TIGER-style 生成式召回，加上请求级搜索预算分配。

公共代码仓库（source of truth）：**<https://github.com/JOrangsepha/Budget-SID>**

这是「基于 GRID 的轻量 TIGER-style 实现与扩展」，不是 TIGER、Semantic ID 或 GRID 论文数值的严格复现。

可归属于本项目的部分：

- 预算分配模块：用低成本特征预测「大 beam 相对小 beam 的 ΔNDCG@20」，再决定搜索宽度
- 低资源路径：冻结 MiniLM、单卡、小 Transformer，拒绝 FLAN-T5-XL 和 DDP 默认配置
- 可复现评测：同一 checkpoint 上的预算对照、全目录指标、长尾与短历史切片、验证集止损检查

TIGER 与 Semantic ID 本身不是本项目的创新。GRID 不打进本仓库，只通过 `scripts/pin_grid.sh` 锁定上游 commit。

## English

Budget-SID is a lightweight TIGER-style generative retriever built on GRID, plus a request-level beam budget. The public source of truth is <https://github.com/JOrangsepha/Budget-SID>. It is not a reproduction of the TIGER, Semantic ID, or GRID papers. Ours is the budget module, the low-resource path, and the reproducible eval. GRID stays upstream at a pinned commit.

## 环境

```bash
pip install -e ".[dev]"
pytest
```

CPU 单元测试使用脚本推荐器与数组替身，不下载 Amazon 数据，不调用付费 API。`--smoke` 走同一条 CPU 路径。

## GRID 锁定

| 项 | 值 |
| --- | --- |
| 上游 | https://github.com/snap-research/GRID |
| Commit | `2fe3475b2d369580234093f35d52b1a2f54d0472` |
| 检出位置 | `third_party/GRID`（gitignore，不入库） |
| 脚本 | `scripts/pin_grid.sh` |

```bash
bash scripts/pin_grid.sh
```

脚本只做 clone / fetch / detached checkout。工作区若有本地改动会直接失败，不会改写 GRID 源码。

GRID 的 `sem_embeds_inference_flat` 默认 `embedding_model: google/flan-t5-xl`，`tiger_train_flat` / `rkmeans_train_flat` 默认 `strategy: ddp`（或 `ddp_find_unused_parameters_true`）且 `devices: -1`。本仓库的配置加载器遇到这些值会拒绝加载。适配器发出的 Hydra 覆盖把 RQ-KMeans 的 `embedding_dim` 设为 384，并把 trainer 收成单卡；MiniLM 不送进 GRID 的 T5 encoder。

可选：`scripts/push_github.sh` 把当前分支推到上述 GitHub 仓库，并拒绝提交 `third_party/GRID/`。

## Beauty 路径

数据用 GRID 已提供的 Amazon P5 预处理包（Beauty，之后可用 Toys 再验一次）。下载入口见 GRID README 中的 Google Drive 链接。放到：

```text
data/amazon_data/beauty/
├── training/     # 训练序列
├── evaluation/   # 验证序列
├── testing/      # 测试序列
└── items/        # 商品文本
```

锁定 commit 里的 flat 配置实际读取的目录名是 `training` / `evaluation` / `testing` / `items`。GRID README 示意图里的 `train` / `validation` / `test` 与加载器不一致，以 pinned 配置为准。

训练与评测用的历史不得包含待预测的那次交互。`budget_sid.data_audit` 会拒绝这种泄漏。测试用户不参与收益模型训练，也不参与阈值选择。

文本侧：冻结 `sentence-transformers/all-MiniLM-L6-v2`，缓存 384 维向量。不要跑 GRID 自带的 FLAN-T5-XL embedding 命令。

Semantic ID：RQ-KMeans，3 个 codebook，宽度 256，再加 TIGER 用来消解冲突的一位。因此 TIGER 的 `num_hierarchies=4`（3+1），RQ-KMeans 训练时 `num_hierarchies=3`。CPU 上的 `--smoke` 用本包里的 numpy 残差量化做替身；正式 centroid 仍交给 pinned GRID。

推荐模型：`d_model=128`，encoder / decoder 各 4 层，`num_heads=4`，每卡 batch 16，梯度累积 4，`strategy: single_device`，`devices: 1`。配置在 `configs/lightweight_beauty.yaml`。

## 推理与预算

搜索宽度只在 {20, 40, 60} 中取，返回值始终是 Top-20。Beam 是搜索宽度，不是截断长度。

路由不是「先跑完小 beam，再整段重跑大 beam」：

1. 所有请求共用同一次首步解码
2. 从首步分布取出历史长度、熵、候选概率差（top1−top2）、top1 概率
3. 收益预测器估计 ΔNDCG@20 = NDCG@20(beam 60) − NDCG@20(beam 20)
4. 只按选定宽度继续搜索一次
5. 合法 Semantic ID 过滤、补齐和路由本身都计入延迟

验证用户划成互不相交的 A / B：

- A：训练收益回归器
- B：在平均 beam 不超过 40 的前提下选阈值。60p + 20(1−p) = 40，所以最多一半请求走 beam 60
- 测试集不参与训练、阈值或调参

同一 checkpoint 上的对照：固定 20 / 40 / 60、随机预算、按首步熵分配、收益感知分配。最终方案用 3 个种子（42、43、44）报告均值和标准差。召回按全目录计算，不用随机负例。

## 指标与止损

指标：Recall@20、NDCG@20、平均延迟、P95 延迟、吞吐、峰值显存。切片：长尾商品（频次处于频率分布最低 20%）、短历史用户（历史长度 ≤ 5）。

止损在验证集上做，命中则进程退出码为 2：

- 大 beam 没有集中收益。平均 ΔNDCG@20 ≤ 0，或者正收益既不集中在少数请求（正收益用户比例 > 0.5）也不被最高的 20% 请求拿走至少一半正收益
- 或者理想预算分配打不赢全体 beam 40。理想分配按真实 ΔNDCG@20 从高到低把预算发给 beam 60，人数取使平均宽度不超过 40 的最大人数

两条都通过才继续加路由。退出码 2 表示实验前提不成立，不是程序崩溃。

## 命令

```bash
python -m budget_sid train_recommender --config configs/lightweight_beauty.yaml --smoke
python -m budget_sid build_semantic_ids --config configs/lightweight_beauty.yaml --smoke
python -m budget_sid train_benefit_predictor --config configs/lightweight_beauty.yaml --smoke
python -m budget_sid evaluate_budgets --config configs/lightweight_beauty.yaml --smoke
python -m budget_sid demo --config configs/lightweight_beauty.yaml --smoke --history 1,2,3
```

不带 `--smoke` 时不会假装已经跑完评测，也不会读取测试标签来训练。它返回退出码 1，并给出 pinned GRID 上的命令计划。

GPU 上的 GRID 命令从 `third_party/GRID` 目录执行，形状如下（`embedding_path`、`ckpt_path`、`semantic_id_path` 用上一步日志里的真实文件补上）：

```bash
python -m src.train experiment=rkmeans_train_flat \
  data_dir=data/amazon_data/beauty embedding_path=<minilm_cache> \
  embedding_dim=384 num_hierarchies=3 codebook_width=256 \
  trainer.strategy=auto trainer.devices=1 trainer.num_nodes=1 \
  trainer.sync_batchnorm=false trainer.accelerator=gpu

python -m src.inference experiment=rkmeans_inference_flat \
  data_dir=data/amazon_data/beauty embedding_path=<minilm_cache> \
  embedding_dim=384 num_hierarchies=3 codebook_width=256 \
  ckpt_path=<rkmeans_ckpt> \
  trainer.strategy=auto trainer.devices=1 trainer.accelerator=gpu

python -m src.train experiment=tiger_train_flat \
  data_dir=data/amazon_data/beauty semantic_id_path=<sid_tensor> \
  num_hierarchies=4 \
  model.huggingface_model.config.d_model=128 \
  model.huggingface_model.config.num_layers=4 \
  model.decoder.config.num_layers=4 \
  trainer.accumulate_grad_batches=4 \
  data_loading.train_dataloader_config.dataloader.batch_size_per_device=16 \
  trainer.strategy=auto trainer.devices=1 trainer.accelerator=gpu

python -m src.inference experiment=tiger_inference_flat \
  data_dir=data/amazon_data/beauty semantic_id_path=<sid_tensor> \
  ckpt_path=<tiger_ckpt> num_hierarchies=4 \
  model.top_k_for_generation=20 \
  trainer.strategy=auto trainer.devices=1 trainer.accelerator=gpu
```

收益预测、路由和评测是本仓库的代码，推荐模型先冻结，再训练预测器。

## TODO(gpu)：SeetaCloud RTX 4090

在 SeetaCloud 的单张 RTX 4090 上跑非 smoke 路径。4090 为 24GB；这套配置按单卡约 16GB 的预算来写（batch 16，梯度累积 4，不做 DDP）。

- [ ] `bash scripts/pin_grid.sh`，确认 HEAD 为 `2fe3475b2d369580234093f35d52b1a2f54d0472`
- [ ] 准备 `data/amazon_data/beauty/{training,evaluation,testing,items}`
- [ ] 冻结 MiniLM，缓存 384 维商品向量；不要加载 FLAN-T5-XL
- [ ] GRID RQ-KMeans（3 codebook）+ TIGER dedup digit，再训练 4+4 层、`d_model=128` 的生成模型
- [ ] 冻结该 checkpoint。验证集 A 训练收益预测器，B 上选平均 beam ≤ 40 的阈值
- [ ] 先跑止损。退出码 2 就停，不要再加路由
- [ ] 通过后，在测试集上比较 fixed 20/40/60、random、entropy、benefit-aware
- [ ] 三个种子，记录 Recall@20、NDCG@20、延迟均值 / P95、吞吐、峰值显存，以及长尾和短历史切片

## 引用

- GRID / Semantic IDs handbook: https://research.snap.com/publications/semantic-ids-in-generative-recommendation-a-practitioners-handbook.html
- GRID repo: https://github.com/snap-research/GRID
- TIGER: https://arxiv.org/abs/2305.05065
- GR4AD（beam / budget）: https://arxiv.org/html/2602.22732v2
- MiniLM: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Protocol (English)

1. Pin GRID at `2fe3475b2d369580234093f35d52b1a2f54d0472`. Do not vendor it.
2. Beauty lives at `data/amazon_data/beauty/{training,evaluation,testing,items}`. History must not contain the target interaction. The test split is locked.
3. Cache frozen `sentence-transformers/all-MiniLM-L6-v2` vectors (384-d). Reject FLAN-T5-XL.
4. RQ-KMeans, 3 codebooks of width 256, plus the TIGER dedup digit (`num_hierarchies=4` for the generator).
5. Train the lightweight transformer (`d_model=128`, 4 encoder and 4 decoder layers, batch 16, grad accumulation 4, one device) and freeze it.
6. Split validation users into disjoint A and B. Fit ΔNDCG@20 on A. Choose the threshold on B so the average beam width stays at or below 40.
7. Route by a shared first step, then continue once at beam 20 or 60. Do not finish the small beam and rerun the large one. Always return Top-20.
8. Compare fixed 20/40/60, random budget, entropy, and benefit-aware routing on one checkpoint. Report mean and std over seeds 42, 43, and 44.
9. Full-catalog Recall@20, NDCG@20, latency mean and P95, throughput, peak VRAM, long-tail items, and short-history users. No sampled negatives.
10. Stop-gate (exit 2) on validation: stop if the larger beam has no concentrated gain, or if oracle budget allocation does not beat uniform beam 40.

`TODO(gpu)`: run that protocol on a SeetaCloud RTX 4090, single device, no DDP.
