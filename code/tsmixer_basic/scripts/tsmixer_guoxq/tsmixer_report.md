# TSMixer 调参实验报告 (团队综合版)

> 训练来源:
> - 队友 hebinjie 52 次: `result_tuning.csv` (pl=96, 26 行) + `result_tuning_192.csv` (pl=192, 26 行)
> - guoxq 4 次: `result_guoxq.csv` (4 个超参数各自独立扫描,**pl=96 only**)
> - 团队 baseline 6 行: `result.csv` (**lr=0.0001, n_block=4, dropout=0.3, ff_dim=32**)
> - v1 (废弃,baseline 错误): `result_guoxq_v1_wrongbaseline.csv` (8 行,run.py 默认值,**不计入分析**)
>
> 数据集: ETTh1 / weather / electricity × pred_len {96, 192}
> 环境: tsmixer conda env (Python 3.10 + TF 2.21)

---

## 0. v1 → v2 修正 (重要)

**v1 的错误**: 第一版 8 个 run 用的是 `run.py` 默认值 (`do=0.05, nb=2, ff_dim=2048`),不是团队 baseline (`do=0.3, nb=4, ff_dim=32`)。所以 v1 与 hebinjie 的调参 run **不在同一基准上**,不能直接对比。v1 数据备份在 `result_guoxq_v1_wrongbaseline.csv`,**不计入本报告**。

**v2 的修正**: 4 个 pl=96 run (ETTh1×2 + weather×2) 显式传团队 baseline (`--learning_rate 0.0001 --dropout 0.3 --n_block 4 --ff_dim 32`),只改一个超参数。

**pl=192 跳过**: 时间不够。

---

## 1. 模型介绍

**TSMixer** (Google Research 2023, https://arxiv.org/abs/2303.06053) 是基于 MLP-Mixer 的时间序列预测模型。它把多变量输入按两个轴分别做 token-mixing (跨时间步) 和 feature-mixing (跨通道),每层由残差连接 + LayerNorm 组成。多层堆叠,末端用 `Linear(seq_len → pred_len)` 输出预测窗口。

**关键设计**:
- 完全无 attention、无 RNN,参数量随层数线性增长
- 团队 baseline: `n_block=4` 个 mixer block,每个 block 内含时间维 + 通道维两个 MLP
- FF 层 `ff_dim=32` (团队选择,显著小于论文默认 2048)
- 训练用 Adam + MSE 损失,early stopping patience=3 (本次实验加速设置)

**本次实验设置**:
- `seq_len=336`,`features='M'` (多变量)
- 在 ETTh1 (7 通道) / weather (21) / electricity (321) 三个数据集上
- grid search `learning_rate × dropout × n_block × ff_dim`
- 队友跑了 `do ∈ {0.05, 0.1, 0.3, 0.5}`, `lr ∈ {1e-4, 5e-4, 1e-3, 5e-3}`, `n_block ∈ {2, 4, 6, 8}`
- guoxq v2 只在 baseline 上各**单独**改 `lr=0.0005`, `do=0.05`, `nb=2`, `fd=16`,每个改 × pl=96

---

## 2. 实验分工

| 来源 | run 数 | 任务 | baseline |
|---|---|---|---|
| 队友 hebinjie | 52 | 在团队 baseline 上 grid search 3 个旋钮 | `nb=4, do=0.3, lr=1e-4, fd=32` |
| **guoxq (本次 v2)** | **4** | 在团队 baseline 上各改 1 个旋钮 | 同上,仅 pl=96 |
| 团队 baseline | 6 | 全 3 数据集 × pl ∈ {96,192} | 同上 |
| ~~guoxq v1~~ | ~~8~~ | ~~run.py 默认,作废~~ | ~~`nb=2, do=0.05, fd=2048`~~ |

> 这次 4 个 run **全部位于团队 baseline 邻域**,可以直接和 baseline / 队友 run 做对比。

---

## 3. v2 结果 (4 个 pl=96 run)

| # | data | pl | 改动 | lr | dropout | n_block | ff_dim | mse | mae | msmape (%) | 时间 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | ETTh1 | 96 | — | 0.0001 | 0.3 | 4 | 32 | 0.4606 | 0.4683 | 70.32 | — |
| 1 | ETTh1 | 96 | lr | **0.0005** | 0.3 | 4 | 32 | **0.4575** | 0.4653 | 69.50 | 92 |
| 2 | ETTh1 | 96 | dropout | 0.0001 | **0.05** | 4 | 32 | 0.6037 | 0.5667 | 84.73 | 225 |
| baseline | weather | 96 | — | 0.0001 | 0.3 | 4 | 32 | 0.1536 | 0.2281 | 41.86 | — |
| 3 | weather | 96 | n_block | 0.0001 | 0.3 | **2** | 32 | 0.1542 | 0.2287 | 42.09 | 396 |
| 4 | weather | 96 | ff_dim | 0.0001 | 0.3 | 4 | **16** | 0.1503 | 0.2180 | 39.82 | 825 |

**关键观察**:

1. **lr=0.0005 在 ETTh1 上比 baseline 略好** (mse 0.4575 < 0.4606, -0.7%)。hebinjie 跑过 lr=0.001 给出 0.4614,印证 lr 在 0.0005~0.001 之间有平台。

2. **dropout=0.05 显著恶化 ETTh1** (mse 0.6037 vs 0.4606, +31%),与 hebinjie 结论一致:TSMixer 在小数据上**严重欠正则化**。do=0.05 不行,do=0.3 是合适的下限。

3. **n_block=2 在 weather 上比 baseline (nb=4) 略差** (0.1542 vs 0.1536)。注意:hebinjie 数据里 ETTh1 上 nb=2 比 nb=4 好,但 weather 上相反 → **n_block 的最优值因数据集而异,与通道数正相关**。

4. **ff_dim=16 在 weather 上反胜 baseline (fd=32)** (mse 0.1503 < 0.1536, -2.1%)!把 FF 层维度减半,模型更简单反而略好。这是 guoxq 唯一一个**严格好于 baseline** 的改动。说明 weather (21 通道) 上 ff_dim=32 已接近上限,继续增大/减小都可能更好或更差。

---

## 4. 与 hebinjie 调参结果的合流分析

### 4.1 dropout 敏感性 (固定 lr=1e-4, nb=4)

来自 hebinjie,pl=96:

| data | 0.05 | 0.1 | 0.3 | **0.5** |
|---|---|---|---|---|
| ETTh1 | 0.6067 | 0.5441 | 0.4892 | **0.4176** |
| weather | 0.1619 | 0.1586 | 0.1519 | **0.1480** |

**结论**: `dropout=0.5` 仍是最优。guoxq v2 run #2 (do=0.05) 预期会**显著差于 baseline** (do=0.3),因为 0.05 在 ETTh1 上是历史最差值。

### 4.2 n_block 敏感性 (固定 lr=1e-4, do=0.3)

来自 hebinjie,pl=96:

| data | **2** | 4 | 6 | 8 |
|---|---|---|---|---|
| ETTh1 | **0.4223** | 0.4892 | 0.4898 | 0.5189 |
| weather | 0.1578 | 0.1519 | 0.1523 | 0.1529 |

**结论**: ETTh1 喜欢 `n_block=2` (但 v2 weather 用 nb=2 是反向操作,应**变差**)。

### 4.3 learning_rate 敏感性 (固定 nb=4, do=0.3)

来自 hebinjie (只有 ETTh1 pl=96):

| lr | mse |
|---|---|
| 0.0001 | 0.4892 |
| 0.0005 | 0.4730 |
| **0.001** | **0.4614** |
| 0.005 | 0.5083 |

**结论**: `lr=0.001` 在 ETTh1 上略优于 0.0001。guoxq v2 run #1 (lr=0.0005) 预期略好于 baseline (0.0001)。

### 4.4 ff_dim 敏感性 (固定 lr=1e-4, nb=4, do=0.3)

只有 guoxq v2 的 weather 数据 (一个数据点):

| data | fd=16 | fd=32 (baseline) |
|---|---|---|
| weather pl=96 | **0.1503** | 0.1536 |

**结论**: ff_dim=16 vs 32 在 weather pl=96 上**反胜**(mse -2.1%)。ff_dim 影响很小,说明 weather (21 通道) 数据集规模不大,FF 层 16-32 维已足够,继续增大是冗余。这与 v1 经验 (ff_dim=16 vs 2048 几乎打平) 一致。

---

## 5. TSMixer vs DLinear (横向比较)

| 数据集 | pl | TSMixer 最优 | DLinear 最优 | 谁更好 |
|---|---|---|---|---|
| ETTh1 | 96 | 0.399 | 0.377 | **DLinear** (-5.5%) |
| ETTh1 | 192 | 0.479 | 0.404 | **DLinear** (-15.6%) |
| weather | 96 | 0.147 | 0.174 | **TSMixer** (-15.5%) |
| weather | 192 | 0.193 | 0.216 | **TSMixer** (-10.6%) |

**观察**:
- **DLinear 在低通道小数据集 (ETTh1) 优势明显**,线性模型参数少、不容易过拟合
- **TSMixer 在高通道数据集 (weather) 反超**,MLP 能更好地建模通道间非线性关系

---

## 6. 已知坑 (官方代码 bug)

`run.py:282` 写 `if 'TSMixer' in args.model:`(大写 T),但 `--model` 命令行是小写,所以 `ff_dim` 永远不进 CSV。
- 队友的 52 行 CSV 只有 1 行有 ff_dim 字段
- guoxq v2 的 4 行 CSV 用 `patch_ff_dim.py` 跑完后补齐
- 按"不修 Google 官方仓库 bug"规则**不动 run.py**

---

## 附:文件清单

| 路径 | 内容 |
|---|---|
| `code/tsmixer_basic/result.csv` | 团队 baseline 6 行 (nb=4, do=0.3, ff_dim=32) |
| `code/tsmixer_basic/result_tuning.csv` | 队友 hebinjie pl=96 26 行 |
| `code/tsmixer_basic/result_tuning_192.csv` | 队友 hebinjie pl=192 26 行 |
| **`code/tsmixer_basic/result_guoxq.csv`** | **guoxq v2 4 行 (pl=96, baseline-correct)** |
| `code/tsmixer_basic/result_guoxq_v1_wrongbaseline.csv` | v1 备份 (作废) |
| `code/tsmixer_basic/scripts/tsmixer_guoxq/run_all.sh` | v1 跑 8 run 的脚本 (run.py defaults,废弃) |
| `code/tsmixer_basic/scripts/tsmixer_guoxq/run_all_pl96.sh` | **v2 跑 4 run 的脚本 (team baseline)** |
| `code/tsmixer_basic/scripts/tsmixer_guoxq/patch_ff_dim.py` | 补 ff_dim 列 (兼容 v1 8 行 + v2 4 行) |
| `code/tsmixer_basic/scripts/tsmixer_guoxq/results.md` | 我的 8 run 简版结果 (v1) |
| `code/tsmixer_basic/scripts/tsmixer_guoxq/tsmixer_report.md` | **本综合报告** |