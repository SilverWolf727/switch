# TSMixer 调参实验报告 (团队综合版)

> 训练来源:
> - 队友 hebinjie 52 次: `result_tuning.csv` (pl=96, 26 行) + `result_tuning_192.csv` (pl=192, 26 行)
> - guoxq 8 次: `result_guoxq.csv` (4 个超参数各自独立扫描)
> - baseline 6 行: `result.csv` (**lr=0.0001, n_block=4, dropout=0.3, ff_dim=32**,团队统一基线)
>
> 数据集: ETTh1 / weather / electricity × pred_len {96, 192}
> 环境: tsmixer conda env (Python 3.10 + TF 2.21)

---

## 1. 模型介绍

**TSMixer** (Google Research 2023, https://arxiv.org/abs/2303.06053) 是基于 MLP-Mixer 的时间序列预测模型。它把多变量输入按两个轴分别做 token-mixing (跨时间步) 和 feature-mixing (跨通道),每层由残差连接 + LayerNorm 组成。多层堆叠,末端用 `Linear(seq_len → pred_len)` 输出预测窗口。

**关键设计**:
- 完全无 attention、无 RNN,参数量随层数线性增长
- 默认 `n_block=4` 个 mixer block,每个 block 内含时间维 + 通道维两个 MLP
- FF 层默认 `ff_dim=2048` (论文原值)
- 训练用 Adam + MSE 损失,early stopping patience=5 (c1ef046 改 5,本次部分 run 用 3)

**本次实验设置**:
- `seq_len=336`
- `features='M'` (多变量)
- 在 ETTh1 (7 通道) / weather (21) / electricity (321) 三个数据集上,grid search `learning_rate × dropout × n_block × ff_dim`

---

## 2. 实验分工与 baseline 差异 ⚠️

| 来源 | run 数 | 任务 |
|---|---|---|
| 队友 hebinjie | 52 | 在团队 baseline (nb=4, do=0.3, lr=0.0001, ff_dim=32) 基础上做 grid search |
| **guoxq (本次)** | **8** | **没跑 baseline**,只在 run.py 默认值 (nb=2, do=0.05, lr=0.0001, ff_dim=2048) 上各动一个超参数,各跑 pl=96/192 |
| 团队 baseline | 6 | lr=0.0001, n_block=4, dropout=0.3, ff_dim=32 (commit `c1ef046`) |

⚠️ 我和队友**用了不同的 baseline**(他用团队基线 nb=4/do=0.3/ff_dim=32;我严格用 run.py 默认 nb=2/do=0.05/ff_dim=2048),所以**两组 run 不能直接做网格平均**。下方分析只对**同一 baseline 内**的 run 做对比。

> 历史原因:队友的 baseline 是 `c1ef046` 提交时定的团队基线;我的 8 个 run 是按组员分配的参数严格使用 run.py 默认值(没改动 baseline),所以两组 run 覆盖了不同的参数区域。

---

## 3. 最优参数 (按 MSE 最小,组内比较)

| 数据集 | pl | baseline MSE | 最优 MSE | 改善 | 最优超参数 |
|---|---|---|---|---|---|
| ETTh1 | 96 | 0.4606 | **0.3990** | -13.4% | n_block=2, dropout=0.5 |
| ETTh1 | 192 | 0.5662 | **0.4785** | -15.5% | n_block=2, dropout=0.5 |
| electricity | 96 | 0.1505 | 0.1505 | 0% | baseline 已经最优 |
| electricity | 192 | 0.1773 | 0.1773 | 0% | baseline 已经最优 |
| weather | 96 | 0.1536 | **0.1469** | -4.4% | n_block=6, dropout=0.5 |
| weather | 192 | 0.1949 | **0.1932** | -1.0% | n_block=6, dropout=0.5 |

**关键观察**:
- **dropout 普遍偏大更好** (0.5 > 0.3 > 0.1 > 0.05),TSMixer 在 7~321 通道数据上都欠正则化
- **小数据集 (ETTh1) 喜欢浅模型** (n_block=2);**大数据集 (weather/electricity) 喜欢深一些** (n_block=4-6)
- **electricity 的 321 通道**是个例外:怎么调都没用,baseline 已经最优

---

## 4. 超参数敏感性

### 4.1 dropout (固定 lr=0.0001, n_block=4)

| 数据集 | pl | 0.05 | 0.1 | 0.3 | **0.5** |
|---|---|---|---|---|---|
| ETTh1 | 96 | 0.6067 | 0.5441 | 0.4892 | **0.4176** |
| weather | 96 | 0.1619 | 0.1586 | 0.1519 | **0.1480** |
| ETTh1 | 192 | 0.7066 | 0.6473 | 0.6118 | **0.5142** |
| weather | 192 | 0.2160 | 0.2057 | 0.1991 | **0.1945** |

**结论**: dropout 对 TSMixer 非常关键。`dropout=0.05` (run.py 默认) 在所有 4 组都是**最差的**,模型严重过拟合。`dropout=0.5` 始终最佳,可能还能更高(没扫到 ≥0.7)。

### 4.2 n_block (固定 lr=0.0001, dropout=0.3)

| 数据集 | pl | 2 | **4** | 6 | 8 |
|---|---|---|---|---|---|
| ETTh1 | 96 | 0.4223 | 0.4892 | 0.4898 | 0.5189 |
| weather | 96 | 0.1578 | 0.1519 | 0.1523 | 0.1529 |
| ETTh1 | 192 | 0.4992 | 0.6118 | 0.6070 | 0.6760 |
| weather | 192 | 0.2021 | 0.1991 | 0.1978 | 0.1991 |

**结论**: 
- **ETTh1 (7 通道) 上 n_block=2 最优**,n_block 越深越差 → 数据少,深度模型学不动
- **weather (21 通道) 上 n_block=4-6 最优**,更深的 8 边际递减 → 数据量适中,深度略有帮助
- n_block=4 是合理的通用默认值

### 4.3 learning_rate (固定 n_block=4, dropout=0.3)

只跑了 ETTh1 pl=96:

| lr | MSE |
|---|---|
| 0.0001 | 0.4892 (4 次均值) |
| 0.0005 | 0.4730 |
| **0.001** | **0.4614** |
| 0.005 | 0.5083 |

**结论**: `lr=0.001` 在 ETTh1 上略优于默认值 0.0001 (-5.7%),但优势不大。0.005 明显过拟合(loss 不收敛)。

### 4.4 ff_dim (guoxq 实验)

只跑了 weather × {96, 192},在 **guoxq baseline** (n_block=2, dropout=0.05) 下与团队 baseline 比较:

| 数据集 | pl | ff_dim=16 (guoxq) | ff_dim=2048 (guoxq 默认) | 团队 baseline (ff_dim=32) |
|---|---|---|---|---|
| weather | 96 | 0.1651 | 0.1822 | **0.1536** |
| weather | 192 | 0.2165 | 0.2178 | **0.1949** |

**结论**:
- 在 guoxq baseline 下 **ff_dim=16 vs 2048 几乎打平** (pl=96 略胜,pl=192 略差),说明默认 `ff_dim=2048` 对 weather 这种**小通道 (21)** 数据**严重冗余**。把 99% 参数砍掉反而略好,印证模型表达能力对这种小数据集是过剩的。
- 但**团队 baseline (ff_dim=32 + nb=4 + do=0.3)** 仍优于 guoxq 任何 ff_dim 选择,说明 **n_block 和 dropout 比 ff_dim 更重要**。

---

## 5. 实验思考

### 5.1 TSMixer vs DLinear (本课程的两个模型)

直接比 MSE 不公平(数据集特性不同),但可粗略比较**最优 MSE 量级**:

| 数据集 | pl | TSMixer 最优 | DLinear 最优 | 谁更好 |
|---|---|---|---|---|
| ETTh1 | 96 | 0.399 | 0.377 | **DLinear** (-5.5%) |
| ETTh1 | 192 | 0.479 | 0.404 | **DLinear** (-15.6%) |
| weather | 96 | 0.147 | 0.174 | **TSMixer** (-15.5%) |
| weather | 192 | 0.193 | 0.216 | **TSMixer** (-10.6%) |

**观察**:
- **DLinear 在低通道小数据集 (ETTh1) 优势明显**,线性模型参数少、不容易过拟合,正合适
- **TSMixer 在高通道数据集 (weather) 反超**,MLP 能更好地建模通道间非线性关系
- 印证两个模型在不同场景下的取舍:**简单场景用线性 (DLinear),复杂场景用 MLP (TSMixer)**

### 5.2 TSMixer 的训练特性

1. **极依赖 dropout**:`dropout=0.05` 时 ETTh1 pl=96 mse=0.607,而 `dropout=0.5` 时 0.418,差 30%。TSMixer 的 Mixer Block 参数量大,必须有强正则化。

2. **n_block 与数据量强相关**:7 通道 ETTh1 偏好 n_block=2,21 通道 weather 偏好 n_block=6。这与"层数 ≈ 数据复杂度"的直觉一致。

3. **lr 不敏感**:`lr=0.0001 ~ 0.001` 都能收敛,差异 < 10%,主要差别在 early stopping 触发时机。

### 5.3 MSMAPE 在 TSMixer 上的表现

- ETTh1 上 MSMAPE 60-95%,MSE 0.4-0.7 范围
- weather 上 MSMAPE 40-55%,MSE 0.15-0.22
- electricity 上 MSMAPE 39-43% (最低,因为 electricity 是用电量,值域大且相对误差天然小)
- 与 DLinear 结论一致:**MSMAPE 始终稳定在 0-100% 范围,而 MAPE 经常 > 100%**,MSMAPE 抑制了真值近零时的发散。

---

## 6. 已知坑 (官方代码 bug)

`run.py:282` 写 `if 'TSMixer' in args.model:`(大写 T),但 `--model` 命令行是小写,所以 `ff_dim` 永远不进 CSV。
- 队友的 52 行 CSV 只有 1 行有 ff_dim 字段
- 我的 8 行 CSV 用 `patch_ff_dim.py` 跑完后补齐

按"不修 Google 官方仓库 bug"规则**不动 run.py**,workaround 详见 `patch_ff_dim.py`。

---

## 附:文件清单

| 路径 | 内容 |
|---|---|
| `code/tsmixer_basic/result.csv` | 团队 baseline 6 行 (nb=4, do=0.3, ff_dim=32) |
| `code/tsmixer_basic/result_tuning.csv` | 队友 hebinjie pl=96 26 行 |
| `code/tsmixer_basic/result_tuning_192.csv` | 队友 hebinjie pl=192 26 行 |
| `code/tsmixer_basic/result_guoxq.csv` | guoxq 8 行 (lr/do/nb/ff_dim 各 2) |
| `code/tsmixer_basic/scripts/tsmixer_guoxq/run_all.sh` | 我跑 8 run 的脚本 |
| `code/tsmixer_basic/scripts/tsmixer_guoxq/patch_ff_dim.py` | 补 ff_dim 列 |
| `code/tsmixer_basic/scripts/tsmixer_guoxq/results.md` | 我的 8 run 简版结果 |
| `code/tsmixer_basic/scripts/tsmixer_guoxq/tsmixer_report.md` | **本综合报告** |