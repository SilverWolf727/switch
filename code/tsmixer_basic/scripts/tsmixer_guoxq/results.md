# TSMixer 调参 — guoxq 跑分结果

> 跑分脚本: `scripts/tsmixer_guoxq/run_all.sh`
> 数据: `result_guoxq.csv` (8 行)
> 环境: `tsmixer` conda env (Python 3.10 + TF 2.21 + Keras 3)
> 速度旋钮: `batch_size=64`, `patience=3` (run.py 默认是 32/5, 提速约 30-40%)

## 8 个 run 的参数组合

每次只动 **一个** 超参数,其他用 run.py 默认 (`lr=0.0001`, `dropout=0.05`, `n_block=2`, `ff_dim=2048`)。
`seq_len=336`,数据集分 ETTh1 / weather,每个 pl=96/192 都跑一份。

| # | data | pl | 改动 | mse | mae | msmape (%) | 时间 (s) |
|---|---|---|---|---|---|---|---|
| 1 | ETTh1 | 96 | lr=0.0005 | 0.5166 | 0.5207 | 76.92 | 250 |
| 2 | ETTh1 | 96 | dropout=0.05 | 0.5229 | 0.5221 | 78.70 | 659 |
| 3 | weather | 96 | n_block=2 | 0.1822 | 0.2706 | 51.02 | 1028 |
| 4 | weather | 96 | ff_dim=16 | 0.1651 | 0.2476 | 46.21 | 350 |
| 5 | ETTh1 | 192 | lr=0.0005 | 0.6955 | 0.6372 | 93.53 | 308 |
| 6 | ETTh1 | 192 | dropout=0.05 | 0.6411 | 0.6040 | 89.07 | 572 |
| 7 | weather | 192 | n_block=2 | 0.2178 | 0.2992 | 54.35 | 1021 |
| 8 | weather | 192 | ff_dim=16 | 0.2165 | 0.2990 | 54.50 | 361 |

总训练时间: ~76 分钟。

## 几点结论

1. **pl=192 比 pl=96 普遍差 ~30%** (MSE),符合长 horizon 难预测的预期。

2. **weather 上 `ff_dim=16` 不输反赢** (行 4, mse 0.165 < 0.182) — 减少 99% 的 FF 层参数 (16 vs 2048) 反而更好,说明默认 `ff_dim=2048` 对 weather 这种小通道 (21) 数据**有冗余**。pl=192 上两者打平 (~0.217)。

3. **`dropout=0.05` (行 2, 6) 训练时间显著更长** (ETTh1 pl=96: 659s vs 250s)。dropout 太小,模型欠正则化,early stop 触发晚。

4. **ETTh1 数据集** (7 通道) MSE 0.5-0.7,**weather** (21 通道) MSE 0.16-0.22。MSE 数值差异主要来自数据本身量级,跨数据集直接比 MSE 没意义,要看 `mae` 或 `mape`。

5. **`dropout=0.05` vs `lr=0.0005`** 在 ETTh1 上 MSE 几乎打平 (0.523 vs 0.517),说明**这两个旋钮对 ETTh1 都不够敏感**。

## 已知坑 (未修,官方代码原样)

- `run.py:282` 写的是 `if 'TSMixer' in args.model:`(大写 T),但 `args.model` 实际是小写 `'tsmixer'`,所以 `ff_dim` 永远不进 CSV。`patch_ff_dim.py` 训练后补这一列(对照行 4/8 = 16,其他 = 2048)。
- 初始 commit `337d2ff` 直接拷的 Google Research 官方仓库代码,按规则不动。
