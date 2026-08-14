# DeconRNASeq synthetic validation dataset

这是一个完全合成、无患者信息的测试集，随机种子为 `20260814`。它有 5,000 个基因、8 种细胞类型和 48 个混合样本，不是只覆盖理想内部点的玩具矩阵。

## 文件

- `reference_signatures.csv`：基因 × 细胞类型参考表达矩阵。
- `mixtures_exact.csv`：严格按 `Y=A×Pᵀ` 生成的无噪声混合矩阵。
- `mixtures_noisy.csv`：加入异方差、乘性和低表达 dropout 噪声的混合矩阵。
- `true_proportions.csv`：样本 × 细胞类型真值，每行和为 1。
- `gene_metadata.csv`：marker、共享谱系 marker、housekeeping、low-expression、background 标签。
- `sample_metadata.csv`：纯样本、均匀混合、稀疏混合、0.5% 稀有细胞、相关谱系和随机混合标签。
- `DeconRNASeq_test_data.xlsx`：上述数据的便于人工浏览版本；程序测试应优先读 CSV。
- `manifest.json`：维度、种子和 SHA-256 清单。
- `validate_dataset.py`：只依赖 NumPy 的独立约束最小二乘校验器。
- `run_r_baseline.R`：原 R 包结果导出脚本，仅用于行为基线，不是 Python 交付实现。

## 已验证数值

- 最大比例行和误差：`1.0e-12`
- `mixtures_exact` 矩阵恒等式最大绝对误差：`3.693e-09`（来自 CSV 十进制序列化）
- 参考矩阵条件数：`3.226936`
- 按 R `scale` 规则标准化后的条件数：`2.314302`
- 独立 active-set 解算器在 exact 数据上的比例最大绝对误差：`4.921e-13`
- noisy 数据的总体比例 RMSE：`0.002497807`
- noisy 数据的比例最大绝对误差：`0.02300665`

重新验证：

```powershell
python validate_dataset.py
```

数学正确性的首要门槛应使用 `mixtures_exact.csv` 和 `use_scale=False`。`use_scale=True` 会改变数值问题，因此它应与原 R 输出比较，而不是直接要求等于生成真值。

