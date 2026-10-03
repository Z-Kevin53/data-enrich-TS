# TS-Aug 实验设计（Teacher-Student 集成增强 few-shot 数据集）

## 1. 问题形式化

给定每类仅 $k$ 张标注图像的支持集 $\mathcal{D}_0 = \{x_i, c_i\}$（$c_i \in \{1..C\}$，本项目 C=20），
目标是迭代扩充至每类 $S_{\text{target}}$ 张，使在扩充集上训练的下游分类器在固定验证集 $\mathcal{V}$ 上精度最大。

## 2. TS 流水线（算法 1）

**步骤 1 — Teacher 学习**：在小 CNN $T$ 上训练
$$\theta_T^{(0)} = \arg\min \; \mathbb{E}_{(x,c)\sim \mathcal{D}_{t-1}}\,\mathcal{L}_{\mathrm{CE}}\big(P_T(\cdot\mid x),\, c\big)$$

**步骤 2 — N 个 Student 生成**：每个 Student $S_i$ 是编码器-解码器（不同随机初始化 + 共享风格先验），
从源样本 $x$（类 $c$）与潜码 $z \sim \mathcal{N}(0,I)$ 生成候选
$$\hat{x}' = G_i\big(\mathrm{Enc}_i(x),\, z\big)$$
Student 训练目标（含类一致性自蒸馏）：
$$\mathcal{L}_S = \big\|\hat{x}' - \mathrm{Tr}(x)\big\|_1 + \mathcal{L}_{\mathrm{CE}}\big(P_i(\cdot\mid \hat{x}'),\, c\big) + \lambda\Big(1 - \cos\big(f_i(\hat{x}'), f_i(x)\big)\Big)$$
其中 $\mathrm{Tr}(\cdot)$ 是风格先验变换（jitter / affine warp / mixed），$f_i$ 为编码器特征。

**步骤 3 — 集成打分与筛选**：候选 $\hat{x}'$（由 Student $i$ 生成、目标类 $c$）的共识分数：
$$S(\hat{x}') = \frac{w_T\, P_T(c\mid \hat{x}') + w_S\, \bar{P}_{\mathrm{peer}}(c\mid \hat{x}')}{w_T + w_S}, \qquad
\bar{P}_{\mathrm{peer}} = \frac{1}{N-1}\sum_{j \neq i} P_j(c\mid \hat{x}')$$
$w_T > w_S$（Teacher 权重更高，$w_S \equiv 1$）；Student 的分类头既监督自己的生成物，也互评同伴。
接受集（每类配额 = 目标缺口）：
$$\mathcal{A}_t = \mathrm{TopK}\Big\{\hat{x}'_k : S(\hat{x}'_k) \ge \theta\Big\}$$

**步骤 4 — 迭代更新**：
$$\mathcal{D}_t = \mathcal{D}_{t-1} \cup \mathcal{A}_t, \qquad \theta_T^{(t)} \leftarrow \text{fine-tune}(T;\, \mathcal{D}_t)$$
重复直至 $|\mathcal{D}_t| \ge S_{\text{target}} \cdot C$ 或达到最大迭代数 $T_{\max}$。
（可选：每个迭代对 Student 做 1-2 epoch 刷新 —— `retrain_students`。）

## 3. 数据集与协议（固定、可复现）

- **数据**：CIFAR-100 训练集，前 20 个 fine 类（seed=42 固定切片）。
- **切分**（seed=42，对所有实验相同 → 验证集完全一致）：
  - 每类 50 张 → 验证集 $\mathcal{V}$（1000 张，从不参与增强）
  - 剩余中每类 $k$ 张（$k \in \{5,10\}$）→ 支持集 $\mathcal{D}_0$
- **目标规模**：$S_{\text{target}} \in \{40, 60\}$ 张/类（最终 ≤ 1200 张）。
- **评测**：ProtoNet 特征提取器（CNN → 128 维）在最终 $\mathcal{D}$ 上训练 4 epoch，
  原型 = 类内特征均值，预测 = 最大余弦相似度 → **`val_acc`（higher-is-better，AutoResearch 目标）**。
- **基线**：同一评测器仅在 $\mathcal{D}_0$（k-shot）上训练的结果（跨实验缓存，因为只依赖 $k$）。

## 4. 搜索空间（AutoResearch）

| 参数 | 含义 | 取值 |
|---|---|---|
| `shots` | 初始每类支持样本数 k | 5, 10 |
| `target_shots` | 目标每类规模 | 40, 60 |
| `n_students` | Student 数量 N | 2, 3, 4 |
| `teacher_weight` | Teacher/Student 打分权重比 w_T | 2.0, 3.0, 5.0 |
| `score_threshold` | 接受阈值 θ | 0.5, 0.6, 0.7 |
| `aug_iters` | 最大增强迭代数 | 3, 4, 6 |
| `candidates_per_source` | 每源样本每 Student 候选数 M | 4, 8 |
| `student_channels` | Student 基础通道 | 32, 64 |
| `latent_dim` | 潜码维度 | 8, 16 |
| `teacher_lr` / `student_lr` | 学习率 | 0.001, 0.003 |
| `student_style` | 风格先验 | jitter, warp, mixed |
| `retrain_students` | 每迭代刷新 Student | 0, 1 |

## 5. 预算与可复现性

- 单实验墙钟目标 < 90 s（GPU-0 RTX 2080 Ti；候选 ≤ ~2600 张/迭代，批量前向）。
- 每实验固定 seed=42（数据切片、训练、生成潜码）→ 可复现。
- 每实验落盘 `results/curves/<sha1(params)>.json`：逐迭代的
  `{n_total, n_candidates, n_accepted, mean_score_all, mean_score_accepted, teacher_val_acc}`，
  供 IEEE 论文绘图（数据增长曲线、teacher 精度曲线、分数分布）。

## 6. 基线与消融（论文用）

- **B0**：k-shot 无增强基线（评测器只训 $\mathcal{D}_0$）。
- **B1**：过采样基线（把 $\mathcal{D}_0$ 重复到 target 规模，无生成）—— 在最佳 TS 配置附近手工复算。
- **消融维度**：N（2/3/4）、w_T（2/3/5）、θ（0.5/0.6/0.7）、style（jitter/warp/mixed）分组均值。

## 7. 成功判据

1. 最佳 TS 配置的 `val_acc` 显著高于 B0（目标 Δ ≥ 2 个百分点）。
2. 数据增长/teacher 精度曲线单调合理（接受分数随迭代上升或稳定）。
3. 消融显示 w_T↑ 与 θ↑ 的权衡符合"精度-多样性"预期。
