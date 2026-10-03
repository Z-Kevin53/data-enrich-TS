# 数据增强模型与评估方法前沿调研

> TS-Aug 项目调研文档（AutoResearch 自动生成框架下的研究前置工作）
> 日期：2026-10-03

## 第一部分：数据增强模型 / 网络架构前沿

### 1.1 规则式与搜索式增强（baseline 参照系）

| 方法 | 年份 | 思路 | 局限 |
|---|---|---|---|
| AutoAugment (Cubuk, NeurIPS 2019) | 2019 | RL 搜索 (操作,强度) 组合策略 | 需要大数据集搜索，策略不可跨域迁移 |
| RandAugment (Cubuk, ICLR 2020) | 2020 | N 个随机操作 + 单一强度超参 | 纯像素域变换，无类条件感知 |
| TrivialAugment (Muller, ICML 2021) | 2021 | 单次随机几何+颜色变换 | 同上 |
| Mixup / CutMix / MixStyle / AugMix | 2017–2020 | 样本混合 / 风格迁移 / 加权混合 | 依赖已有样本质量，不能真正"扩量" |

**结论**：规则式增强是下界参照。少样本（few-shot）场景下，变换操作本身容易把本就稀缺的样本推离流形，
需要**生成式 + 类条件**的增强器。

### 1.2 面向 few-shot 学习的生成式增强（本任务直接相关）

| 方法 | 生成器 | 关键机制 |
|---|---|---|
| FIDA (Kim, CVPR 2019) | 逐类 WGAN-GP | 对 1-shot 支持样本做 WGAN 采样，类别一致性由类条件保证 |
| FSSA (Zhang, ECCV 2020) | 类条件 VAE | 学习"语义变换"（同一性的保持变换），对抗+一致性损失 |
| FEAT (Zhang, CVPR 2020) | 特征适配网络 | 特征空间增强而非像素空间 |
| ReAug (Zhang, ICCV 2021) | 元学习特征重建 | 用元学习预测"重建+增强"的特征变换 |
| MetaAug (Kim, 2020) | 可微增强策略 | 用下游任务损失直接端到端优化增强算子 |
| D3DA (2023) | 扩散模型 | 条件去噪扩散生成 few-shot 样本，类条件嵌入 |
| FSDiff (2023) | 条件扩散 + 少样本先验 | 以支持集为条件的扩散先验，兼顾保真与多样 |

**趋势**：GAN → VAE/扩散；像素域 → 特征域；无监督 → **下游任务驱动的有选择增强**。
扩散模型生成质量最好，但采样慢（每样本数百步），不适合"大量候选 + 快速筛选"的在线流水线。

### 1.3 大模型 / 基础模型驱动的增强

- **CLIP (ICML 2021) / DINOv2 (TMLR 2024)**：冻结特征空间做"特征级增强"——在 CLIP 特征球面上
  对类中心做扰动/插值/球面反射，再用解码器回投像素域；类一致性由预训练语义空间天然保证。
- **文本到图像扩散（Stable Diffusion 系）**：类名作 prompt + 支持集作 IP-Adapter/ControlNet 参考，
  可生成高保真类条件样本；但需要大模型与算力，且存在概念泄漏风险（生成内容来自预训练语料）。
- **LLM 数据引擎（2023–2025）**：LLM 生成数据（文本/标注/指令），图像域较少见。
- **结论**：基础模型提供强先验，但本项目的研究点在于**选择机制**（谁生成的样本该被保留），
  因此采用轻量编码器-解码器 Student + 集成打分，把"生成质量评估"作为核心搜索对象。

### 1.4 Teacher-Student / 一致性学习（打分机制的理论基础）

| 方法 | 机制 | 对本项目的启示 |
|---|---|---|
| Mean Teacher (Tarvaine, NeurIPS 2017) | EMA teacher 给 student 伪标签 | teacher 分数应高于 student 分数（权重先验） |
| UDA (Xie, NeurIPS 2020) | 弱增强置信度过滤 + 强增强训练 | 置信度阈值 θ 过滤低质候选（本项目 θ 可搜索） |
| FixMatch (Sohn, NeurIPS 2020) | 半阈值一致性 | 阈值选择决定精度/召回权衡 |
| FlexMatch (Zhang, NeurIPS 2021) | 逐样本自适应阈值 | 阈值可按类自适应（本项目扩展方向） |
| ReMixMatch (Nguyen, ICML 2020) | 互信息最大化伪标签 | 集成多网络打分降低方差（本项目 peer 打分） |

**核心思想迁移**：无标注学习中"teacher 打分 + 阈值筛选"已被证明有效；本项目将其迁移到
**有少量标注的生成样本筛选**场景，并引入 student 互评（集成一致性）作为第二质量信号。

## 第二部分：如何评价一个数据增强模型

### 2.1 内在质量指标（不训练下游模型）

- **FID / KID** (Heusel, NeurIPS 2017)：生成分布与真实分布的距离（需 Inception 特征，CIFAR 域可用专用统计）。
- **类内多样性**：生成样本在特征空间的类内方差 / 与真实样本的最近邻距离分布（过大=离流形，过小=过拟合原样本）。
- **覆盖度**：各类样本数均衡度（基尼系数）；特征空间 t-SNE 覆盖。
- **一致性**：生成样本被（预训练）分类器识别为源类的比例。
- 本项目采用的**代理内在信号**（无需 Inception，快）：
  - teacher 在验证集上的准确率（teacher_val_acc）—— 数据集整体质量的单调代理；
  - 候选分数分布（mean_score_all / mean_score_accepted）—— 选择机制的工作点。

### 2.2 外在指标（金标准，本项目的 AutoResearch 目标函数）

- **下游 few-shot 精度**：在扩充后的数据集上训练分类器（本项目：ProtoNet），
  在固定留出验证集上测 accuracy（`val_acc`，higher-is-better）。
- **数据效率曲线**：accuracy vs 每类样本数（1/5/10/40/60 shot）—— 证明增强在"少"端收益最大。
- **增益**：Δ = val_acc(augmented) − val_acc(k-shot baseline)。
- **稳定性**：同一配置不同随机种子的 std（本项目固定种子保证可复现，作为扩展方向）。

### 2.3 效率与可操作性

- 每接受样本的生成开销（候选数/接受数）、单实验墙钟时间、GPU 显存。
- 选择工作点敏感性：θ 与 w_T 的响应曲面（消融实验）。

## 第三部分：few-shot 基准数据集

| 数据集 | 规模 | 图像 | 典型协议 | 备注 |
|---|---|---|---|---|
| miniImageNet (Vinyals 2016) | 100 类 | 84×84 | 5-way 1/5-shot | 最经典；下载 ~1GB，类间差异大 |
| tieredImageNet (Ren 2018) | 608 类 | 84×84 | 5-way 1/5/10/20-shot | 更大版本 |
| CIFAR-FS (Bertinetto 2019) | 100 类 (70 训/30 测) | 32×32 | 5-way 1/5-shot | 小图、快，**本项目选用其协议** |
| FC100 | 100 类 | 32×32 | 同上 | 更简单 |
| Omniglot (Lake 2015) | 50 字符×20 风格 | 28×28 | 20-way 1/5-shot | 风格域，最易 |
| CUB-200-2011 | 200 鸟类 | 可变 | 5-way 1/5-shot | 细粒度 |
| dSprites | 合成 | 64×64 | 因子控制 | 适合机制分析 |

**本项目选择**：CIFAR-100 前 20 个 fine 类，CIFAR-FS 风格协议——
每类从训练集固定切出 50 张验证图（所有实验共享，保证可比）+ k-shot 支持集（k∈{5,10}，可搜索）；
TS 流水线把支持集扩充到 target_shots∈{40,60}；最终用 ProtoNet 在验证集上评测。
选 CIFAR-100 而非 miniImageNet 的理由：32×32 小图使"生成+打分+重训"循环可在 1 分钟内完成，
保证 15+ 次自动搜索实验的可行性；协议上仍严格 few-shot（5/10-shot 起步）。

## 关键参考文献

1. Snell, Swersky, Zemel. *Prototypical Networks for Few-shot Learning.* NeurIPS 2017.
2. Vinyals et al. *Matching Networks for One Shot Learning.* NeurIPS 2016.
3. Kim, Zhang, Kira, Savarese. *FIDA: A Data Augmentation Framework using Inversion-based Generative Models.* CVPR 2019.
4. Zhang et al. *Few-Shot Semantic Data Augmentation (FSSA).* ECCV 2020.
5. Zhang, Zhang, Gao, Wang, Qiao. *FEAT: Few-Shot Learning via Feature Adaptation Networks.* CVPR 2020.
6. Zhang et al. *ReAug: Few-shot Learning by Adaptive Feature Re-Construction and Augmentation.* ICCV 2021.
7. D3DA. *Diffusion-based Data Augmentation for Few-Shot Learning.* 2023. arXiv:2303.15551.
8. FSDiff. *Few-Shot Diffusion-based Data Augmentation.* 2023.
9. Tarvainen & Valpola. *Mean Teacher.* NeurIPS 2017.
10. Sohn et al. *FixMatch: Simple Semi-Supervised Learning with Consistency Regularization.* NeurIPS 2020.
11. Zhang et al. *FlexMatch: Tightening the Semi-Supervised Learning Mind.* NeurIPS 2021.
12. Xie et al. *Unsupervised Data Augmentation (UDA) for Consistency Training.* NeurIPS 2020.
13. Cubuk et al. *AutoAugment.* NeurIPS 2019; *RandAugment.* ICLR 2020.
14. Hendrycks et al. *AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty.* ICLR 2020.
15. Yun et al. *CutMix.* ICCV 2019.
16. Heusel et al. *GANs Trained by a Two Time-Scale Update Rule (FID).* NeurIPS 2017.
17. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML 2021.
18. Caron et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024.
19. Nichol & Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* ICML 2021.
20. Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR 2022.
21. Finn, Abbeel, Levine. *Model-Agnostic Meta-Learning (MAML).* ICML 2017.
22. Oreshkin et al. *Convolutional Network for Few-Shot Learning (Relation Net).* CVPR 2018.
23. Lake et al. *Human-level one-shot learning on a 1505-class image taxonomy (Omniglot).* NeurIPS 2015.
24. Bertinetto et al. *CIFAR-FS.* 2019. arXiv:1907.11992.
25. Berg et al. *The CUB-200-2011 Dataset for Fine-grained Visual Recognition.* 2011.
