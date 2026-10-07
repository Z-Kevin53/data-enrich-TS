# 03 文本数据增强：TS-Aug 从 CIFAR-100 到 AG News（研究第 2 轮）

## 研究问题

第 1 轮证明了：在图像 few-shot 分类中，"Teacher 打分 + Student 生成 + 集成筛选"
的 TS 增强能把 5/10-shot 数据集增长到目标规模并提升下游 ProtoNet 精度。
本轮问题：**同样的框架能否迁移到文本数据？**

- 数据集：**AG News**（4 类短文本：World / Sports / Business / Sci/Tech，~120k 行）
- 协议：k-shot（5/10，每类 k 行支撑）→ 目标 30/50 shots/类；
  固定验证集每类 100 行（seed=42），所有实验共享同一验证集。
- 评测：ProtoText（原型均值 + 余弦），4 epoch 训练在最终 D 上，`val_acc` 越高越好。

## 架构迁移（图像 → 文本）

| 组件 | 图像版 (`tsaug/`) | 文本版 (`textaug/`) |
|---|---|---|
| 编码器 | 3 块 CNN（256 维特征） | Embedding + BiLSTM → 2h 维特征（pack 处理变长） |
| 生成器 | 潜码条件 ConvTranspose 解码器（像素重建） | 潜码条件 LSTM 自回归解码器，初始状态 = f(Enc(x)‖z)，共享嵌入表 |
| 重建项 | ‖x' − Tr(x)‖₁ | teacher-forced token CE(x' ‖ Tr(x)) |
| 风格先验 Tr(x) | warp / jitter（几何+像素扰动） | denoise（随机删 15–25% token 后重写）/ swap（~20% 局部词对交换）/ mixed |
| 类一致性 | CE(Head(x'), c) | 同（在**真实采样生成物**上） |
| 特征自蒸馏 | 1 − cos(f(x'), f(x)) | 同（λ = 0.5） |
| 集成打分 | S = [w_T·P_T + mean_{j≠i} P_j] / (w_T+1) | **公式完全相同**（排除生成者，去自偏置） |
| 选择/迭代 | 每类 top-K + 阈值 θ，达标停止 | 同（每类目标 shots，teacher 每轮微调 10 epoch） |
| 评测 | ProtoNet（128 维，余弦） | ProtoText（同协议） |

### 文本生成细节

- 解码器每步以"上一步采样 token"为输入，自回归展开 L 步（L = seq_len，48/64）。
- 训练：teacher-forced 展开（一次前向，带梯度）算重建项；类/特征项在
  **no-grad 贪心采样**的生成物上算，梯度只穿过 encoder/cls head——
  与图像版"在真实生成物上做类一致性"的设计一致，且避免 48 步反向的显存开销。
- 候选生成：每个源句按风格先验取输入端（denoise 时源句带删除掩码），
  每个 student 采样 M 个候选（z ~ N(0, I)），teacher 与 peer students 打分。

### 为什么 denoise 是合理的"文本风格先验"

图像 Tr(x) = 扰动(x) 让解码器学习"源的邻域"；文本没有低层噪声概念，
对应的语义邻域 = **不完整源的补全/改写**（denoise）与**局部词序变体**（swap）。
两者都保持类语义不变，为生成器提供了"围绕源句"的采样目标。

## 数据协议

- 词表：top-20k 词（来自完整训练集，等价于预训练分词器；仅**标签**被屏蔽）。
- 特殊 token：0=UNK, 1=PAD, 2=BOS。
- 切分：seed=42 固定，每类 100 行 → 验证集（400 行，永不参与增强/训练）；
  其余 ~116k 行为训练池（few-shot 只用其中每类 k 行）。
- 缓存：`data/ag_news/ag_news_cache.npz`（git 忽略）；
  下载源按序回退：mlg.ucd.edu.cn fasttext.zip → danielpnash CSV → hf-mirror CSV。

## 搜索空间（AutoResearch，`program_config.yaml` task=text）

shots, target_shots(30/50), n_students(2/3), teacher_weight(2/3/5),
score_threshold(0.5/0.6/0.7), aug_iters(2/3/4), candidates_per_source(2/4),
seq_len(48/64), embed_dim(64/128), student_hidden(64/128), latent_dim(8/16),
teacher_lr, student_lr, text_style(denoise/swap/mixed), retrain_students(0/1)

## 运行

```bash
# 远程 GPU 服务器（GPU 0）
python run_remote.py --fresh --max-experiments 4   # 验证轮
python run_remote.py --max-experiments 50          # 全量
```

- 结果文件独立：`results/results_text.tsv`（不与图像 run 混在同一 ratchet 中）
- 曲线：`results/curves_text/<sha>.json`；基线缓存 `results/baseline_cache_text.json`
- 论文：markdown 格式（IEEE 模板为图像专用，文本版待后续扩展）
- 切回图像任务：`program_config.yaml` 里 `task: image` + 原 experiment_file 即可
