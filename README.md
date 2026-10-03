# TS-Aug: Teacher-Student 集成数据增强用于 Few-Shot 学习

**研究问题**：每类只有 k 张标注图像（5/10-shot，极小样本）时，如何用一条
**Teacher-Student（T-S）集成增强流水线**把数据集迭代扩充到 40–60 张/类，
使下游 Prototypical Network 在固定验证集上的精度 `val_acc` 最大？

- **基准**：CIFAR-100 前 20 个 fine 类（CIFAR-FS 风格），seed=42 固定切片
- **指标**：`val_acc`（higher-is-better），1000 张共享验证集，所有实验一致
- **框架**：AutoResearch 自动研究循环（生成想法 → 实验 → 保留/丢弃 → 论文）
- **硬件**：服务器 GPU-0（RTX 2080 Ti），单实验 10–60 秒

```
目录
1. 研究背景
2. Teacher-Student 架构详解（核心）
3. 下游评测
4. 数据协议
5. 项目结构
6. AutoResearch 搜索空间
7. 快速开始
8. 当前结果
9. 可复现性
```

---

## 1. 研究背景

Few-shot 学习中支持集极小，直接训练下游分类器方差大、易过拟合。
TS-Aug 的思路（`docs/02_experimental_design.md` 算法 1）：

1. **Teacher**：在当前数据 D 上训练的分类器，充当"质量法官"；
2. **N 个 Student**：不同随机初始化的编码器-解码器生成器，从源样本扰动
   出候选图像，每个 Student 自带分类头、可对同伴的候选**互评**（peer vote）；
3. **加权集成打分 + 阈值筛选**：只保留 Teacher 与同伴都认可的候选；
4. **迭代**：D ← D ∪ 接受集，Teacher 在更大数据上 fine-tune，重复至目标规模。

多样性来自 N 个异构生成器 + 随机潜码 z；质量来自加权共识分数 S(x')。
整条流水线被嵌入 AutoResearch，由框架自动搜索 13 维超参空间并生成论文。

## 2. Teacher-Student 架构详解（核心）

### 2.1 总体数据流

```
                 ┌──────────────── 迭代 t（重复至 |D| ≥ target·C 或 T_max）────────────┐
                 │                                                                    │
 当前数据集 D ───┼─训练/fine-tune─▶  Teacher T (TeacherCNN)                           │
                 │                              │  P_T(c|x')                          │
                 │  每类 ≤4 张源样本             │                                     │
                 │ ┌────────────┐ ┌────────────┐ │ ┌────────────┐                      │
                 │ │ Student S1 │ │ Student S2 │ │ │ Student Sn │                      │
                 │ │ Enc + Dec  │ │ Enc + Dec  │ │ │ Enc + Dec  │                      │
                 │ └─────┬──────┘ └─────┬──────┘ │ └─────┬──────┘                      │
                 │       │ x'=G_i(Enc(x), z)     │       │                             │
                 │       ▼              ▼        │       ▼                             │
                 │   候选 x'（每源图 M 张/Student，批量前向）                           │
                 │                              │                                      │
                 │   集成打分: S(x') = (wT·P_T(c|x') + mean_{j≠i} P_j(c|x'))/(wT+1)   │
                 │                              │                                      │
                 │   筛选: 按 S 降序, S ≥ θ, 每类 TopK 补足至 target                   │
                 │                              │                                      │
                 │   D ← D ∪ A ──(回到循环顶部, Teacher 10-epoch fine-tune)───────────┘
                 │
 最终 D ──▶ ProtoNet 训练 4 epoch ──▶ 原型分类 ──▶ val_acc（目标指标）
```

三个角色（`tsaug/models.py`）：

| 角色 | 类 | 结构 | 职责 |
|---|---|---|---|
| Teacher | `TeacherCNN` | 3 conv 块 CNN（32→16→8→4，32/64/128 通道）+ 256 维 MLP 头 | 用类概率 P_T(c\|x') 给候选打分（权重 w_T 更高） |
| Student ×N | `StudentNet` | 编码器（→256 维）+ 潜码条件解码器 + 256 维分类头 | 从源图生成候选；分类头还互评同伴候选（peer vote） |
| 评测器 | `ProtoNet` | CNN 特征提取器（→128 维）+ 线性头 | 最终下游评测（原型 = 类内特征均值，余弦相似度） |

### 2.2 Teacher —— `TeacherCNN` + `train_teacher`

```python
# tsaug/models.py
class TeacherCNN(nn.Module):
    # 32x32 -> 16 -> 8 -> 4;  features (c3*4*4) -> 256 -> logits
    def __init__(self, n_classes, channels=(32, 64, 128), feat_dim=256, dropout=0.2):
```

- 初始：在 D_0（k-shot）上训 **40 epoch**（AdamW，CE，weight_decay 1e-4）；
- 每个增强迭代末：在当前 D 上 **fine-tune 10 epoch**（`run_ts` 循环内）；
- 它**只参与打分，不生成数据**——生成全部交给 Student。
  这样 Teacher 看到的"训练集"始终只含真实图像 + 已通过质量门的候选。

### 2.3 Student —— `StudentNet`（编码器-解码器 + 潜码 + 分类头）

```python
# tsaug/models.py
class StudentNet(nn.Module):
    def encode(self, x):                      # x -> f ∈ R^256
        return F.relu(self.enc_fc(self.encoder(x).flatten(1)))
    def generate(self, x, z):                 # (f, z) -> x' ∈ [0,1]^3x32x32
        v = self.dec_fc(torch.cat([self.encode(x), z], dim=1))
        v = v.view(b, -1, 4, 4)
        for layer in self.decoder:            # 3× ConvTranspose: 4→8→16→32
            v = layer(v)
        return v
    def classify(self, x):                    # x -> logits（peer 打分用）
        return self.cls(self.encode(x))
```

两条前向路径：

- **生成路径**：`x → Enc → f(256维) → 与潜码 z 拼接 → Linear → (c2,4,4) → 3×ConvTranspose → x'`。
  同一个源图配不同 z 会得到不同的候选 —— 多样性直接来自 `z ~ N(0, I)`。
- **打分路径**：`x' → Enc → cls 头 → logits`，用于对**其他 Student** 的候选投票。

训练目标（`train_student`，10 epoch 初始化 / 5 epoch 可选刷新）：

```
L_S = ‖x' − Tr(x)‖_1                        # 重建：贴近"风格先验"变换后的源图
    + CE( cls(x'), c )                      # 类一致性：生成物仍应被判为源图类别
    + λ·(1 − cos( f(x'), f(x) ))            # 特征自蒸馏：生成物特征对齐源图特征
```

（λ = `LAMBDA_FEAT` = 0.5；`f` 是编码器特征，`f(x)` 做了 detach。）

风格先验 Tr(x)（`transform_batch`）按 `student_style` 逐图随机：

- `warp`：仿射变换（角度 ±18°、缩放 0.88–1.12、平移 ±0.18）
- `jitter`：亮度 0.75–1.25 × 通道增益 0.8–1.2 × 偏移 ±0.08
- `mixed`：两者 50/50

N 个 Student 用**不同种子**初始化（`seed + 101·(i+1)`），得到异构的生成器集合。

### 2.4 集成打分 —— `ensemble_score`

对候选 x'（由 Student i 生成、目标类 c）：

```
S(x') = ( w_T · P_T(c|x') + mean_{j≠i} P_j(c|x') ) / ( w_T + 1 )
```

```python
# tsaug/pipeline.py
p_t  = F.softmax(teacher(G).float(), dim=1)[idx, cand_cls]
peer_sum = Σ_j  [j≠src_idx] · softmax(students[j].classify(G))[idx, cand_cls]
peer   = peer_sum / max(1, n_s - 1)
return (wT * p_t + peer) / (wT + 1.0)
```

要点：

- **w_T > w_S（w_S≡1）**：Teacher 权重更高，质量门槛由"见过真图"的模型主导；
- **peer 排除生成者自身**（`src_idx != j`）：Student 对自己的生成物有偏，
  只让"没生成它"的同伴投票，抑制自偏好；
- 全部批量前向、`no_grad`，单迭代候选量 ~2000–2600 张在 2080Ti 上毫秒级完成。

### 2.5 生成、筛选与迭代 —— `run_ts`（主循环）

每个迭代 t：

1. **生成**：对每个未满 target 的类 c，无放回抽 `min(4, |D_c|)` 张源图，
   每张源图在每个 Student 下生成 M 个候选（M = `candidates_per_source`），
   即每类候选数 ≈ `min(4,cur) × M × N`；
2. **打分**：`ensemble_score` 对所有候选一次性算出 S ∈ [0,1]；
3. **筛选**：按 S 降序扫描，遇 `S < θ` 截断；每类只取"目标缺口"
   `target − |D_c|` 张（防止某类膨胀、保持类间均衡）；
4. **更新**：`D ← D ∪ A`；Teacher fine-tune 10 epoch；
   若 `retrain_students=1` 则每个 Student 刷新 5 epoch；
5. 直到 `|D| ≥ target·C` 或达到 `aug_iters`。

循环结束后：最终 D 训练 ProtoNet → 验证集 `val_acc`；同时跑一份 k-shot
无增强基线（B0，结果缓存在 `results/baseline_cache.json`）用于对比。

## 3. 下游评测（`ProtoNet` + `proto_eval`）

- **训练**：`train_proto` 在最终 D 上用 CE 训 4 epoch（特征提取器 + 线性头）；
- **推理**：L2 归一化特征 → 每类原型 = 类内特征均值（再归一化）→
  预测 = 最大余弦相似度。`val_acc` 即 1000 张验证集上的精度。

选 ProtoNet 而非深度分类器，是因为它天然适配 few-shot 评测协议，
且对"扩充数据的质量"敏感、对"扩充数据的数量"不敏感（不会被简单过采样骗到）。

## 4. 数据协议（`tsaug/data.py`）

- CIFAR-100 训练集前 20 个 fine 类（20 类 × 500 张）；
- 每类 **50 张 → 验证集**（从不参与增强，从不参与任何训练）；
- 每类剩余中取 **k 张 → 支持集 D_0**（k = 5 或 10）；
- 切片用 `np.random.default_rng(42)`，**所有实验共享同一验证集**；
- 首次运行自动下载（torchvision，失败则直连官方 tarball）并缓存为
  `data/cifar100/cifar100_train.npz`（git-ignored）。


## 5. 项目结构

```
data_enrich/
├── tsaug/                      # ★ TS 增强核心包
│   ├── data.py                 # CIFAR-100 few-shot 切片 + 下载/缓存
│   ├── models.py               # TeacherCNN / StudentNet / ProtoNet
│   └── pipeline.py             # 训练循环、ensemble_score、run_ts 主循环
├── experiments/
│   ├── ts_aug_experiment.py    # AutoResearch 实验入口
│   │                           #   （参数解析 / GPU0 钉扎 / 曲线落盘 / val_acc 输出）
│   └── enrich_experiment.py    # （旧任务）Spambase 数据增强实验
├── autoresearch/               # AutoResearch 框架
│   ├── orchestrator.py         # 研究循环 + 搜索空间（_build_search_space）
│   ├── idea_generator.py       # 启发式想法生成（baseline/guided/exploration）
│   ├── evaluator.py            # 指标解析（末行 "val_acc: X.XXXXXX"）+ 比较
│   ├── tracker.py              # results.tsv 追踪 + 收敛统计
│   ├── git_manager.py          # 改进实验的 git commit（kept/分支历史）
│   ├── paper_generator.py      # Markdown 论文
│   ├── ieee_paper.py           # IEEE 格式论文（paper.tex/html + SVG 图）
│   ├── svg_gen.py              # 论文图表（架构/增长/精度/消融/收敛）
│   ├── program.py              # program_config.yaml 的配置加载
│   └── experiment.py           # 实验记录数据结构
├── docs/
│   ├── 01_research_survey.md   # 文献调研（DA/蒸馏/ProtoNet 背景）
│   └── 02_experimental_design.md # 实验设计（算法 1 完整公式）
├── program_config.yaml         # 目标/指标/搜索空间/约束/论文设置
├── run_autoresearch.py         # 本地运行入口
├── run_remote.py               # ★ 远程服务器运行入口（sync/run/pull）
├── test_run.py                 # 本地 simulate 模式冒烟测试
└── .gitignore                  # data/ results/ papers/ 等不入库
```

## 6. AutoResearch 搜索空间

`program_config.yaml` 定义（目标：`val_acc` 越大越好，单实验 ≤ 900 s）：

| 参数 | 含义 | 取值 |
|---|---|---|
| `shots` | 初始每类支持样本数 k | 5, 10 |
| `target_shots` | 目标每类规模 | 40, 60 |
| `n_students` | Student 数量 N | 2, 3, 4 |
| `teacher_weight` | Teacher 权重 w_T | 2.0, 3.0, 5.0 |
| `score_threshold` | 接受阈值 θ | 0.5, 0.6, 0.7 |
| `aug_iters` | 最大增强迭代数 | 3, 4, 6 |
| `candidates_per_source` | 每源图每 Student 候选数 M | 4, 8 |
| `student_channels` | Student 基础通道 | 32, 64 |
| `latent_dim` | 潜码维度 | 8, 16 |
| `teacher_lr` / `student_lr` | 学习率 | 0.001, 0.003 |
| `student_style` | 风格先验 | jitter, warp, mixed |
| `retrain_students` | 每迭代刷新 Student | 0, 1 |

## 7. 快速开始

### 远程服务器（推荐，GPU 0）

```powershell
python run_remote.py                        # sync + 跑满 50 个实验 + 拉回结果
python run_remote.py --max-experiments 4    # 小规模验证
python run_remote.py --fresh --max-experiments 10   # 先清空远端历史
python run_remote.py --sync-only            # 只推代码
python run_remote.py --pull-only            # 只拉结果/论文
```

- 远端目录 `/home/zwk/data_enrich`，Python 环境符号链接复用 `.venv`；
- GPU 钉扎为**物理 GPU 0**（`ts_aug_experiment.py` 用 nvidia-smi UUID 设置
  `CUDA_VISIBLE_DEVICES`，`AR_GPU` 环境变量可覆盖）；
- 远端任务以 `nohup` 脱离 SSH 会话运行 + 30 s 轮询，P2P 隧道断开不影响运行；
- 产出：`results/results.tsv`、`results/curves/*.json`、
  `papers/`（Markdown + IEEE `paper.tex`/`paper.html` + 5 张 SVG 图）。

### 本地单实验（CPU 或本地 GPU，调试用）

```powershell
python experiments\ts_aug_experiment.py --params "{\"shots\": 5, \"target_shots\": 40}"
```

stdout 最后一行固定为 `val_acc: X.XXXXXX`（框架 evaluator 解析该行）。
每实验另写 `results/curves/<sha1(params)>.json`（逐迭代统计，供论文绘图）。

## 8. 当前结果（2026-10-03，4 实验验证跑，GPU 0）

| 实验 | 配置要点 | val_acc | 状态 |
|---|---|---|---|
| baseline_experiment | 5-shot→40, N=2, θ=0.5, M=4 | 0.186 | KEPT |
| architecture_exploration | 10-shot→60, N=4, latent 16, retrain=1 | **0.279** | **BEST** |
| regularization_experiment | 同 baseline + dropout/weight_decay | 0.175 | discarded |
| hyperparameter_search | lr 0.003, mixed style | 0.143 | discarded |

相对 k-shot 无增强基线（0.186），最佳 TS 配置 **+9.3 个百分点**。
注意：本项目的 mini-CNN 流水线按"单实验 < 90 s"的快速搜索预算设计，
`val_acc` 绝对值与文献中强骨干网络的 CIFAR-FS SOTA（5-shot 20 类 ~50%+）
不在同一量级——目标是打通 **T-S 增强 + AutoResearch 自动搜索 + 论文生成**
的完整研究闭环，而非刷 SOTA。完整 50 实验的正式研究跑待执行。

## 9. 可复现性

- 全链路 `seed=42` 固定：数据切片、模型初始化、生成潜码、DataLoader shuffle；
- 每个实验的参数指纹 `sha1(params)[:12]` 命名曲线文件，避免覆盖；
- 基线 B0 只依赖 `shots/seed`，跨实验缓存（`results/baseline_cache.json`）；
- 保留的实验自动 `git commit`（分支 `autoresearch/experiment`），
  仓库历史即"哪些配置被证明有效"的审计轨迹。

