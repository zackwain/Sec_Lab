# Sec_Lab — DP-FL 自适应隐私预算分配 项目文档

## 一句话概述

> 联邦学习 + 差分隐私。核心创新：**三因素综合评分（KL散度 + 损失下降 + 梯度对齐）自适应分配隐私预算**，让总预算不变的情况下准确率更高、收敛更快。

---

## 目录结构

```
Sec_Lab/
├── requirements.txt
├── CLAUDE.md                       # 本文件
├── PROGRESS.md                     # 当前进度（对话交接用）
├── PLAN_V3.md                      # V3 修复计划
│
├── step1_verify/                   # 环境验证 ✓
│   └── test_flower.py
│
├── step2_baseline/                 # 基线 FL（无DP）✓
│   ├── model.py                   # CNN 定义
│   ├── data.py                    # 数据 + 客户端划分 + 标签提取
│   ├── client.py                  # Flower 客户端（备用）
│   ├── server.py                  # Flower 服务端（备用）
│   └── run_baseline.py           # 自实现 simulation → 96%
│
├── step3_adaptive/                 # 自适应 DP-FL（当前阶段）
│   ├── quality/
│   │   ├── __init__.py
│   │   ├── kl_divergence.py      # KL 散度计算
│   │   └── quality_estimator.py  # 多因素综合评分
│   ├── allocator.py               # ε 分配 + σ 校准 + 梯度对齐
│   └── run_adaptive.py           # 主脚本
│
├── step4_experiments/              # 正式实验（待做）
├── results/logs/                   # JSON 日志
└── analysis/                       # 可视化（待做）
```

---

## 核心创新：三因素综合评分

### 公式

```
Score_i = w1 × KL_norm_i + w2 × loss_score_i + w3 × align_score_i
```

### 因素 1：KL 散度（静态，训练前一次）

```
KL_i = Σ P_i(k) × log(P_i(k) / Q(k))    ← 客户端 vs 全局分布
KL_norm = KL_i / max(KL_all)            ← 归一化 [0,1]

含义：KL 高 → 数据分布与全局差异大 → 数据稀缺/不可替代 → 基础分高
     KL 低 → 数据分布接近全局 → 冗余/可替代 → 基础分低
```

信息来源：客户端上报标签计数（10 个整数，一次性）

### 因素 2：损失下降（动态，每轮）

```
raw_drop = (loss_before - loss_after) / loss_before
loss_score = raw_drop × min(1, raw_drop / 0.1)

              ┌ raw_drop ≥ 0.1  → ×1.0  满分
              │ raw_drop = 0.05 → ×0.5  学得慢，打五折
惩罚逻辑：     │ raw_drop = 0.02 → ×0.2  几乎不动，打两折
              └ raw_drop = 0.01 → ×0.1  学不动，归零
```

信息来源：客户端上报 loss_before / loss_after（每轮 fit 返回）

### 因素 3：梯度对齐（动态，每轮，KL 门控）

```
raw_align = cos(Δ_client, Δ_global)            ← [0,1]，cos 截断
align_score = raw_align × (1 - KL_norm)        ← KL 做裁判！

KL 高 → (1-KL) 小 → 对齐发言权小 → 不对齐不扣分（你特别，方向不同是应该的）
KL 低 → (1-KL) 大 → 对齐发言权大 → 不对齐要扣分（你普通，方向偏有问题）
```

信息来源：服务端自己算（参数更新本来就上传了）

### 综合得分

```
Score_i = w1 × KL_norm_i + w2 × loss_score_i + w3 × align_score_i
```

三项都在 [0,1]，w1+w2+w3=1。第 1 轮动态因素给默认值 0.5。

---

## 分配方式：指数缩放

```
               exp(Score_i / T)
ε_i = budget × ─────────────────
               Σ_j exp(Score_j / T)

T = 0.5（温度参数，T 越小差距越大）
```

指数缩放替代了旧的比例分配，把分数差距放大为筹码差距，让好客户端拿更多预算。

---

## 收敛检测与早停

### EMA 平滑

```
smooth_r = 0.3 × acc_r + 0.7 × smooth_{r-1}    (r ≥ 3)
```

### 两个停止条件（满足任一即停）

```
条件 1 — 收敛：
  最近 patience 轮 smooth 最大值 - 前 patience 轮 smooth 最大值 < 0.002
  (patience=5)

条件 2 — 过拟合：
  连续 2 轮 smooth < best_ever_smooth - 0.003
```

### 对比输出

```
                Uniform     Adaptive
准确率          93.6%       94.1%
收敛轮数          18          13        ← 更少
总隐私消耗      18×40       13×40      ← 更低
```

---

## 通信流程

```
训练前（一次性）：
  客户端 → 服务端: 标签计数 [n_0,...,n_9]

每轮：
  服务端 → 客户端: 全局参数 + 本轮 ε_i
  客户端: eval(loss_before) → DP-SGD(ε_i) → eval(loss_after)
  客户端 → 服务端: θ_i(加噪) + loss_before + loss_after + 样本数
  服务端: FedAvg → 算对齐+损失下降 → 综合得分 → 分配下轮 ε → 收敛检测
```

服务端从头到尾不碰原始数据，仅靠 FL 协议自带的元数据做决策。

---

## 核心模块

### 模型 (`step2_baseline/model.py`)

```
MNISTCNN: conv1(1→32)→conv2(32→64)→fc(64*7*7→128)→fc(128→10)
```

### KL 散度 (`quality/kl_divergence.py`)

- `compute_kl_divergence(client_counts, global_counts)` → float
- `compute_all_kl(label_counts_list)` → list[float]
- `counts_from_labels(labels)` → np.array

### 质量评估 (`quality/quality_estimator.py`)

- `compute_static_scores(counts_list)` → 归一化 KL [0,1]
- `compute_loss_drop(loss_before, loss_after)` → [0,1]
- `compute_multi_factor_score(kl, drop, align, weights)` → [0,1]
- `compute_all_scores(static, drops, aligns, weights)` → list[float]

### 分配器 (`allocator.py`)

- `allocate_uniform(n, budget)` → 均匀
- `allocate_adaptive(scores, budget)` → 比例分配（旧）
- `allocate_exponential(scores, budget, T)` → 指数分配（新）
- `calculate_noise_multiplier(ε, δ, rate, epochs)` → Opacus 校准 ε→σ
- `compute_gradient_alignment(client, before, after)` → cos [0,1]

### 主脚本 (`run_adaptive.py`)

- `train_one_client_dp()` — DP-SGD + loss_before/loss_after 上报
- `run_dynamic_adaptive()` — 动态评分 + 早停
- `run_uniform_baseline()` — 均匀基线 + 早停
- `main()` — Phase 1 网格搜索 → Phase 2 完整对比

---

## 实验

### 参数

| 参数 | Grid Search | Full |
|---|---|---|
| clients | 5 | 10 |
| rounds | 5 | 20（早停可能提前结束） |
| epochs | 2 | 3 |
| ε/轮 | 20 | 40 |
| 分布 | Dirichlet α=0.5 | Dirichlet α=0.5 |

### 对比基线

| | 数据 | DP | 结果 |
|---|---|---|---|
| 无DP FL | IID | 无 | 96.05% |
| 均匀 DP-FL | Dirichlet | 有 | 93.64% |
| 自适应 DP-FL (v3) | Dirichlet | 有 | 待跑 |

---

## 环境

- Windows 11, conda env `SecLab_env`
- Python 3.13, PyTorch 2.12.0 CPU, Opacus 1.4.0, Flower 1.7.0
- `KMP_DUPLICATE_LIB_OK=TRUE`
- 项目路径: `c:\Users\zackwain\Desktop\Sec_Lab`
