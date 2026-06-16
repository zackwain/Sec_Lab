# 基于时间自适应隐私预算分配的差分隐私联邦学习研究

> 损失动量驱动的隐私预算调度：用训练过程自身的 loss 反馈信号，自动决定每轮该花多少隐私预算。

- **仓库地址**：https://github.com/zackwain/Sec_Lab
- **作者**：冷睿 周巍
- **邮箱**：3342423804@qq.com

---

## 项目简介

在差分隐私联邦学习（DP-FL）中，每个客户端拥有固定总隐私预算 ε=6.0。如何将这有限的预算跨 20 轮训练合理分配？现有方法（如 Kiani et al. ICLR 2025 的线性递增调度）采用**预设的固定分配曲线**，一旦实际训练节奏与预设不符（如模型提前收敛），策略就会失效。

本项目提出**损失动量驱动的自适应 ε 调度方法**：根据训练过程中 loss 变化的一阶速度与二阶加速度，动态计算每轮应花多少预算。loss 加速下降 → 加注（减少噪声）；loss 趋于平缓 → 省钱（保留预算）。在不消耗额外隐私的前提下，实现比均匀分配和固定调度更高的模型准确率。

**核心实验结论**：在 ε=6.0 的高隐私约束下，OURS 达到 92.57% 准确率（vs Uniform 92.28%），同时总 ε 消耗更低（4.30 vs 4.36）——帕累托改进。相对于 Uniform 基线的提升量是 Kiani 方法的 3.2 倍。

---

## 环境与依赖

### 运行环境

| 项目 | 版本 | 说明 |
|------|------|------|
| 操作系统 | Windows 11 | 开发与测试环境 |
| Python | 3.13 | Conda 环境 `SecLab_env` |
| PyTorch | 2.12.0 CUDA 12.6 | 深度学习框架 |
| GPU | NVIDIA RTX 3060 (6GB) | 训练加速 |
| Opacus | 1.4.0 | 差分隐私引擎 |

> **必须设置环境变量**：`KMP_DUPLICATE_LIB_OK=TRUE`（Windows 下 PyTorch + OpenMP 冲突解决）

### Python 依赖

依赖清单见 `requirements.txt`，安装命令：

```bash
pip install -r requirements.txt
```

核心依赖：

| 依赖 | 版本 | 用途 |
|------|------|------|
| torch | ≥2.0.0 | 深度学习框架 |
| torchvision | ≥0.15.0 | MNIST 数据集加载 |
| opacus | ≥1.4.0 | DP-SGD + 隐私会计 |
| numpy | ≥1.24.0 | 参数序列化与聚合 |
| matplotlib | ≥3.7.0 | 实验结果可视化 |
| scikit-learn | ≥1.3.0 | MIA 攻击的 ROC-AUC 计算 |

---

## 配置说明

所有超参数集中在 `step4_time_adaptive/config.py` 中管理，改动一处全局生效：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `TOTAL_EPSILON` | 6.0 | 每个客户端总隐私预算 |
| `DELTA` | 1e-5 | DP δ 参数（违规容忍概率） |
| `MAX_GRAD_NORM` | 1.0 | 梯度裁剪阈值 C |
| `MAX_ROUNDS` | 20 | 最大通信轮数 |
| `EPOCHS_PER_ROUND` | 1 | 每轮本地训练 epoch 数 |
| `BATCH_SIZE` | 600 | 批次大小（q≈0.01） |
| `LR` | 0.01 | 学习率 |
| `GAMMA` | 2.0 | 损失动量指数响应强度 |
| `MOMENTUM_WINDOW` | 4 | 动量计算的回看窗口 |
| `SEED` | 42 | 基础随机种子 |

---

## 数据集

| 数据集 | 来源 | 大小 | 格式 | 说明 |
|--------|------|------|------|------|
| MNIST | torchvision 自动下载 | 60K train / 10K test | 28×28 灰度图 | 10 类手写数字识别 |

数据集由 `torchvision.datasets.MNIST` 自动下载至 `data/MNIST/`，首次运行时需联网。`data/` 目录已加入 `.gitignore`。

多客户端实验使用 Dirichlet 分布（α=0.5）模拟 Non-IID 数据划分，函数位于 `step2_baseline/data.py`。

---

## 快速开始

```bash
# 1. 激活环境
conda activate SecLab_env

# 2. 安装依赖
pip install -r requirements.txt

# 3. 设置环境变量
set KMP_DUPLICATE_LIB_OK=TRUE

# 4. 运行主实验（3 次重复，取均值±标准差，约 2-3h）
python -u step4_time_adaptive/experiment.py

# 5. 生成可视化图表（DPI≥300，Times New Roman）
python step5_analysis/plot_figures.py
```

---

## 项目结构

```
Sec_Lab/
├── step1_verify/                      # 环境验证
│   └── test_flower.py                # Flower API 连通性测试
│
├── step2_baseline/                    # 模型 + 数据 + 基线
│   ├── model.py                       # MNISTCNN 定义与参数序列化
│   ├── data.py                        # MNIST 加载 + Dirichlet 切分
│   ├── client.py / server.py          # Flower FL 组件
│   └── run_baseline.py               # 无 DP 基线实验
│
├── step3_adaptive/                    # 旧方案（废弃，仅保留存档）
│   ├── allocator.py                   # 质量评估 + ε 分配器
│   ├── run_adaptive.py               # 实验脚本
│   └── quality/                       # 数据质量评估模块
│
├── step4_time_adaptive/               # ★ 核心模块：时间自适应 DP-FL
│   ├── config.py                      # 全局超参数（单点配置）
│   ├── scheduler.py                   # 三种 ε 调度 + 损失动量 + ε→σ 转换
│   ├── trainer.py                     # DP-SGD 训练 + 评估 + 训练循环
│   ├── experiment.py                  # 主实验（3 repeats, mean±std）
│   └── run_ablation.py               # γ 消融 + 轮数消融
│
├── step5_analysis/                    # 分析模块
│   ├── run_mia.py                     # MIA 成员推理攻击验证
│   └── plot_figures.py               # 科研论文级可视化（6 张, DPI≥300）
│
├── results/
│   ├── logs/                          # 实验结果 JSON
│   ├── figures/                       # 可视化图表 PNG
│   └── models/                        # 训练好的模型参数（.pkl）
│
├── requirements.txt                   # Python 依赖
├── .gitignore                         # Git 忽略规则
└── README.md                          # 本文件
```

---

## 三种对比策略

| 策略 | 公式 | 特点 |
|------|------|------|
| **A. Uniform（基线）** | ε_r = ε_total / R = 0.30 | 固定均分，隐式假设所有轮次同等重要 |
| **B. KianiLinear** | ε_r ∝ 0.5→1.5 线性递增 | 预设"前期省钱、后期花"的固定节奏 |
| **C. Kiani + Momentum（OURS）** | ε_r = ε_kiani(r) × exp(γ×m_r) | **损失动量信号驱动，自适应训练节奏** |

---

## 核心实验结果

| 策略 | 准确率 | Σε 消耗 | vs Uniform 提升 |
|------|--------|---------|----------------|
| Uniform | 92.28% ± 0.09 | 4.36 | — |
| KianiLinear | 92.37% ± 0.30 | 4.33 | +0.09 pp |
| **Kiani + Momentum** | **92.57% ± 0.25** | **4.30** | **+0.29 pp (3.2×)** |

> 3 repeats (seed=42/142/242), mean ± std。提升比 IR = Δ_OURS / Δ_Kiani = 3.2×。

---

## 消融实验

γ 灵敏度测试（γ ∈ {1, 2, 3, 5, 10, 20}）：

所有 γ 值下 OURS 均不低于 Uniform 基线。在 MNIST 场景下 γ 对准确率的影响极小（差异 < 0.1pp），因为 loss 信号本身过于温和——这是 MNIST 数据集的固有局限，更丰富信号的效果需在复杂数据集上验证。

---

## 关键注意事项

1. 每轮训练**必须创建新模型**（`MNISTCNN()`），避免 Opacus hook 重复注册
2. `get_noise_multiplier` 是近似值（~30% 偏差），三策略误差一致，对比公平
