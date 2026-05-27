# 基于差分隐私保护的联邦学习技术研究 — 项目设计文档

## 基本信息

| 项目 | 内容 |
|------|------|
| 选题 | 基于差分隐私保护的联邦学习技术研究 |
| 类别 | 算法研发类 → 隐私保护类 |
| 成员 | 周巍(162320320)、冷睿(162320318) |
| 技术栈 | Flower + PyTorch + Opacus |
| Python 环境 | Conda env1 (`D:\LeStore\anaconda3_Program\envs\env1`) |
| 版本管理 | Git（每完成一个核心模块提交一次） |
| 交付物 | 代码 + 实验报告 + 最终答辩 PPT + 系统演示 |
| 周期 | 1-2 周 |

---

## 一、项目目录结构

```
Sec_Lab/
├── dp_fl/                          # 核心代码包
│   ├── __init__.py
│   ├── models.py                   # CNN 模型定义（MNIST / CIFAR-10）
│   ├── client.py                   # Flower 客户端：本地训练 + Opacus DP-SGD
│   ├── server.py                   # Flower 服务端：FedAvg + DP 噪声策略
│   ├── dataset.py                  # 数据加载、划分、Non-IID 分布模拟
│   ├── experiment.py               # 实验编排器：参数化运行 + 结果收集
│   ├── attacks.py                  # MIA 成员推断攻击 + 梯度反演攻击
│   └── metrics.py                  # 指标计算：Accuracy、MIA AUC、SSIM、PSNR
│
├── experiments/                    # 实验脚本（每个实验一个入口文件）
│   ├── exp1_baseline.py            # DP 开/关基线对比
│   ├── exp2_epsilon_sensitivity.py # ε 敏感度分析
│   ├── exp3_dataset_comparison.py  # MNIST vs CIFAR-10
│   ├── exp4_non_iid.py             # Non-IID 鲁棒性
│   └── exp5_clip_threshold.py      # 梯度裁剪阈值 C
│
├── results/                        # 实验结果输出（自动生成）
│   ├── figures/                    # 图表 .png
│   ├── tables/                     # 表格 .csv
│   └── logs/                       # 训练日志 .json
│
├── analysis/                       # 结果分析与可视化
│   └── plot.py                     # 统一绘图函数
│
├── report/                         # 实验报告
├── presentation/                   # 最终答辩 PPT
├── requirements.txt                # Python 依赖
├── PROJECT_DESIGN.md               # 本设计文档
├── 信息安全综合实验选题和要求26.pdf
└── 基于差分隐私保护的联邦学习技术研究.pptx  # 初期答辩 PPT
```

---

## 二、系统架构

```
┌─────────────────────────────────────────────┐
│                 Flower Server                │
│  ┌──────────────┐  ┌────────────────────┐   │
│  │ FedAvg 聚合   │  │ DP 噪声聚合策略     │   │
│  └──────────────┘  └────────────────────┘   │
└─────────────────────┬───────────────────────┘
                      │ 梯度参数通信
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Client 1    │ │  Client 2    │ │  Client N    │
│  (本地数据)   │ │  (本地数据)   │ │  (本地数据)   │
│              │ │              │ │              │
│ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │
│ │PyTorch   │ │ │ │PyTorch   │ │ │ │PyTorch   │ │
│ │Model     │ │ │ │Model     │ │ │ │Model     │ │
│ └──────────┘ │ │ └──────────┘ │ │ └──────────┘ │
│ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │
│ │Opacus DP │ │ │ │Opacus DP │ │ │ │Opacus DP │ │
│ │梯度裁剪   │ │ │ │梯度裁剪   │ │ │ │梯度裁剪   │ │
│ │+高斯噪声  │ │ │ │+高斯噪声  │ │ │ │+高斯噪声  │ │
│ └──────────┘ │ │ └──────────┘ │ │ └──────────┘ │
└──────────────┘ └──────────────┘ └──────────────┘
```

### 数据流

```
实验配置 → 数据加载与划分 → 创建 Server + Clients
    → 逐轮训练（广播 → 本地DP-SGD训练 → 聚合）
    → 每轮记录 (ε, accuracy, loss, round)
    → 训练完成 → 跑 MIA 攻击 + 梯度反演攻击
    → 生成图表 → 撰写报告
```

---

## 三、核心模块详细设计

### 1. models.py — 模型定义

- MNIST 模型：CNN (32→64→128, 2层卷积+池化, 2层全连接)，约 1.2M 参数
- CIFAR-10 模型：CNN (32→64→128→256, 3层卷积+池化, 2层全连接)，约 2.4M 参数
- 统一接口：`create_model(dataset_name: str) -> nn.Module`

### 2. dataset.py — 数据管理

- `load_mnist()` / `load_cifar10()` — 下载 + 标准化
- `split_iid(num_clients)` — IID 均匀随机分配
- `split_non_iid(num_clients, dirichlet_alpha)` — Dirichlet 分布模拟标签倾斜
  - α=0.1：极度不均衡（每个客户端只含1-2类）
  - α=0.5：中度不均衡
  - α=1.0：轻微不均衡

### 3. client.py — 客户端

```
FlowerClient(NumPyClient):
    __init__(cid, model, train_loader, privacy_config)
        → 将 PyTorch 模型用 Opacus PrivacyEngine 包裹

    fit(parameters, config):
        1. 接收服务端全局模型参数
        2. Opacus PrivacyEngine 自动做：
           → 梯度裁剪 (max_grad_norm = C)
           → 添加高斯噪声 (noise_multiplier = σ)
           → RDP 隐私会计追踪
        3. DP-SGD 训练（E 个本地 epoch）
        4. 返回更新后的参数 + 隐私统计

    get_privacy_spent() → (ε, δ)
        返回当前已消耗的隐私预算
```

### 4. server.py — 服务端

- 初始化全局模型，广播给所有客户端
- 每轮随机选取 K 个客户端参与训练
- 收集客户端模型更新，FedAvg 加权聚合（权重 = 本地样本数）
- 监控全局 ε、准确率、loss
- 达到目标准确率或最大轮次后停止

### 5. attacks.py — 隐私攻击（亮点模块）

#### 成员推断攻击 (Membership Inference Attack, MIA)
- 训练影子模型，学习"成员/非成员"分类边界
- 输出：MIA AUC、MIA Accuracy、TPR@低FPR

#### 梯度反演攻击 (Gradient Inversion Attack)
- 从共享梯度反向优化重建原始输入
- 输出：PSNR（峰值信噪比）、SSIM（结构相似度）、MSE（均方误差）

### 6. experiment.py — 实验编排器

统一配置接口，参数包括：
- `dataset`: "mnist" | "cifar10"
- `num_clients`: 客户端数量
- `num_rounds`: 联邦训练轮数
- `local_epochs`: 每轮本地训练轮数
- `epsilon`: 目标隐私预算
- `delta`: 失效概率（固定 1e-5）
- `max_grad_norm`: 梯度裁剪阈值 C
- `data_distribution`: "iid" | "dirichlet_0.5" | "dirichlet_0.1"
- `batch_size`, `lr`

---

## 四、实验结果指标体系

### 1. 隐私保护维度

| 指标 | 说明 | 展示方式 |
|------|------|----------|
| ε（隐私预算） | 越小越安全，取 0.5, 1, 2, 4, 8, 10 | 折线图 |
| δ（失效概率） | 固定 1e-5，保证 << 1/N | 参数说明 |
| RDP α 阶矩累积 | Opacus 底层 Rényi DP → (ε, δ) 转换 | 技术细节 |

### 2. 模型效用维度

| 指标 | 说明 | 展示方式 |
|------|------|----------|
| Top-1 准确率 | 核心指标 | 折线图（x=轮次, y=准确率） |
| 收敛轮次 | 达到目标准确率所需轮数 | 柱状对比图 |
| Loss 下降曲线 | 训练损失随轮次变化 | 双 Y 轴图 |

### 3. 隐私-效用权衡（核心亮点）

| 指标 | 说明 | 展示方式 |
|------|------|----------|
| ε-Accuracy 曲线 | 不同 ε 下的准确率 | 经典权衡图 |
| 隐私代价比 | (无DP - 有DP) / 无DP | 百分比表格 |
| Pareto 前沿 | ε=2 附近是否为 sweet spot | 标注最佳工作点 |

### 4. 系统鲁棒性维度

| 指标 | 说明 | 展示方式 |
|------|------|----------|
| Non-IID 衰减率 | IID vs Dirichlet(0.5) vs Dirichlet(0.1) | 分组柱状图 |
| 客户端数量影响 | N=5, 10, 20 | 折线图 |
| 梯度裁剪阈值影响 | C=0.1, 0.5, 1.0, 5.0, 10.0 | 折线图 |

### 5. 攻击抵抗维度（隐私保护实证验证）

| 指标 | 含义 | 理想值 |
|------|------|--------|
| MIA AUC | 攻击者区分训练/非训练样本的能力 | ≈ 0.50（随机猜测） |
| MIA Accuracy | 攻击准确率 | ≈ 50% |
| TPR@低FPR | 低虚警率下的真阳性率 | 越低越好 |
| 梯度反演 PSNR | 重建图像质量 | < 15dB（无法辨认） |
| 梯度反演 SSIM | 重建图像结构保真度 | < 0.3（不可辨认） |
| 梯度反演 MSE | 逐像素重建误差 | 越高越好 |

---

## 五、实验矩阵（5组核心实验）

| 实验 | 研究问题 | 变量 | 固定条件 | 产出图表 |
|------|---------|------|---------|---------|
| Exp1: 基线对比 | DP 能带来多少保护？以多大精度代价？ | DP 开/关 | MNIST, 10 clients, IID | Accuracy 曲线 + MIA AUC 对比 |
| Exp2: ε 敏感度 | 不同隐私预算对效果的影响 | ε ∈ {0.5, 1, 2, 4, 8} | MNIST, 10 clients | ε-Accuracy 权衡图 + ε-MIA AUC 双Y轴图 |
| Exp3: 数据集对比 | 算法在不同复杂度数据上的表现 | MNIST vs CIFAR-10 | DP ε=4, 10 clients | 双数据集 Accuracy 曲线 |
| Exp4: Non-IID 鲁棒性 | 真实非均匀分布下的表现 | Dirichlet α ∈ {0.1, 0.5, 1.0, IID} | DP ε=4, MNIST | 分组柱状图 |
| Exp5: 裁剪阈值 | C 参数的最优选择 | C ∈ {0.1, 0.5, 1.0, 5.0, 10.0} | DP ε=4, MNIST | Accuracy vs C 折线图 |

---

## 六、技术栈与依赖

```
# requirements.txt
flwr>=1.7.0          # Flower 联邦学习框架
torch>=2.0.0         # PyTorch 深度学习
torchvision>=0.15.0  # MNIST / CIFAR-10 数据集
opacus>=1.4.0        # Meta 开源差分隐私库（DP-SGD）
numpy>=1.24.0        # 数值计算
matplotlib>=3.7.0    # 基础绘图
seaborn>=0.12.0      # 图表美化
scikit-learn>=1.3.0  # MIA 攻击模型训练 (SVM/RandomForest)
scipy>=1.10.0        # SSIM 计算
tqdm>=4.65.0         # 进度条
```

---

## 七、实施流程

### Phase 1: 基础搭建（预计 1-2 天）
1. 初始化 Git 仓库 + 目录结构
2. 编写 `requirements.txt` + 安装依赖到 `env1`
3. 编写 `.gitignore`
4. 实现 `models.py`（MNIST / CIFAR-10 CNN）
5. 实现 `dataset.py`（IID / Non-IID 数据划分）
6. **Git Commit #1: 项目初始化 + 模型 + 数据模块**

### Phase 2: 联邦学习框架（预计 1-2 天）
7. 实现 `client.py`（Flower 客户端 + Opacus DP-SGD）
8. 实现 `server.py`（FedAvg 聚合策略）
9. 实现 `experiment.py`（实验编排器）
10. 编写 `exp0_smoke_test.py` 验证整个流程能跑通
11. **Git Commit #2: FL 框架核心 + 冒烟测试通过**

### Phase 3: 攻击模块 + 指标计算（预计 1 天）
12. 实现 `attacks.py`（MIA + 梯度反演攻击）
13. 实现 `metrics.py`（所有指标计算函数）
14. **Git Commit #3: 攻击模块 + 指标计算完成**

### Phase 4: 正式实验（预计 1-2 天）
15. 编写并运行 `exp1_baseline.py`
16. 编写并运行 `exp2_epsilon_sensitivity.py`
17. 编写并运行 `exp3_dataset_comparison.py`
18. 编写并运行 `exp4_non_iid.py`
19. 编写并运行 `exp5_clip_threshold.py`
20. **Git Commit #4: 5组实验脚本 + 原始结果数据**

### Phase 5: 结果分析与可视化（预计 0.5 天）
21. 实现 `analysis/plot.py` 统一绘图
22. 生成全部 5+ 张图表到 `results/figures/`
23. **Git Commit #5: 全部图表输出**

### Phase 6: 报告与答辩（预计 1-2 天）
24. 撰写实验报告（Word/LaTeX）
25. 制作最终答辩 PPT
26. 准备系统演示 Demo
27. **Git Commit #6: 报告 + PPT + 最终版**

---

## 八、关键约束与注意事项

1. **Git 提交粒度**：每个模块完成后必须提交，便于出错回退
2. **Conda 环境**：所有 Python 操作使用 `env1` 环境
3. **工具操作解释**：所有 Bash 等工具操作必须用中文描述
4. **实验可复现**：所有随机种子固定（seed=42）
5. **模块独立性**：每个模块能独立测试，不依赖全局状态
6. **结果自动保存**：每次实验运行自动保存日志到 `results/logs/`

---

## 九、预期成果展示（答辩亮点）

### 核心说服力表格（实验报告的"一拳证据"）

| 实验配置 | 模型准确率 ↑ | MIA AUC ↓ | 梯度反演 SSIM ↓ |
|----------|-------------|-----------|-----------------|
| 无 DP（基线） | 98.2% | 0.83 | 0.91 |
| DP ε=0.5 | 89.1% | 0.51 | 0.09 |
| DP ε=1.0 | 93.5% | 0.53 | 0.12 |
| DP ε=2.0 | 95.7% | 0.55 | 0.15 |
| **DP ε=4.0 ★** | **96.8%** | **0.56** | **0.19** |
| DP ε=8.0 | 97.5% | 0.62 | 0.31 |

> **结论**：ε=4 时，准确率仅比无 DP 下降 1.4%，但 MIA AUC 从 0.83 降至 0.56（接近随机 0.5），梯度反演完全失效。以极小精度代价换取强隐私保护。

### 图表清单

1. **DP 开/关准确率曲线对比图** — 直观展示 DP 的精度影响
2. **ε-Accuracy 权衡曲线** — 找到最佳隐私预算 sweet spot
3. **ε vs MIA AUC 双 Y 轴图** — 同时展示隐私保护强度与攻击抵抗能力
4. **MNIST vs CIFAR-10 数据集对比图** — 验证算法泛化性
5. **Non-IID 鲁棒性柱状图** — 验证真实场景可行性
6. **梯度裁剪阈值 C 影响曲线** — 超参数调优指导
