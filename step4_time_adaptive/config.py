"""统一配置 — 所有超参数集中管理"""
import random
import numpy as np
import torch

# ============================================================
# 隐私参数
# ============================================================
TOTAL_EPSILON = 6.0           # 总隐私预算（NIST 中等范围）
DELTA = 1e-5                  # δ 参数
MAX_GRAD_NORM = 1.0           # 梯度裁剪阈值 C

# ============================================================
# 训练参数
# ============================================================
MAX_ROUNDS = 20               # 最大通信轮数
EPOCHS_PER_ROUND = 1          # 每轮本地 epoch
LR = 0.01                     # 学习率
BATCH_SIZE = 600              # 批大小（与 Abadi 一致，q=0.01）
MOMENTUM = 0.9                # SGD 动量
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 调度参数
# ============================================================
GAMMA = 2.0                   # 损失动量指数响应强度
MOMENTUM_WINDOW = 4           # 动量计算的回看窗口（最近几轮）
KIANI_WEIGHT_START = 0.5      # Kiani 基调起始权重
KIANI_WEIGHT_END = 1.5        # Kiani 基调结束权重
MULTIPLIER_MIN = 0.2          # 乘数下限
MULTIPLIER_MAX = 5.0          # 乘数上限

# ============================================================
# 数据参数
# ============================================================
NUM_CLIENTS = 1               # 客户端数（Phase 1: 1, Phase 2: 5-10）
DISTRIBUTION = "dirichlet"    # 数据分布类型
DIRICHLET_ALPHA = 0.5         # Dirichlet 集中度

# ============================================================
# 实验参数
# ============================================================
SEED = 42                     # 随机种子
N_REPEATS = 3                 # 每组实验重复次数
EARLY_STOP_ACC = 2.0          # 早停准确率阈值（2.0 = 永不触发）

# ============================================================
# 策略定义
# ============================================================
STRATEGY_CONFIGS = {
    "Uniform": {
        "description": "每轮均分 ε = total / rounds",
    },
    "KianiLinear": {
        "description": "固定线性递增（对标 Kiani ICLR 2025）",
    },
    "KianiPlusMomentum": {
        "description": "Kiani 基调 × 损失动量指数调节（OURS）",
        "gamma": 2.0,
    },
}

# ============================================================
# 重复实验种子
# ============================================================
# 每个 repeat 用不同种子，保证模型初始化、数据 shuffle、DP 噪声独立采样
SEEDS = [SEED + i * 100 for i in range(N_REPEATS)]  # [42, 142, 242]


def set_seed(seed: int):
    """固定所有随机源，保证可复现。

    不同种子 → 不同的模型初始化 / 数据打乱 / DP 噪声采样。
    同一种子 → 结果完全一致。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
