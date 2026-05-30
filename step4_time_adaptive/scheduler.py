"""噪声调度策略：Uniform / Kiani Linear / Kiani + Loss Momentum"""

import numpy as np
from opacus.accountants.utils import get_noise_multiplier


def epsilon_to_sigma(epsilon, delta=1e-5, sample_rate=0.1, epochs=1):
    """ε → σ 转换（Opacus RDP 会计师）

    用于单轮 (epsilon, delta)-DP 的噪声校准。
    基本组合下每轮独立，累积 ε = Σε_r。

    Args:
        epsilon: 本轮目标 ε
        delta: DP δ 参数
        sample_rate: batch_size / dataset_size
        epochs: 每轮本地 epoch 数

    Returns:
        float: 噪声乘数 σ
    """
    if epsilon <= 0.01:
        return 100.0
    sigma = get_noise_multiplier(
        target_epsilon=epsilon,
        target_delta=delta,
        sample_rate=sample_rate,
        epochs=epochs,
        accountant="rdp",
    )
    return max(sigma, 0.1)


# ============================================================
# Strategy A: Uniform
# ============================================================

def uniform_schedule(total_epsilon, total_rounds):
    """每轮均分 ε。

    Args:
        total_epsilon: 总隐私预算
        total_rounds: 最大轮数

    Returns:
        list[float]: 长度 = total_rounds，Σ = total_epsilon
    """
    eps_per_round = total_epsilon / total_rounds
    return [eps_per_round] * total_rounds


# ============================================================
# Strategy B: Kiani Linear（固定两阶段线性递增）
# ============================================================

def kiani_linear_schedule(total_epsilon, total_rounds,
                          weight_start=0.5, weight_end=1.5):
    """Kiani ICLR 2025 等价：线性递增基调。

    前半段省钱（权重低）、后半段花钱（权重高）。
    权重从 weight_start 线性增长到 weight_end，归一化后 Σ = total_epsilon。

    Args:
        total_epsilon: 总隐私预算
        total_rounds: 最大轮数
        weight_start: 第 1 轮权重（默认 0.5）
        weight_end: 第 R 轮权重（默认 1.5）

    Returns:
        list[float]: 长度 = total_rounds，Σ = total_epsilon
    """
    weights = np.linspace(weight_start, weight_end, total_rounds)
    normalized = weights / weights.sum() * total_epsilon
    return normalized.tolist()


# ============================================================
# Strategy C: Kiani + Loss Momentum（OURS）
# ============================================================

def loss_momentum(loss_history, window=4):
    """计算损失动量。

    综合当前下降率 + 加速度（二阶变化）。
    加速度大 → 指数放大（正在突破，值得加码）。
    加速度负 → 弱化信号（强弩之末，该省）。

    Args:
        loss_history: list[float]，最近 window 轮的 loss 值
        window: 回看窗口大小

    Returns:
        float in [-1, 1]: 正值 = 加速下降，负值 = 趋于平缓
    """
    if len(loss_history) < 3:
        return 0.0

    # 只取最近 window 轮
    recent = loss_history[-window:]

    # 计算逐轮相对下降率
    drops = []
    for i in range(len(recent) - 1):
        before = recent[i]
        after = recent[i + 1]
        if before <= 0:
            drops.append(0.0)
        else:
            drop = (before - after) / before
            drops.append(max(drop, 0.0))  # loss 上升视为 0（异常）

    if len(drops) < 2:
        return float(np.clip(drops[-1] * 5, -1.0, 1.0)) if drops else 0.0

    # 当前下降率（一阶）
    current_drop = drops[-1]

    # 加速度（二阶）: 正的 = 下降在加速
    acceleration = drops[-1] - drops[-2]

    # 综合动量 = 当前速度 × exp(γ × 加速度)
    # acceleration 从 -0.2 到 +0.2 大概范围，乘 10 放大
    accel_boost = np.exp(np.clip(acceleration * 10.0, -5.0, 5.0))

    momentum = current_drop * accel_boost

    return float(np.clip(momentum * 3.0, -1.0, 1.0))


def kiani_plus_momentum_schedule(total_epsilon, total_rounds, loss_history,
                                 gamma=2.0, weight_start=0.5, weight_end=1.5,
                                 multiplier_min=0.2, multiplier_max=5.0):
    """Kiani 基调 × 损失动量指数调节（OURS）。

    ε(r) = ε_kiani(r) × exp(γ × momentum(r))

    然后在当前已分配轮次内归一化，保证 Σε ≈ total_epsilon。

    Args:
        total_epsilon: 总隐私预算
        total_rounds: 最大轮数
        loss_history: list[float]，历史 loss 值
        gamma: 指数响应强度
        weight_start: Kiani 起始权重
        weight_end: Kiani 结束权重
        multiplier_min: 乘数下限
        multiplier_max: 乘数上限

    Returns:
        list[float]: 长度 = total_rounds，Σ = total_epsilon
    """
    # 第一步：Kiani 基调
    kiani_base = kiani_linear_schedule(total_epsilon, total_rounds,
                                       weight_start, weight_end)

    # 第二步：对每一轮施加动量乘数
    epsilons = []
    for r in range(total_rounds):
        if r == 0:
            # 第一轮无历史，乘数 = 1
            multiplier = 1.0
        else:
            # 用前 r 轮的 loss 历史计算动量
            available_history = loss_history[:r + 1]
            momentum = loss_momentum(available_history)
            multiplier = np.exp(gamma * momentum)
            multiplier = np.clip(multiplier, multiplier_min, multiplier_max)

        epsilons.append(kiani_base[r] * multiplier)

    # 第三步：归一化保证总预算不超
    total = sum(epsilons)
    if total > 0:
        epsilons = [e * total_epsilon / total for e in epsilons]

    return epsilons


# ============================================================
# 调度器工厂函数
# ============================================================

def create_schedule(strategy_name, total_epsilon, total_rounds,
                    loss_history=None, gamma=2.0):
    """创建指定策略的 ε 调度。

    Args:
        strategy_name: "Uniform" | "KianiLinear" | "KianiPlusMomentum"
        total_epsilon: 总隐私预算
        total_rounds: 最大轮数
        loss_history: loss 历史（KianiPlusMomentum 需要）
        gamma: 指数响应强度

    Returns:
        list[float]: 每轮 ε 值
    """
    if strategy_name == "Uniform":
        return uniform_schedule(total_epsilon, total_rounds)

    elif strategy_name == "KianiLinear":
        return kiani_linear_schedule(total_epsilon, total_rounds)

    elif strategy_name == "KianiPlusMomentum":
        if loss_history is None:
            # 无历史时退化为 KianiLinear
            return kiani_linear_schedule(total_epsilon, total_rounds)
        return kiani_plus_momentum_schedule(total_epsilon, total_rounds,
                                            loss_history, gamma=gamma)

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
