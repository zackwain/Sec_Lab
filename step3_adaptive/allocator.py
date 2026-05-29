"""自适应隐私预算分配器"""
import numpy as np


def allocate_uniform(num_clients, total_budget):
    """均匀分配（基线方法）

    每个客户端分配相同的 ε
    """
    return [total_budget / num_clients] * num_clients


def allocate_adaptive(qualities, total_budget):
    """自适应分配（核心创新）

    根据数据质量按比例分配隐私预算：
    - 高质量客户端 → 更多 ε（加较少噪声）
    - 低质量客户端 → 更少 ε（加较多噪声）

    总隐私成本保持不变：Σ ε_i = total_budget
    """
    total_quality = sum(qualities)
    if total_quality == 0:
        return allocate_uniform(len(qualities), total_budget)

    normalized = [q / total_quality for q in qualities]
    epsilons = [total_budget * q for q in normalized]
    return epsilons


def calculate_noise_multiplier(epsilon, delta=1e-5, sample_rate=0.1,
                                epochs=1, max_grad_norm=1.0):
    """根据目标 ε 计算噪声乘数 σ

    使用 Opacus 内置的 RDP 会计师校准，正确考虑采样率、迭代步数和目标 epsilon。
    """
    from opacus.accountants.utils import get_noise_multiplier

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


def allocate_exponential(scores, total_budget, T=0.5):
    """指数分配（softmax + 温度，V3）

    ε_i ∝ exp(Score_i / T)

    T 越小差距越大：
      T → ∞  : 趋近均匀分配
      T = 0.5: 中等放大（3-5x）
      T → 0  : 极端通吃
    """
    scores = np.array(scores, dtype=np.float64)
    exp_scores = np.exp(scores / T)
    if exp_scores.sum() == 0:
        return allocate_uniform(len(scores), total_budget)
    normalized = exp_scores / exp_scores.sum()
    return [float(total_budget * s) for s in normalized]


def compute_gradient_alignment(client_params, global_before, global_after=None,
                               aggregation_params=None):
    """计算客户端梯度与全局方向的对齐度

    对齐度 = cos(客户端更新, 全局方向)
    高 → 方向一致 → 学到共识知识
    低 → 方向偏离 → 可能数据异常或过拟合

    Args:
        client_params: 该客户端上传的参数
        global_before: 本轮开始前的全局参数
        global_after: 本轮聚合后的全局参数（优先使用）
        aggregation_params: 如果没有 global_after，可以用其他客户端的平均参数

    Returns:
        float: 余弦相似度，截断到 [0, 1]
    """
    if global_after is not None:
        # 使用全局方向作为参考
        ref_direction = [a - b for a, b in zip(global_after, global_before)]
    elif aggregation_params is not None:
        ref_direction = [a - b for a, b in zip(aggregation_params, global_before)]
    else:
        return 0.5  # 第一轮没有参考方向，返回中性值

    client_direction = [p - g for p, g in zip(client_params, global_before)]

    dot = sum(np.sum(c * r) for c, r in zip(client_direction, ref_direction))
    norm_c = np.sqrt(sum(np.sum(c ** 2) for c in client_direction))
    norm_r = np.sqrt(sum(np.sum(r ** 2) for r in ref_direction))

    if norm_c == 0 or norm_r == 0:
        return 0.5

    cos_sim = dot / (norm_c * norm_r)
    return float(np.clip(cos_sim, 0.0, 1.0))


def print_allocation(epsilons, qualities=None):
    """打印分配结果"""
    print("\n   隐私预算分配：")
    print(f"   {'客户端':>6} | {'质量分数':>8} | {'分配 ε':>8}")
    print(f"   {'------':>6} | {'--------':>8} | {'--------':>8}")

    for i, eps in enumerate(epsilons):
        q = qualities[i] if qualities else None
        q_str = f"{q:.4f}" if q else "N/A"
        print(f"   {i:>6} | {q_str:>8} | {eps:>8.2f}")

    print(f"   总计: Σε = {sum(epsilons):.2f}")
    print(f"   平均: ε = {np.mean(epsilons):.2f}")
    print(f"   范围: [{min(epsilons):.2f}, {max(epsilons):.2f}]")
