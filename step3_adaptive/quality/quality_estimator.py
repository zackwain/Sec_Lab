"""质量评估模块：多因素综合评分

三因素：
  1. KL散度（静态） — 数据稀缺度
  2. 损失下降率（动态） — 学习潜力
  3. 梯度对齐度（动态） — 方向一致性
"""
import numpy as np
from step3_adaptive.quality.kl_divergence import compute_all_kl


def compute_static_scores(label_counts_list):
    """计算静态得分（KL散度归一化）

    Args:
        label_counts_list: 每个客户端的标签计数列表（num_classes 维数组）

    Returns:
        list[float]: 每个客户端的静态得分 [0, 1]
    """
    kl_values = compute_all_kl(label_counts_list)
    max_kl = max(kl_values) if kl_values else 1.0
    if max_kl == 0:
        return [0.5] * len(kl_values)
    return [kl / max_kl for kl in kl_values]


def compute_loss_drop(loss_before, loss_after):
    """计算损失下降率

    Returns:
        float: (loss_before - loss_after) / loss_before，截断到 [0, 1]
    """
    if loss_before <= 0:
        return 0.0
    drop = (loss_before - loss_after) / loss_before
    return float(np.clip(drop, 0.0, 1.0))


def compute_multi_factor_score(kl_norm, loss_drop, alignment, weights):
    """计算单个客户端的综合得分（V3：KL门控 + 过拟合惩罚）

    KL 门控：对齐度 × (1-KL)。KL 高 = 数据稀缺 → 不对齐不扣分。
    过拟合惩罚：下降率 < 0.1 时平方级衰减，学不动的客户端自动降权。

    Args:
        kl_norm: 归一化 KL 散度 [0, 1]
        loss_drop: 原始损失下降率 [0, 1]
        alignment: 原始梯度对齐度 [0, 1]
        weights: (w_kl, w_loss, w_align) 权重元组，和为 1

    Returns:
        float: 综合得分 [0, 1]
    """
    w_kl, w_loss, w_align = weights

    # 损失下降：过拟合惩罚 — raw < 0.1 时平方衰减
    loss_score = loss_drop * min(1.0, loss_drop / 0.1)

    # 梯度对齐：KL 门控 — 稀缺数据不对齐也正常
    align_score = alignment * (1.0 - kl_norm)

    return w_kl * kl_norm + w_loss * loss_score + w_align * align_score


def compute_all_scores(static_scores, loss_drops, alignments, weights):
    """计算所有客户端的综合得分

    Args:
        static_scores: 每个客户端的静态得分列表
        loss_drops: 每个客户端的损失下降率列表
        alignments: 每个客户端的梯度对齐度列表
        weights: (w_kl, w_loss, w_align)

    Returns:
        list[float]: 每个客户端的综合得分
    """
    scores = []
    for i in range(len(static_scores)):
        s = compute_multi_factor_score(
            static_scores[i], loss_drops[i], alignments[i], weights
        )
        scores.append(s)
    return scores
