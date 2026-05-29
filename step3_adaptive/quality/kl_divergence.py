"""质量评估：KL 散度 — 衡量数据稀缺度"""
import numpy as np


def compute_kl_divergence(client_counts, global_counts):
    """计算单个客户端的 KL 散度

    KL(P_client || P_global) = Σ P(k) * log(P(k) / Q(k))

    KL 越高 → 客户端分布与全局差异越大 → 数据越稀缺/不可替代
    """
    P = client_counts.astype(np.float64) / client_counts.sum()
    Q = global_counts.astype(np.float64) / global_counts.sum()

    mask = (P > 0) & (Q > 0)
    if mask.sum() == 0:
        return 0.0

    return float(np.sum(P[mask] * np.log(P[mask] / Q[mask])))


def compute_all_kl(label_counts_list):
    """计算所有客户端的 KL 散度"""
    total_counts = np.sum(label_counts_list, axis=0)
    kls = []
    for counts in label_counts_list:
        kls.append(compute_kl_divergence(np.array(counts), total_counts))
    return kls


def counts_from_labels(labels, num_classes=10):
    """从标签数组计算每个类别的计数"""
    return np.bincount(labels, minlength=num_classes)
