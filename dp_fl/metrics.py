import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(predictions == labels))


def compute_mia_metrics(attack_probs: np.ndarray,
                         membership_labels: np.ndarray) -> dict:
    """计算 MIA 攻击的评估指标"""
    preds = (attack_probs >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(membership_labels, attack_probs)),
        "accuracy": float(accuracy_score(membership_labels, preds)),
    }


def compute_privacy_cost(no_dp_accuracy: float,
                          dp_accuracy: float) -> float:
    """隐私代价比：(无DP准确率 - 有DP准确率) / 无DP准确率"""
    if no_dp_accuracy == 0:
        return 0.0
    return float((no_dp_accuracy - dp_accuracy) / no_dp_accuracy)


def format_results_table(results: list) -> str:
    """将结果列表格式化为 Markdown 表格"""
    if not results:
        return ""

    keys = ["config", "accuracy", "epsilon", "mia_auc", "inversion_ssim"]
    header = "| 配置 | 准确率 | ε | MIA AUC | 反演 SSIM |"
    sep = "|------|--------|---|---------|-----------|"

    rows = []
    for r in results:
        row = (f"| {r.get('config', '')} "
               f"| {r.get('accuracy', 0):.4f} "
               f"| {r.get('epsilon', 0):.2f} "
               f"| {r.get('mia_auc', 0):.4f} "
               f"| {r.get('inversion_ssim', 0):.4f} |")
        rows.append(row)

    return "\n".join([header, sep] + rows)
