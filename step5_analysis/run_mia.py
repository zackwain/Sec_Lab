"""Step 5: Membership Inference Attack (MIA) 隐私验证

对每个策略训练一个模型，然后用 confidence-based threshold attack
验证 DP 是否有效防止了成员推理。

原理: 模型对训练过的样本置信度更高 → 攻击者可通过置信度推断
      如果 DP 保护有效 → member/non-member 置信度分布重叠 → AUC ≈ 0.50

Attack:  max(softmax(output)) — Shokri et al. 2017
        高置信度 → 预测为 member, 低置信度 → 预测为 non-member

Usage:
  python -u step5_analysis/run_mia.py              # 训练 + MIA
  python -u step5_analysis/run_mia.py --skip-train # 仅 MIA（加载已保存模型）
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, ".")

import torch
import torch.nn as nn
import numpy as np
import json
import time
import pickle

from torch.utils.data import DataLoader, Subset
from step2_baseline.data import load_mnist, get_test_loader
from step2_baseline.model import MNISTCNN, set_parameters
from step4_time_adaptive import config
from step4_time_adaptive.config import set_seed
from step4_time_adaptive.trainer import run_training


MODEL_DIR = "results/models"

# ============================================================
# 模型保存 / 加载
# ============================================================

def model_path(model_key):
    """模型文件名。"""
    return os.path.join(MODEL_DIR, f"{model_key}_params.pkl")


def save_model(model_params, train_result, model_key):
    """保存模型参数和训练摘要到磁盘。"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    data = {
        "model_params": model_params,
        "strategy": train_result["strategy"],
        "final_accuracy": train_result["final_accuracy"],
        "total_eps_spent": train_result["total_eps_spent"],
        "rounds_completed": train_result["rounds_completed"],
    }
    with open(model_path(model_key), "wb") as f:
        pickle.dump(data, f)
    print(f"    Model saved → {model_path(model_key)}")


def load_model(model_key):
    """从磁盘加载模型参数和摘要。"""
    path = model_path(model_key)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


# ============================================================
# 单样本置信度计算
# ============================================================

@torch.no_grad()
def compute_per_sample_confidences(model_params, samples, device="cpu"):
    """用最终模型对给定样本逐条计算 max(softmax) 置信度。

    置信度高 → 模型对该样本"确定" → 更可能是训练集成员。
    """
    model = MNISTCNN().to(device)
    set_parameters(model, model_params)
    model.eval()

    confidences = []
    for image, label in samples:
        img = image.unsqueeze(0).to(device)
        output = model(img)
        probs = torch.softmax(output, dim=1)
        conf = probs.max().item()
        confidences.append(conf)

    return np.array(confidences)


# ============================================================
# MIA 主流程
# ============================================================

def run_mia(model_params, train_set, test_set, n_samples=5000, device="cpu"):
    """对给定模型执行 confidence-based 成员推理攻击。"""
    rng = np.random.RandomState(42)
    train_indices = rng.choice(len(train_set), size=n_samples, replace=False)
    test_indices = rng.choice(len(test_set), size=n_samples, replace=False)

    members = [(train_set[i][0], train_set[i][1]) for i in train_indices]
    non_members = [(test_set[i][0], test_set[i][1]) for i in test_indices]

    print(f"    Computing per-sample confidences ({n_samples}×2)...")
    member_confs = compute_per_sample_confidences(model_params, members, device)
    non_member_confs = compute_per_sample_confidences(model_params, non_members, device)

    scores = np.concatenate([member_confs, non_member_confs])
    labels = np.concatenate([
        np.ones(n_samples),
        np.zeros(n_samples),
    ])

    try:
        from sklearn.metrics import roc_auc_score, roc_curve
        auc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)

        low_fpr_idx = np.argmin(np.abs(fpr - 0.01))
        tpr_at_1pct_fpr = tpr[low_fpr_idx]

        best_acc = 0.0
        for thresh in np.percentile(scores, np.linspace(0, 100, 200)):
            preds = (scores >= thresh).astype(int)
            acc = (preds == labels).mean()
            if acc > best_acc:
                best_acc = acc

    except ImportError:
        print("    Warning: sklearn not available")
        auc = 0.0
        tpr_at_1pct_fpr = 0.0
        best_acc = 0.0
        for pct in range(5, 95, 5):
            thresh = np.percentile(member_confs, pct)
            preds = (scores >= thresh).astype(int)
            acc = (preds == labels).mean()
            if acc > best_acc:
                best_acc = acc

    return {
        "auc": float(auc),
        "best_accuracy": float(best_acc),
        "tpr_at_1pct_fpr": float(tpr_at_1pct_fpr),
        "member_conf_mean": float(np.mean(member_confs)),
        "member_conf_std": float(np.std(member_confs)),
        "non_member_conf_mean": float(np.mean(non_member_confs)),
        "non_member_conf_std": float(np.std(non_member_confs)),
        "n_samples": n_samples,
    }


# ============================================================
# 训练 + MIA
# ============================================================

def train_and_attack(strategy_name, train_set, test_set, use_dp=True, device="cpu"):
    """训练一个模型，保存到磁盘，然后执行 MIA。"""
    set_seed(42)
    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True,
    )
    test_loader = get_test_loader()

    label = f"{strategy_name}" + (" (DP)" if use_dp else " (no DP)")
    print(f"\n{'─' * 50}")
    print(f"  Training: {label}")
    print(f"{'─' * 50}")

    train_result = run_training(
        strategy_name, train_loader, test_loader,
        use_dp=use_dp, return_model=True, verbose=True,
    )

    acc = train_result["final_accuracy"]
    print(f"  Trained: acc={acc:.4f}, Σε={train_result['total_eps_spent']:.2f}")

    # 保存模型
    model_key = "nodp" if not use_dp else strategy_name.lower()
    save_model(train_result["model_params"], train_result, model_key)

    print(f"\n  Running MIA...")
    mia_result = run_mia(train_result["model_params"], train_set, test_set, device=device)

    return train_result, mia_result


# ============================================================
# 主入口
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MIA - Membership Inference Attack")
    parser.add_argument("--skip-train", action="store_true",
                       help="跳过训练，从 results/models/ 加载已有模型")
    args = parser.parse_args()

    device = config.DEVICE
    print("=" * 70)
    print("Step 5: Membership Inference Attack (MIA)")
    print(f"  Device: {device}")
    print(f"  Attack: confidence-based (Shokri 2017), 5000 v 5000")
    print(f"  Mode: {'SKIP TRAIN (load from disk)' if args.skip_train else 'TRAIN + ATTACK'}")
    print("=" * 70)

    train_set, test_set = load_mnist()
    print(f"\n  Data: {len(train_set)} train, {len(test_set)} test")

    model_keys = ["nodp", "uniform", "kianilinear", "kianiplusmomentum"]
    strategy_map = {
        "nodp": ("Uniform", False),
        "uniform": ("Uniform", True),
        "kianilinear": ("KianiLinear", True),
        "kianiplusmomentum": ("KianiPlusMomentum", True),
    }

    results = {}
    t0 = time.time()

    for key in model_keys:
        strat_name, use_dp = strategy_map[key]

        if args.skip_train:
            # 从磁盘加载
            print(f"\n{'─' * 50}")
            print(f"  Loading: {key}")
            print(f"{'─' * 50}")
            data = load_model(key)
            train_result = {
                "strategy": data["strategy"],
                "final_accuracy": data["final_accuracy"],
                "total_eps_spent": data["total_eps_spent"],
                "rounds_completed": data["rounds_completed"],
            }
            print(f"  Loaded: acc={train_result['final_accuracy']:.4f}, "
                  f"Σε={train_result['total_eps_spent']:.2f}")

            print(f"\n  Running MIA...")
            mia_result = run_mia(data["model_params"], train_set, test_set, device=device)
        else:
            # 训练 + 保存
            train_result, mia_result = train_and_attack(
                strat_name, train_set, test_set, use_dp=use_dp, device=device,
            )

        display_key = "NoDP" if key == "nodp" else strat_name
        results[display_key] = {"training": train_result, "mia": mia_result}

    duration = time.time() - t0

    # ========================================
    # 汇总对比
    # ========================================
    print("\n" + "=" * 70)
    print(f"MIA RESULTS  (confidence-based, AUC > 0.50 = privacy leak)")
    print("=" * 70)
    print(f"  {'Strategy':>24} | {'Accuracy':>10} | {'Σε':>8} | "
          f"{'AUC':>8} | {'Best Acc':>10} | {'TPR@1%FPR':>10}")
    print(f"  {'─'*24}─┼─{'─'*10}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*10}")

    order = ["NoDP", "Uniform", "KianiLinear", "KianiPlusMomentum"]
    for key in order:
        tr = results[key]["training"]
        mia = results[key]["mia"]
        acc_str = f"{tr['final_accuracy']:.4f}"
        eps_str = f"{tr['total_eps_spent']:.2f}" if tr['total_eps_spent'] > 0 else "─"
        print(f"  {key:>24} | {acc_str:>10} | {eps_str:>8} | "
              f"{mia['auc']:>8.4f} | {mia['best_accuracy']:>10.4f} | "
              f"{mia['tpr_at_1pct_fpr']:>10.4f}")

    print(f"\n  Interpretation:")
    print(f"    AUC ≈ 0.50  → 攻击者无法区分 → 隐私保护有效")
    print(f"    AUC ≈ 0.60+ → 攻击者能区分   → 隐私泄露")
    dp_aucs = [results[s]["mia"]["auc"] for s in ["Uniform", "KianiLinear", "KianiPlusMomentum"]]
    print(f"    DP AUC range: {min(dp_aucs):.4f} ~ {max(dp_aucs):.4f}")

    print(f"\n  Confidence stats (member vs non-member):")
    for key in order:
        mia = results[key]["mia"]
        gap = mia["member_conf_mean"] - mia["non_member_conf_mean"]
        print(f"    {key:>24}: member={mia['member_conf_mean']:.4f}, "
              f"non_member={mia['non_member_conf_mean']:.4f}, gap={gap:+.4f}")

    # ---- 保存 ----
    os.makedirs("results/logs", exist_ok=True)
    output = {
        "experiment": "mia_confidence",
        "attack_method": "confidence-based (Shokri 2017)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "total_epsilon": config.TOTAL_EPSILON,
            "max_rounds": config.MAX_ROUNDS,
            "epochs_per_round": config.EPOCHS_PER_ROUND,
            "batch_size": config.BATCH_SIZE,
            "n_attack_samples": 5000,
            "device": device,
            "skip_train": args.skip_train,
        },
        "results": {
            key: {
                "strategy": val["training"]["strategy"],
                "final_accuracy": val["training"]["final_accuracy"],
                "total_eps_spent": val["training"]["total_eps_spent"],
                "rounds_completed": val["training"]["rounds_completed"],
                "mia": val["mia"],
            }
            for key, val in results.items()
        },
        "duration_seconds": duration,
    }

    save_path = "results/logs/step5_mia.json"
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved → {save_path}")


if __name__ == "__main__":
    main()
