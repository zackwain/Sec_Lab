"""Step 4 消融实验：γ 灵敏度 + 轮数影响

两个实验，单次运行，验证方法的鲁棒性。

  A. γ 消融: γ ∈ {1.0, 2.0, 3.0}
     → 证明改进不是调参调出来的，对超参数鲁棒

  B. 轮数消融: max_rounds ∈ {20, 30, 40}
     → 验证信号驱动在不同训练长度下都有效
     → 检验 Kiani 理论在更长的训练中是否成立

Uniform 和 KianiLinear 不受 γ 影响，γ 消融只跑 KianiPlusMomentum。

Usage:
  python -u step4_time_adaptive/run_ablation.py
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, ".")

import torch
import numpy as np
import json
import time

from torch.utils.data import DataLoader
from step2_baseline.data import load_mnist, get_test_loader
from step4_time_adaptive import config
from step4_time_adaptive.config import set_seed
from step4_time_adaptive.trainer import run_training


def print_result_table(results, title):
    """打印结果表格。"""
    print(f"\n{'─' * 65}")
    print(f"  {title}")
    print(f"{'─' * 65}")
    print(f"  {'Config':>24} | {'Strategy':>22} | {'Accuracy':>10}")
    print(f"  {'─' * 24}─┼─{'─' * 22}─┼─{'─' * 10}")

    for entry in results:
        label = entry.get("label", "")
        strat = entry["strategy"]
        acc = entry["final_accuracy"]
        print(f"  {label:>24} | {strat:>22} | {acc:>10.4f}")

    print(f"{'─' * 65}")


def run_gamma_ablation(train_set, test_set):
    """γ 消融实验：KianiPlusMomentum 在 γ=1.0, 2.0, 3.0 下的表现。

    Uniform 和 KianiLinear 不受 γ 影响，各跑一次作为参考基线。
    """
    print("\n" + "=" * 70)
    print("ABLATION A: GAMMA SENSITIVITY  (γ = 1.0, 2.0, 3.0)")
    print("=" * 70)

    set_seed(42)
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = get_test_loader()

    results = []
    t0 = time.time()

    # ---- 参考基线（γ 不影响这两者） ----
    for strat in ["Uniform", "KianiLinear"]:
        print(f"\n  [{strat}] baseline (γ irrelevant)")
        r = run_training(strat, train_loader, test_loader)
        r["label"] = f"γ=─ (baseline)"
        results.append(r)

    # ---- KianiPlusMomentum × 3 个 γ ----
    for gamma in [1.0, 2.0, 3.0]:
        label = f"γ={gamma:.1f}"
        print(f"\n  [KianiPlusMomentum] {label}")
        set_seed(42)
        train_ldr = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
        r = run_training("KianiPlusMomentum", train_ldr, test_loader, gamma=gamma)
        r["label"] = label
        results.append(r)

    duration = time.time() - t0
    print_result_table(results, f"Gamma Ablation Results  ({duration:.0f}s)")

    # ---- 保存 ----
    os.makedirs("results/logs", exist_ok=True)
    output = {
        "experiment": "gamma_ablation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "total_epsilon": config.TOTAL_EPSILON,
            "max_rounds": config.MAX_ROUNDS,
            "epochs_per_round": config.EPOCHS_PER_ROUND,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LR,
            "gamma_values": [1.0, 2.0, 3.0],
            "seed": 42,
        },
        "results": [
            {
                "label": r["label"],
                "strategy": r["strategy"],
                "final_accuracy": r["final_accuracy"],
                "rounds_completed": r["rounds_completed"],
                "total_eps_spent": r["total_eps_spent"],
                "duration_seconds": r["duration_seconds"],
                "per_round": r["per_round"],
            }
            for r in results
        ],
        "duration_seconds": duration,
    }
    path = "results/logs/step4_ablation_gamma.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved → {path}")

    return results


def run_rounds_ablation(train_set, test_set):
    """轮数消融实验：T ∈ {20, 30, 40} 下三种策略的对比。

    随着轮数增加：
      - 每轮 ε 被摊薄 → 学习变慢 → 粗/细粒度阶段开始分离
      - 预测：KianiLinear 在 T=30/40 时可能追上 Uniform
      - 预测：KianiPlusMomentum 在所有长度下都最优
    """
    print("\n" + "=" * 70)
    print("ABLATION B: ROUNDS ABLATION  (T = 20, 30, 40)")
    print("=" * 70)

    test_loader = get_test_loader()

    strategies = ["Uniform", "KianiLinear", "KianiPlusMomentum"]
    round_values = [20, 30, 40]
    results = []
    t0 = time.time()

    for max_rounds in round_values:
        eps_per = config.TOTAL_EPSILON / max_rounds
        print(f"\n{'─' * 55}")
        print(f"  T = {max_rounds} rounds  (ε/round ≈ {eps_per:.3f})")
        print(f"{'─' * 55}")

        for strat in strategies:
            print(f"    [{strat}]")
            set_seed(42)
            train_loader = DataLoader(
                train_set, batch_size=config.BATCH_SIZE, shuffle=True,
            )
            r = run_training(strat, train_loader, test_loader,
                           max_rounds=max_rounds)
            r["label"] = f"T={max_rounds}"
            results.append(r)

    duration = time.time() - t0
    print_result_table(results, f"Rounds Ablation Results  ({duration:.0f}s)")

    # ---- 按 T 分组打印 ----
    print(f"\n  Summary by rounds:")
    print(f"  {'T':>4} | {'Uniform':>10} | {'KianiLinear':>12} | {'Momentum':>12}")
    print(f"  {'─'*4}─┼─{'─'*10}─┼─{'─'*12}─┼─{'─'*12}")
    for mr in round_values:
        accs = {}
        for r in results:
            if r["label"] == f"T={mr}":
                accs[r["strategy"]] = r["final_accuracy"]
        print(f"  {mr:>4} | {accs.get('Uniform', 0):>10.4f} | "
              f"{accs.get('KianiLinear', 0):>12.4f} | "
              f"{accs.get('KianiPlusMomentum', 0):>12.4f}")

    # ---- 保存 ----
    os.makedirs("results/logs", exist_ok=True)
    output = {
        "experiment": "rounds_ablation",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "total_epsilon": config.TOTAL_EPSILON,
            "rounds_values": round_values,
            "epochs_per_round": config.EPOCHS_PER_ROUND,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LR,
            "gamma": config.GAMMA,
            "seed": 42,
        },
        "results": [
            {
                "label": r["label"],
                "strategy": r["strategy"],
                "final_accuracy": r["final_accuracy"],
                "rounds_completed": r["rounds_completed"],
                "total_eps_spent": r["total_eps_spent"],
                "duration_seconds": r["duration_seconds"],
                "per_round": r["per_round"],
            }
            for r in results
        ],
        "duration_seconds": duration,
    }
    path = "results/logs/step4_ablation_rounds.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved → {path}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ablation experiments")
    parser.add_argument("--gamma", action="store_true", help="Run gamma ablation only")
    parser.add_argument("--rounds", action="store_true", help="Run rounds ablation only")
    args = parser.parse_args()

    # 没指定就全跑
    run_gamma = args.gamma or not args.rounds
    run_rounds = args.rounds or not args.gamma

    print("=" * 70)
    print("Step 4: Ablation Studies")
    print(f"  ε_total = {config.TOTAL_EPSILON}")
    print(f"  epochs/round = {config.EPOCHS_PER_ROUND}")
    print(f"  seed = 42  (fixed for reproducibility)")
    print(f"  gamma ablation:  {'ON' if run_gamma else 'OFF'}")
    print(f"  rounds ablation: {'ON' if run_rounds else 'OFF'}")
    print("=" * 70)

    train_set, test_set = load_mnist()
    print(f"\n  Dataset: {len(train_set)} train, {len(test_set)} test")

    # ========================================
    # 实验 A: γ 消融
    # ========================================
    if run_gamma:
        run_gamma_ablation(train_set, test_set)

    # ========================================
    # 实验 B: 轮数消融
    # ========================================
    if run_rounds:
        run_rounds_ablation(train_set, test_set)

    # ========================================
    # 汇总
    # ========================================
    print("\n" + "=" * 70)
    print("ALL ABLATIONS COMPLETE")
    print("=" * 70)
    print(f"  Results saved to:")
    print(f"    results/logs/step4_ablation_gamma.json")
    print(f"    results/logs/step4_ablation_rounds.json")


if __name__ == "__main__":
    main()
