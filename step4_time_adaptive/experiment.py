"""Step 4: Time-Adaptive DP-FL — Three Strategies Comparison

单客户端 MNIST，ε_total = 6。
三种策略：
  A. Uniform              — 每轮均分
  B. KianiLinear          — 固定线性递增（对标 Kiani ICLR 2025）
  C. KianiPlusMomentum    — Kiani 基调 × 损失动量指数调节（OURS）

Usage:
  python -u step4_time_adaptive/experiment.py
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
from step4_time_adaptive.trainer import run_training


def main():
    print("=" * 70)
    print("Step 4: Time-Adaptive DP Noise Scheduling")
    print(f"  ε_total = {config.TOTAL_EPSILON}")
    print(f"  max_rounds = {config.MAX_ROUNDS}")
    print(f"  epochs/round = {config.EPOCHS_PER_ROUND}")
    print(f"  lr = {config.LR}")
    print(f"  gamma = {config.GAMMA}")
    print("=" * 70)

    # ========================================
    # 数据加载
    # ========================================
    train_set, test_set = load_mnist()
    test_loader = get_test_loader()
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)

    sample_rate = config.BATCH_SIZE / len(train_set)
    print(f"\n  Samples: train={len(train_set)}, test={len(test_set)}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Sample rate (q): {sample_rate:.4f}")

    # ========================================
    # 预计算固定调度（预览用）
    # ========================================
    from step4_time_adaptive.scheduler import (
        uniform_schedule,
        kiani_linear_schedule,
    )
    sched_u = uniform_schedule(config.TOTAL_EPSILON, config.MAX_ROUNDS)
    sched_k = kiani_linear_schedule(config.TOTAL_EPSILON, config.MAX_ROUNDS)

    print(f"\n  Schedule preview (first 5 of {config.MAX_ROUNDS} rounds):")
    print(f"    Uniform:      {[f'{e:.3f}' for e in sched_u[:5]]} ... "
          f"Σ={sum(sched_u):.2f}")
    print(f"    KianiLinear:  {[f'{e:.3f}' for e in sched_k[:5]]} ... "
          f"Σ={sum(sched_k):.2f}")

    results = {}

    # ========================================
    # Experiment 1: Uniform
    # ========================================
    print("\n" + "-" * 60)
    print("[1/3] Uniform Baseline")
    print("-" * 60)
    results["Uniform"] = run_training("Uniform", train_loader, test_loader)

    # ========================================
    # Experiment 2: Kiani Linear
    # ========================================
    print("\n" + "-" * 60)
    print("[2/3] Kiani Linear (Fixed Schedule — ICLR 2025)")
    print("-" * 60)
    results["KianiLinear"] = run_training("KianiLinear", train_loader, test_loader)

    # ========================================
    # Experiment 3: Kiani + Loss Momentum (OURS)
    # ========================================
    print("\n" + "-" * 60)
    print("[3/3] Kiani + Loss Momentum (OURS)")
    print("-" * 60)
    results["KianiPlusMomentum"] = run_training(
        "KianiPlusMomentum", train_loader, test_loader,
    )

    # ========================================
    # 最终对比
    # ========================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  {'Strategy':>24} | {'Accuracy':>10} | {'Rounds':>8} | {'Σε':>8} | {'Time':>8}")
    print(f"  {'-'*24}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for key in ["Uniform", "KianiLinear", "KianiPlusMomentum"]:
        r = results[key]
        print(f"  {r['strategy']:>24} | {r['final_accuracy']:>10.4f} | "
              f"{r['rounds_completed']:>8} | {r['total_eps_spent']:>8.2f} | "
              f"{r['duration_seconds']:>7.1f}s")

    best_acc = max(r['final_accuracy'] for r in results.values())
    fewest_rounds = min(r['rounds_completed'] for r in results.values())
    print(f"\n  Best accuracy:  {best_acc:.4f}")
    print(f"  Fewest rounds:  {fewest_rounds}")

    # 打印各自的 ε 分配轨迹
    print(f"\n  Epsilon allocation by round:")
    print(f"  {'Round':>5} | {'Uniform':>10} | {'KianiLinear':>10} | {'Momentum':>10} | {'Signal':>10}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for i in range(config.MAX_ROUNDS):
        u = results["Uniform"]["per_round"]
        k = results["KianiLinear"]["per_round"]
        m = results["KianiPlusMomentum"]["per_round"]
        u_eps = u[i]["epsilon_allocated"] if i < len(u) else "-"
        k_eps = k[i]["epsilon_allocated"] if i < len(k) else "-"
        m_eps = m[i]["epsilon_allocated"] if i < len(m) else "-"
        m_mom = m[i]["momentum"] if i < len(m) else "-"
        u_str = f"{u_eps:.3f}" if isinstance(u_eps, float) else u_eps
        k_str = f"{k_eps:.3f}" if isinstance(k_eps, float) else k_eps
        m_str = f"{m_eps:.3f}" if isinstance(m_eps, float) else m_eps
        mom_str = f"{m_mom:+.3f}" if isinstance(m_mom, float) else m_mom
        print(f"  {i+1:>5} | {u_str:>10} | {k_str:>10} | {m_str:>10} | {mom_str:>10}")

    # ========================================
    # 保存结果
    # ========================================
    os.makedirs("results/logs", exist_ok=True)
    output = {
        "version": "v1_time_adaptive",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "total_epsilon": config.TOTAL_EPSILON,
            "max_rounds": config.MAX_ROUNDS,
            "epochs_per_round": config.EPOCHS_PER_ROUND,
            "learning_rate": config.LR,
            "batch_size": config.BATCH_SIZE,
            "delta": config.DELTA,
            "max_grad_norm": config.MAX_GRAD_NORM,
            "gamma": config.GAMMA,
        },
        "results": {k: {
            "strategy": v["strategy"],
            "final_accuracy": v["final_accuracy"],
            "rounds_completed": v["rounds_completed"],
            "total_eps_spent": v["total_eps_spent"],
            "duration_seconds": v["duration_seconds"],
            "per_round": v["per_round"],
            "loss_history": v["loss_history"],
            "momentum_history": v["momentum_history"],
        } for k, v in results.items()},
    }

    save_path = "results/logs/step4_v1_experiment.json"
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved → {save_path}")

    return results


if __name__ == "__main__":
    main()
