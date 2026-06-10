"""Step 4: Time-Adaptive DP-FL — Three Strategies Comparison

单客户端 MNIST，ε_total = 6。
三种策略：
  A. Uniform              — 每轮均分
  B. KianiLinear          — 固定线性递增（对标 Kiani ICLR 2025）
  C. KianiPlusMomentum    — Kiani 基调 × 损失动量指数调节（OURS）

支持多次重复，取均值 ± 标准差。

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
from step4_time_adaptive.config import set_seed, SEEDS
from step4_time_adaptive.trainer import run_training
from step4_time_adaptive.scheduler import (
    uniform_schedule,
    kiani_linear_schedule,
)


def aggregate_repeats(repeat_list):
    """对多次重复结果计算均值 ± 标准差。

    Args:
        repeat_list: list[dict], 每个元素是一次 run_training 的返回值

    Returns:
        dict: {
            "accuracy_mean", "accuracy_std",
            "rounds_mean", "rounds_std",
            "eps_spent_mean", "eps_spent_std",
            "duration_mean", "duration_std",
            "per_round_accuracy": [{round, acc_mean, acc_std, ...}],
        }
    """
    n = len(repeat_list)
    if n == 0:
        return {}

    # ---- 标量指标 ----
    accs = [r["final_accuracy"] for r in repeat_list]
    rounds = [r["rounds_completed"] for r in repeat_list]
    eps = [r["total_eps_spent"] for r in repeat_list]
    durations = [r["duration_seconds"] for r in repeat_list]

    def mean_std(vals):
        vals = np.array(vals)
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    acc_m, acc_s = mean_std(accs)
    rnd_m, rnd_s = mean_std(rounds)
    eps_m, eps_s = mean_std(eps)
    dur_m, dur_s = mean_std(durations)

    # ---- 逐轮聚合 ----
    max_rounds = max(r["rounds_completed"] for r in repeat_list)
    per_round_agg = []
    for rnd in range(max_rounds):
        rnd_accs = []
        rnd_eps_alloc = []
        rnd_eps_spent = []
        rnd_momentum = []
        for r in repeat_list:
            if rnd < len(r["per_round"]):
                rnd_accs.append(r["per_round"][rnd]["accuracy"])
                rnd_eps_alloc.append(r["per_round"][rnd]["epsilon_allocated"])
                rnd_eps_spent.append(r["per_round"][rnd]["epsilon_spent"])
                rnd_momentum.append(r["per_round"][rnd]["momentum"])

        acc_m_r, acc_s_r = mean_std(rnd_accs)
        eps_a_m, _ = mean_std(rnd_eps_alloc)
        eps_s_m, _ = mean_std(rnd_eps_spent)
        mom_m, _ = mean_std(rnd_momentum)

        per_round_agg.append({
            "round": rnd + 1,
            "accuracy_mean": acc_m_r,
            "accuracy_std": acc_s_r,
            "epsilon_allocated_mean": eps_a_m,
            "epsilon_spent_mean": eps_s_m,
            "momentum_mean": mom_m,
        })

    return {
        "accuracy_mean": acc_m,
        "accuracy_std": acc_s,
        "rounds_mean": rnd_m,
        "rounds_std": rnd_s,
        "eps_spent_mean": eps_m,
        "eps_spent_std": eps_s,
        "duration_mean": dur_m,
        "duration_std": dur_s,
        "per_round_aggregated": per_round_agg,
    }


def main():
    n_repeats = config.N_REPEATS
    seeds = SEEDS[:n_repeats]

    print("=" * 70)
    print("Step 4: Time-Adaptive DP Noise Scheduling")
    print(f"  ε_total = {config.TOTAL_EPSILON}")
    print(f"  max_rounds = {config.MAX_ROUNDS}")
    print(f"  epochs/round = {config.EPOCHS_PER_ROUND}")
    print(f"  lr = {config.LR}")
    print(f"  gamma = {config.GAMMA}")
    print(f"  n_repeats = {n_repeats}")
    print(f"  seeds = {seeds}")
    print("=" * 70)

    # ========================================
    # 数据加载（只加载一次，每次 repeat 重新包 DataLoader）
    # ========================================
    train_set, test_set = load_mnist()
    test_loader = get_test_loader()

    sample_rate = config.BATCH_SIZE / len(train_set)
    print(f"\n  Samples: train={len(train_set)}, test={len(test_set)}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Sample rate (q): {sample_rate:.4f}")

    # ========================================
    # 预计算固定调度（预览用）
    # ========================================
    sched_u = uniform_schedule(config.TOTAL_EPSILON, config.MAX_ROUNDS)
    sched_k = kiani_linear_schedule(config.TOTAL_EPSILON, config.MAX_ROUNDS)

    print(f"\n  Schedule preview (first 5 of {config.MAX_ROUNDS} rounds):")
    print(f"    Uniform:      {[f'{e:.3f}' for e in sched_u[:5]]} ... "
          f"Σ={sum(sched_u):.2f}")
    print(f"    KianiLinear:  {[f'{e:.3f}' for e in sched_k[:5]]} ... "
          f"Σ={sum(sched_k):.2f}")

    # ========================================
    # 多次重复实验
    # ========================================
    strategy_names = ["Uniform", "KianiLinear", "KianiPlusMomentum"]
    all_raw = {s: [] for s in strategy_names}

    t_total_start = time.time()

    for rep_idx, seed in enumerate(seeds):
        print(f"\n{'#' * 70}")
        print(f"# REPEAT {rep_idx + 1}/{n_repeats}  (seed = {seed})")
        print(f"{'#' * 70}")

        set_seed(seed)

        # 重建 DataLoader：让 shuffle 受新种子影响
        train_loader = DataLoader(
            train_set, batch_size=config.BATCH_SIZE, shuffle=True,
        )

        for strat_name in strategy_names:
            label = f"[{rep_idx+1}/{n_repeats}] {strat_name}"
            print(f"\n{'-' * 50}")
            print(f"  {label}")
            print(f"{'-' * 50}")
            result = run_training(strat_name, train_loader, test_loader)
            result["seed"] = seed
            all_raw[strat_name].append(result)

    total_duration = time.time() - t_total_start

    # ========================================
    # 汇总统计
    # ========================================
    summaries = {}
    for strat_name in strategy_names:
        summaries[strat_name] = aggregate_repeats(all_raw[strat_name])

    # ========================================
    # 最终对比输出
    # ========================================
    print("\n" + "=" * 70)
    print(f"FINAL RESULTS  ({n_repeats} repeats, mean ± std)")
    print("=" * 70)
    header = (f"  {'Strategy':>24} | {'Accuracy':>16} | {'Rounds':>10} | "
              f"{'Σε':>12} | {'Time':>10}")
    print(header)
    print(f"  {'-'*24}-+-{'-'*16}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")

    for strat_name in strategy_names:
        s = summaries[strat_name]
        acc_str = f"{s['accuracy_mean']*100:.2f}% ± {s['accuracy_std']*100:.2f}%"
        rnd_str = f"{s['rounds_mean']:.1f} ± {s['rounds_std']:.1f}"
        eps_str = f"{s['eps_spent_mean']:.2f} ± {s['eps_spent_std']:.2f}"
        dur_str = f"{s['duration_mean']:.0f}s"
        print(f"  {strat_name:>24} | {acc_str:>16} | {rnd_str:>10} | "
              f"{eps_str:>12} | {dur_str:>10}")

    # ---- 优选出 ----
    best_idx = np.argmax([summaries[s]["accuracy_mean"] for s in strategy_names])
    best_name = strategy_names[best_idx]
    print(f"\n  ★ Best: {best_name} ({summaries[best_name]['accuracy_mean']*100:.2f}%)")

    # ---- 与 Uniform 的差距 ----
    uni_acc = summaries["Uniform"]["accuracy_mean"]
    mom_acc = summaries["KianiPlusMomentum"]["accuracy_mean"]
    improvement = (mom_acc - uni_acc) * 100
    print(f"  ★ Improvement over Uniform: +{improvement:.2f} pp")

    # ========================================
    # ε 分配轨迹对比
    # ========================================
    print(f"\n  Per-round ε allocation (mean across repeats):")
    print(f"  {'Round':>5} | {'Uniform':>10} | {'KianiLinear':>10} | "
          f"{'Momentum':>10} | {'Signal':>10}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for rnd in range(config.MAX_ROUNDS):
        u_eps = summaries["Uniform"]["per_round_aggregated"][rnd]["epsilon_allocated_mean"]
        k_eps = summaries["KianiLinear"]["per_round_aggregated"][rnd]["epsilon_allocated_mean"]
        m_eps = summaries["KianiPlusMomentum"]["per_round_aggregated"][rnd]["epsilon_allocated_mean"]
        m_mom = summaries["KianiPlusMomentum"]["per_round_aggregated"][rnd]["momentum_mean"]
        print(f"  {rnd+1:>5} | {u_eps:>10.3f} | {k_eps:>10.3f} | "
              f"{m_eps:>10.3f} | {m_mom:>+10.3f}")

    # ========================================
    # 保存结果
    # ========================================
    os.makedirs("results/logs", exist_ok=True)

    output = {
        "version": "v2_repeats",
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
            "n_repeats": n_repeats,
            "seeds": seeds,
        },
        "summaries": {},
        "repeats": {},
    }

    for strat_name in strategy_names:
        # 汇总
        output["summaries"][strat_name] = {
            k: v for k, v in summaries[strat_name].items()
            if k != "per_round_aggregated"
        }
        output["summaries"][strat_name]["per_round"] = (
            summaries[strat_name]["per_round_aggregated"]
        )
        # 每轮原始数据（精简版，只保留关键字段）
        output["repeats"][strat_name] = []
        for r in all_raw[strat_name]:
            output["repeats"][strat_name].append({
                "seed": r["seed"],
                "final_accuracy": r["final_accuracy"],
                "rounds_completed": r["rounds_completed"],
                "total_eps_spent": r["total_eps_spent"],
                "duration_seconds": r["duration_seconds"],
                "per_round": r["per_round"],
                "loss_history": r["loss_history"],
                "momentum_history": r["momentum_history"],
            })

    output["total_duration_seconds"] = total_duration

    save_path = "results/logs/step4_v2_repeats.json"
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved → {save_path}")

    return summaries, all_raw


if __name__ == "__main__":
    main()
