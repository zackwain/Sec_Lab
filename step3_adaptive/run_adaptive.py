"""Step 3 V4: Per-Client Budget Tracked DP-FL

每客户端 ε_total=8，独立追踪分配值，花光停训。
Pool=4.0/轮，三因素评分决定池内比例。20轮上限。

python -u step3_adaptive/run_adaptive.py
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
from opacus import PrivacyEngine

from step2_baseline.model import MNISTCNN, get_parameters, set_parameters
from step2_baseline.data import (
    load_mnist, split_data, get_test_loader, get_client_labels,
)
from step3_adaptive.quality.quality_estimator import (
    compute_static_scores, compute_loss_drop, compute_all_scores,
)
from step3_adaptive.quality.kl_divergence import counts_from_labels
from step3_adaptive.allocator import (
    calculate_noise_multiplier, compute_gradient_alignment,
)

# ============================================================
# 配置
# ============================================================
TOTAL_EPSILON = 8.0       # 每客户端总预算
POOL = 4.0                # 每轮总池子 (10 × 0.4)
EPSILON_BASE = 0.4        # uniform 基准线
TEMPERATURE = 0.5         # softmax 温度
MAX_ROUNDS = 20           # 轮数上限
EPOCHS = 1
DELTA = 1e-5
MAX_GRAD_NORM = 1.0
LR = 0.01
BATCH_SIZE = 64
DISTRIBUTION = "dirichlet"
DIRICHLET_ALPHA = 0.3
DEVICE = "cpu"

BEST_WEIGHTS = (0.7, 0.15, 0.15)


def train_one_client_dp(model, train_loader, epsilon, delta=DELTA,
                         max_grad_norm=MAX_GRAD_NORM, device=DEVICE,
                         lr=LR, epochs=EPOCHS):
    dataset_size = len(train_loader.dataset)
    sample_rate = train_loader.batch_size / dataset_size
    criterion = nn.CrossEntropyLoss()

    # loss_before
    model.eval()
    loss_before, n_before = 0.0, 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_before += criterion(outputs, labels).item() * images.size(0)
            n_before += images.size(0)
    loss_before /= n_before if n_before > 0 else 1.0

    # ε 过小则跳过训练
    if epsilon < 0.05:
        params = get_parameters(model)
        return params, 0, loss_before, 0.0, loss_before

    # DP-SGD
    sigma = calculate_noise_multiplier(epsilon, delta, sample_rate, epochs=epochs)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    dp_loader = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=train_loader.batch_size, shuffle=True,
    )
    privacy_engine = PrivacyEngine()
    model, optimizer, dp_loader = privacy_engine.make_private(
        module=model, optimizer=optimizer, data_loader=dp_loader,
        noise_multiplier=sigma, max_grad_norm=max_grad_norm,
    )
    total_loss, num_samples = 0.0, 0
    for _ in range(epochs):
        for images, labels in dp_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

    eps_spent = privacy_engine.get_epsilon(delta=delta)
    unwrapped = getattr(model, '_module', model)
    unwrapped.eval()
    loss_after, n_after = 0.0, 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = unwrapped(images)
            loss_after += criterion(outputs, labels).item() * images.size(0)
            n_after += images.size(0)
    loss_after /= n_after if n_after > 0 else 1.0

    return get_parameters(unwrapped), num_samples, loss_after, eps_spent, loss_before


def evaluate_model(model, test_loader, device=DEVICE):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += images.size(0)
    return correct / total if total > 0 else 0.0, total_loss / total if total > 0 else 0.0


def fedavg_aggregate(client_updates):
    total_samples = sum(n for _, n, _, _, _ in client_updates if n > 0)
    if total_samples == 0:
        return None
    aggregated = None
    for params, n, _, _, _ in client_updates:
        if n == 0:
            continue
        w = n / total_samples
        if aggregated is None:
            aggregated = [p * w for p in params]
        else:
            for i in range(len(aggregated)):
                aggregated[i] = aggregated[i] + params[i] * w
    return aggregated


def allocate_pool(scores, remaining, pool=POOL, T=TEMPERATURE):
    """指数分配 + budget clamping + 余量重分配"""
    n = len(scores)
    eps = np.zeros(n)
    active = [i for i in range(n) if remaining[i] >= 0.05]

    if not active:
        return eps.tolist()

    active_scores = np.array([scores[i] for i in active], dtype=np.float64)
    exp_s = np.exp(active_scores / T)
    total = exp_s.sum()

    for idx, i in enumerate(active):
        e = pool * exp_s[idx] / total
        e = min(e, remaining[i])
        eps[i] = e

    for _ in range(20):
        gap = pool - eps.sum()
        if gap < 0.001:
            break
        available = [i for i in active if eps[i] < remaining[i] - 0.001]
        if not available:
            break
        per = gap / len(available)
        for i in available:
            e = eps[i] + per
            e = min(e, remaining[i])
            eps[i] = e

    return eps.tolist()


def run_uniform(train_loaders, test_loader, verbose=True):
    N = len(train_loaders)
    model = MNISTCNN()
    rounds_data = []
    remaining = [TOTAL_EPSILON] * N
    t0 = time.time()

    for rnd in range(1, MAX_ROUNDS + 1):
        if all(r < EPSILON_BASE * 0.5 for r in remaining):
            break

        updates = []
        for cid in range(N):
            if remaining[cid] >= EPSILON_BASE * 0.5:
                local = MNISTCNN()
                set_parameters(local, get_parameters(model))
                e = min(EPSILON_BASE, remaining[cid])
                params, n_s, la, spent, lb = train_one_client_dp(
                    local, train_loaders[cid], epsilon=e,
                )
                remaining[cid] -= e
                remaining[cid] = max(remaining[cid], 0.0)
            else:
                params = get_parameters(model)
                n_s, la, spent, lb = 0, 0.0, 0.0, 0.0
            updates.append((params, n_s, la, spent, lb))

        agg = fedavg_aggregate(updates)
        if agg is None:
            break
        set_parameters(model, agg)
        acc, loss = evaluate_model(model, test_loader)

        rounds_data.append({
            "round": rnd, "accuracy": acc, "loss": loss,
            "max_epsilon": max(e for _, _, _, e, _ in updates),
        })
        if verbose:
            print(f"   [uniform]  Round {rnd:2d}/{MAX_ROUNDS}: "
                  f"acc={acc:.4f}, eps={EPSILON_BASE:.1f}")

    return {
        "final_accuracy": rounds_data[-1]["accuracy"] if rounds_data else 0.0,
        "converged_round": len(rounds_data),
        "duration_seconds": time.time() - t0,
        "rounds": rounds_data,
    }


def run_adaptive(train_loaders, test_loader, static_scores, weights,
                  verbose=True):
    N = len(train_loaders)
    model = MNISTCNN()
    rounds_data = []
    remaining = [TOTAL_EPSILON] * N
    epsilons = [EPSILON_BASE] * N
    stored_before, stored_after = None, None

    for rnd in range(1, MAX_ROUNDS + 1):
        if all(r < 0.05 for r in remaining):
            break

        params_before = get_parameters(model)
        updates = []
        param_list = []
        lbs, las = [], []

        for cid in range(N):
            if remaining[cid] < 0.05:
                p = get_parameters(model)
                updates.append((p, 0, 0.0, 0.0, 0.0))
                param_list.append(p)
                lbs.append(0.0); las.append(0.0)
                continue

            local = MNISTCNN()
            set_parameters(local, get_parameters(model))
            e = min(epsilons[cid], remaining[cid])
            p, n_s, la, spent, lb = train_one_client_dp(
                local, train_loaders[cid], epsilon=e,
            )
            remaining[cid] -= e
            remaining[cid] = max(remaining[cid], 0.0)
            updates.append((p, n_s, la, spent, lb))
            param_list.append(p)
            lbs.append(lb); las.append(la)

        agg = fedavg_aggregate(updates)
        if agg is None:
            break
        set_parameters(model, agg)
        acc, loss = evaluate_model(model, test_loader)

        ref_before = stored_before if stored_before is not None else params_before
        ref_after = stored_after if stored_after is not None else agg
        drops = [compute_loss_drop(lb, la) for lb, la in zip(lbs, las)]
        aligns = [compute_gradient_alignment(param_list[c], ref_before, ref_after)
                  for c in range(N)]
        scores = compute_all_scores(static_scores, drops, aligns, weights)
        epsilons = allocate_pool(scores, remaining, POOL)
        stored_before, stored_after = params_before, agg

        rounds_data.append({
            "round": rnd, "accuracy": acc, "loss": loss,
            "max_epsilon": float(max(spent for _, _, _, spent, _ in updates)),
            "loss_drops": [float(d) for d in drops],
            "alignments": [float(a) for a in aligns],
            "epsilons": [float(e) for e in epsilons],
        })

        if verbose:
            avg_drop = np.mean(drops) if drops else 0.0
            avg_align = np.mean(aligns) if aligns else 0.0
            print(f"   [adaptive] Round {rnd:2d}/{MAX_ROUNDS}: "
                  f"acc={acc:.4f}, eps_range=[{min(epsilons):.1f},{max(epsilons):.1f}], "
                  f"drop={avg_drop:.3f}, align={avg_align:.3f}")

    return {
        "final_accuracy": rounds_data[-1]["accuracy"] if rounds_data else 0.0,
        "converged_round": len(rounds_data),
        "rounds": rounds_data,
    }


def main():
    print("=" * 64)
    print("Step 3: Per-Client Budget Tracked DP-FL")
    print(f"  ε_total/人={TOTAL_EPSILON}, pool/轮={POOL}")
    print(f"  max_rounds={MAX_ROUNDS}, T={TEMPERATURE}")
    print(f"  weights={BEST_WEIGHTS}")
    print(f"  data: {DISTRIBUTION} α={DIRICHLET_ALPHA}")
    print("=" * 64)

    train_set, test_set = load_mnist()
    test_loader = get_test_loader()

    full_loaders, full_indices = split_data(
        train_set, 10, DISTRIBUTION,
        dirichlet_alpha=DIRICHLET_ALPHA, batch_size=BATCH_SIZE,
    )
    full_labels = get_client_labels(train_set, full_indices)
    full_counts = [counts_from_labels(l) for l in full_labels]
    full_static = compute_static_scores(full_counts)
    print(f"\n  KL: {[f'{s:.3f}' for s in full_static]}")

    print("\n[Exp 1] Uniform ...")
    result_u = run_uniform(full_loaders, test_loader)

    print("\n[Exp 2] Adaptive ...")
    result_a = run_adaptive(full_loaders, test_loader, full_static,
                            weights=BEST_WEIGHTS)

    print("\n" + "=" * 64)
    print("FINAL RESULTS")
    print("=" * 64)
    au, aa = result_u["final_accuracy"], result_a["final_accuracy"]
    ru, ra = result_u["converged_round"], result_a["converged_round"]
    delta = aa - au
    print(f"  {'':>12} {'Uniform':>10} {'Adaptive':>10}")
    print(f"  {'Accuracy':>12} {au:>10.4f} {aa:>10.4f}")
    print(f"  {'Rounds':>12} {str(ru):>10} {str(ra):>10}")
    print(f"  {'Δ':>12} {'':>10} {delta:>+10.4f}")
    print(f"\n  >>> {'WIN' if delta > 0 else 'NO WIN'} <<<")

    os.makedirs("results/logs", exist_ok=True)
    out = {
        "version": "v4_budget_tracked",
        "config": {
            "total_epsilon_per_client": TOTAL_EPSILON,
            "pool_per_round": POOL,
            "temperature": TEMPERATURE,
            "max_rounds": MAX_ROUNDS,
            "epochs": EPOCHS,
            "distribution": DISTRIBUTION,
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "weights": list(BEST_WEIGHTS),
        },
        "uniform": result_u,
        "adaptive": result_a,
        "delta_accuracy": delta,
    }
    with open("results/logs/step3_adaptive.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved → results/logs/step3_adaptive.json")


if __name__ == "__main__":
    main()
