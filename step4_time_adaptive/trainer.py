"""DP 训练核心模块：单轮训练 + 评估 + 全流程训练循环"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import torch.nn as nn
import numpy as np
from opacus import PrivacyEngine

from step2_baseline.model import MNISTCNN, get_parameters, set_parameters
from step4_time_adaptive import config
from step4_time_adaptive.scheduler import (
    epsilon_to_sigma,
    create_schedule,
    loss_momentum,
)


# ============================================================
# 单轮 DP 训练
# ============================================================

def train_one_round(master_params, train_loader, epsilon, *, delta=config.DELTA,
                    max_grad_norm=config.MAX_GRAD_NORM, lr=config.LR,
                    epochs=config.EPOCHS_PER_ROUND, device=config.DEVICE):
    """单轮 (epsilon, delta)-DP 训练。

    每轮创建全新的模型和 PrivacyEngine，避免 Opacus hook 冲突。
    总隐私通过基本组合追踪：R 轮后 = (Σε_r, R×δ)-DP。

    Args:
        master_params: 从主模型拷贝的参数（list of numpy arrays）
        train_loader: 训练数据 DataLoader
        epsilon: 本轮 ε
        delta: DP δ
        max_grad_norm: 裁剪阈值 C
        lr: 学习率
        epochs: 本地 epoch 数
        device: 计算设备

    Returns:
        dict: {
            "params":         训练后模型参数,
            "loss_before":    训练前 loss,
            "loss_after":     训练后 loss,
            "eps_spent":      Opacus 报告的实际 ε 消耗,
            "params_before":  训练前模型参数 (numpy),
            "num_samples":    训练样本数,
        }
    """
    dataset_size = len(train_loader.dataset)
    sample_rate = train_loader.batch_size / dataset_size
    criterion = nn.CrossEntropyLoss()

    # ---- 创建新模型并加载 master 参数 ----
    local_model = MNISTCNN().to(device)
    set_parameters(local_model, master_params)
    params_before = master_params  # 训练前的参数就是 master_params

    # ---- 训练前 loss ----
    local_model.eval()
    loss_before, n_before = 0.0, 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = local_model(images)
            loss_before += criterion(outputs, labels).item() * images.size(0)
            n_before += images.size(0)
    loss_before /= max(n_before, 1)

    # ---- 预算太小则跳过训练 ----
    if epsilon < 0.02:
        return {
            "params": master_params,
            "loss_before": loss_before,
            "loss_after": loss_before,
            "eps_spent": 0.0,
            "params_before": params_before,
            "num_samples": 0,
        }

    # ---- DP-SGD 训练 ----
    sigma = epsilon_to_sigma(epsilon, delta, sample_rate, epochs=epochs)

    local_model.train()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=lr,
                                momentum=config.MOMENTUM)
    dp_loader = torch.utils.data.DataLoader(
        train_loader.dataset, batch_size=train_loader.batch_size, shuffle=True,
    )

    privacy_engine = PrivacyEngine()
    local_model, optimizer, dp_loader = privacy_engine.make_private(
        module=local_model, optimizer=optimizer, data_loader=dp_loader,
        noise_multiplier=sigma, max_grad_norm=max_grad_norm,
    )

    total_loss, num_samples = 0.0, 0
    for _ in range(epochs):
        for images, labels in dp_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = local_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

    eps_spent = privacy_engine.get_epsilon(delta=delta)

    # ---- 训练后状态 ----
    unwrapped = getattr(local_model, '_module', local_model)
    unwrapped.eval()
    loss_after, n_after = 0.0, 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = unwrapped(images)
            loss_after += criterion(outputs, labels).item() * images.size(0)
            n_after += images.size(0)
    loss_after /= max(n_after, 1)

    new_params = get_parameters(unwrapped)

    return {
        "params": new_params,
        "loss_before": loss_before,
        "loss_after": loss_after,
        "eps_spent": float(eps_spent),
        "params_before": params_before,
        "num_samples": num_samples,
    }


# ============================================================
# 评估
# ============================================================

def evaluate(model, test_loader, device=config.DEVICE):
    """在测试集上评估模型。

    Returns:
        tuple: (accuracy, loss)
    """
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
    acc = correct / total if total > 0 else 0.0
    loss = total_loss / total if total > 0 else 0.0
    return acc, loss


# ============================================================
# 完整训练流程（单个策略）
# ============================================================

def run_training(strategy_name, train_loader, test_loader,
                 total_epsilon=config.TOTAL_EPSILON,
                 max_rounds=config.MAX_ROUNDS,
                 verbose=True):
    """运行完整训练流程。

    对 Uniform 和 KianiLinear：预计算全部 schedule 后逐轮执行。
    对 KianiPlusMomentum：每轮根据累积的 loss 历史动态计算下轮 ε。

    Args:
        strategy_name: "Uniform" | "KianiLinear" | "KianiPlusMomentum"
        train_loader: 训练数据 DataLoader
        test_loader: 测试数据 DataLoader
        total_epsilon: 总隐私预算
        max_rounds: 最大轮数
        verbose: 是否打印每轮进度

    Returns:
        dict: {
            "strategy":        策略名,
            "final_accuracy":  最终准确率,
            "rounds_completed": 实际完成轮数,
            "total_eps_spent": 总 ε 消耗,
            "duration_seconds": 耗时,
            "per_round":       [{round, accuracy, loss, epsilon, momentum, ...}],
            "loss_history":    [每轮 loss],
            "momentum_history": [每轮动量],
        }
    """
    # 主模型：维护为 numpy 参数，每轮 train_one_round 内部创建新模型
    master_params = get_parameters(MNISTCNN())
    cumulative_eps = 0.0
    loss_history = []
    momentum_history = []
    rounds_data = []
    t0 = time.time()

    # 评估用模型
    eval_model = MNISTCNN().to(config.DEVICE)

    # 对于固定调度策略，预计算 schedule
    if strategy_name in ("Uniform", "KianiLinear"):
        fixed_schedule = create_schedule(strategy_name, total_epsilon, max_rounds)
    else:
        fixed_schedule = None

    for rnd in range(1, max_rounds + 1):
        # ---- 确定本轮 ε ----
        if strategy_name == "KianiPlusMomentum":
            # 动态计算
            dynamic_schedule = create_schedule(
                strategy_name, total_epsilon, max_rounds,
                loss_history=loss_history, gamma=config.GAMMA,
            )
            eps_r = dynamic_schedule[rnd - 1]
            # 计算当前动量（取最近窗口）
            momentum = loss_momentum(loss_history) if len(loss_history) >= 3 else 0.0
        else:
            eps_r = fixed_schedule[rnd - 1]
            momentum = 0.0

        momentum_history.append(float(momentum))

        # 不能超花
        if cumulative_eps + eps_r > total_epsilon:
            eps_r = total_epsilon - cumulative_eps
        if eps_r < 0.01:
            eps_r = 0.01

        # ---- 训练一轮 ----
        result = train_one_round(master_params, train_loader, eps_r)
        master_params = result["params"]
        cumulative_eps += result["eps_spent"]

        # ---- 更新 loss 历史 ----
        loss_history.append(result["loss_after"])

        # ---- 评估 ----
        set_parameters(eval_model, master_params)
        acc, test_loss = evaluate(eval_model, test_loader)

        rounds_data.append({
            "round": rnd,
            "accuracy": acc,
            "loss": test_loss,
            "train_loss": result["loss_after"],
            "epsilon_allocated": float(eps_r),
            "epsilon_spent": float(result["eps_spent"]),
            "cumulative_eps": float(cumulative_eps),
            "momentum": float(momentum),
        })

        if verbose:
            print(f"  [{strategy_name:>20}] Round {rnd:2d}: "
                  f"acc={acc:.4f}, ε={eps_r:.3f}, "
                  f"momentum={momentum:+.3f}, Σε={cumulative_eps:.2f}")

        # ---- 早停 ----
        if acc > config.EARLY_STOP_ACC and rnd >= 5:
            if verbose:
                print(f"  [{strategy_name:>20}] Early stop at round {rnd} (acc={acc:.4f})")
            break

    duration = time.time() - t0
    final_acc = rounds_data[-1]["accuracy"] if rounds_data else 0.0

    return {
        "strategy": strategy_name,
        "final_accuracy": final_acc,
        "rounds_completed": len(rounds_data),
        "total_eps_spent": float(cumulative_eps),
        "duration_seconds": duration,
        "per_round": rounds_data,
        "loss_history": [float(l) for l in loss_history],
        "momentum_history": [float(m) for m in momentum_history],
    }
