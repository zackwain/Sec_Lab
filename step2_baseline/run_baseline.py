"""Step 2: 基线联邦学习运行脚本

运行方式：
    python step2_baseline/run_baseline.py

说明：
    使用自实现的简单 simulation，不依赖 ray，避免 DLL 冲突。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, ".")

import torch
import numpy as np
import json
import time

from step2_baseline.model import MNISTCNN, get_parameters, set_parameters
from step2_baseline.data import load_mnist, split_data, get_test_loader


def train_one_client(model, train_loader, device="cpu", lr=0.01):
    """客户端本地训练"""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    num_samples = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        num_samples += images.size(0)

    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    return get_parameters(model), num_samples, avg_loss


def evaluate_model(model, test_loader, device="cpu"):
    """评估模型"""
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += images.size(0)

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return accuracy, avg_loss


def fedavg_aggregate(client_updates):
    """FedAvg 聚合：按样本数加权平均"""
    total_samples = sum(n for _, n, _ in client_updates)
    aggregated = None

    for params, num_samples, _ in client_updates:
        weight = num_samples / total_samples
        if aggregated is None:
            aggregated = [p * weight for p in params]
        else:
            for i in range(len(aggregated)):
                aggregated[i] = aggregated[i] + params[i] * weight

    return aggregated


def run_simulation(num_clients=10, num_rounds=10, batch_size=64):
    """运行基线联邦学习（简单 simulation，不用 ray）"""
    print("=" * 60)
    print("Step 2: 基线联邦学习 (无 DP)")
    print(f"客户端: {num_clients}, 轮数: {num_rounds}")
    print("=" * 60)

    # 加载数据
    print("\n1. 加载 MNIST 数据...")
    train_set, test_set = load_mnist()
    train_loaders, client_indices = split_data(
        train_set, num_clients, "iid", batch_size=batch_size
    )
    test_loader = get_test_loader()
    print(f"   训练集: {len(train_set)}, 测试集: {len(test_set)}")
    print(f"   每个客户端: ~{len(train_set) // num_clients} 样本")

    # 创建全局模型
    print("\n2. 开始训练...")
    global_model = MNISTCNN()
    device = "cpu"
    start_time = time.time()

    rounds_data = []

    for round_idx in range(1, num_rounds + 1):
        # 每轮：所有客户端本地训练
        client_updates = []
        for cid in range(num_clients):
            # 创建本地模型，加载全局参数
            local_model = MNISTCNN()
            set_parameters(local_model, get_parameters(global_model))

            # 本地训练
            params, num_samples, loss = train_one_client(
                local_model, train_loaders[cid], device
            )
            client_updates.append((params, num_samples, loss))

        # FedAvg 聚合
        aggregated_params = fedavg_aggregate(client_updates)
        set_parameters(global_model, aggregated_params)

        # 评估全局模型
        accuracy, loss = evaluate_model(global_model, test_loader, device)

        rounds_data.append({
            "round": round_idx,
            "accuracy": accuracy,
            "loss": loss,
        })
        print(f"   Round {round_idx}/{num_rounds}: accuracy={accuracy:.4f}, loss={loss:.4f}")

    duration = time.time() - start_time
    final_accuracy = rounds_data[-1]["accuracy"] if rounds_data else 0.0

    # 保存结果
    result = {
        "method": "baseline_no_dp",
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "final_accuracy": final_accuracy,
        "duration_seconds": duration,
        "rounds": rounds_data,
    }

    os.makedirs("results/logs", exist_ok=True)
    with open("results/logs/step2_baseline.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n   最终准确率: {final_accuracy:.4f}")
    print(f"   耗时: {duration:.1f}s")
    print(f"   结果保存至: results/logs/step2_baseline.json")

    print("\n" + "=" * 60)
    print("✓ 基线 FL 训练完成！")
    print("=" * 60)
    print("\n下一步: 运行 Step 3 自适应分配")
    print("命令: python step3_adaptive/run_adaptive.py")

    return result


if __name__ == "__main__":
    run_simulation()
