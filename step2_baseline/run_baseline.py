"""Step 2: 基线 DP-FL 运行脚本

运行方式：
    python step2_baseline/run_baseline.py

预期输出：
    Round 1: accuracy=0.xx
    Round 2: accuracy=0.xx
    ...
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.insert(0, ".")

import json
import time
import numpy as np
import flwr as fl

from step2_baseline.data import load_mnist, split_data, get_test_loader
from step2_baseline.client import create_client_fn
from step2_baseline.server import create_strategy


def run_baseline(num_clients=10, num_rounds=10, batch_size=64):
    """运行基线联邦学习"""
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

    # 创建策略
    print("\n2. 创建 FedAvg 策略...")
    strategy = create_strategy(num_clients)

    # 创建客户端工厂
    client_fn = create_client_fn(train_loaders, test_loader)

    # 启动训练
    print("\n3. 开始训练...")
    start_time = time.time()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    duration = time.time() - start_time

    # 解析结果
    print("\n4. 训练结果：")
    rounds_data = []
    final_accuracy = 0.0

    if history.metrics_centralized:
        for key, values in history.metrics_centralized.items():
            if key == "accuracy":
                for round_idx, value in enumerate(values, 1):
                    acc = value[1] if isinstance(value, tuple) else value
                    rounds_data.append({
                        "round": round_idx,
                        "accuracy": float(acc),
                    })
                    print(f"   Round {round_idx}: accuracy={acc:.4f}")
                    final_accuracy = float(acc)

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
    run_baseline()
