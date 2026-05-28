"""Step 1: 验证 Flower 1.7.0 API 可用性

这是一个最小化的测试脚本，验证：
1. Flower 1.7.0 能正常导入
2. fl.simulation.start_simulation() 能正常运行
3. 客户端的基本功能（get_parameters, fit, evaluate）能调用

预期输出：✓ Flower API works!
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import flwr as fl
from flwr.common.typing import NDArrays, Scalar
import numpy as np


class SimpleClient(fl.client.NumPyClient):
    """最简单的 Flower 客户端，用于验证 API"""

    def get_parameters(self, config):
        """返回模型参数"""
        return [np.array([1.0, 2.0, 3.0], dtype=np.float32)]

    def fit(self, parameters: NDArrays, config):
        """本地训练（模拟）"""
        # 模拟训练：参数稍微更新
        updated_params = [p + 0.1 for p in parameters]
        # 返回：更新后的参数, 样本数, 指标
        return updated_params, 100, {"loss": 0.5}

    def evaluate(self, parameters: NDArrays, config):
        """评估模型（模拟）"""
        # 返回：loss, 样本数, 指标
        return 0.3, 100, {"accuracy": 0.8}


def test_flower_simulation():
    """测试 Flower simulation API"""
    print("=" * 60)
    print("Step 1: 验证 Flower 1.7.0 API")
    print("=" * 60)

    print("\n1. 导入 flwr...")
    print(f"   ✓ Flower 版本: {fl.__version__}")

    print("\n2. 启动 simulation (2 个客户端, 1 轮)...")

    try:
        history = fl.simulation.start_simulation(
            client_fn=lambda cid: SimpleClient(),
            num_clients=2,
            config=fl.server.ServerConfig(num_rounds=1),
        )
        print("   ✓ Simulation 启动成功")

        print("\n3. 检查训练历史...")
        if history.losses_centralized:
            print(f"   ✓ 记录了 {len(history.losses_centralized)} 轮 loss")
            print(f"   ✓ 最后一轮 loss: {history.losses_centralized[-1]}")

        if history.metrics_centralized:
            print(f"   ✓ 记录了 {len(history.metrics_centralized)} 轮 metrics")
            for key, values in history.metrics_centralized.items():
                print(f"   ✓ {key}: {values[-1]}")

        return True

    except Exception as e:
        print(f"   ✗ 错误: {e}")
        return False


if __name__ == "__main__":
    success = test_flower_simulation()

    print("\n" + "=" * 60)
    if success:
        print("✓ Flower API 验证通过！")
        print("=" * 60)
        print("\n下一步：运行 Step 2 基线 DP-FL")
        print("命令: python step2_baseline/run_baseline.py")
    else:
        print("✗ Flower API 验证失败，请检查依赖安装")
        print("=" * 60)
