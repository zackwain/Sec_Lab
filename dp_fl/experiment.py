import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

import torch
import flwr as fl
from flwr.client import ClientApp

from dp_fl.models import create_model
from dp_fl.dataset import load_dataset, split_data, get_test_loader
from dp_fl.client import FlowerDPClient, get_model_params
from dp_fl.server import create_server_app


@dataclass
class ExperimentConfig:
    dataset: str = "mnist"
    num_clients: int = 10
    num_rounds: int = 30
    local_epochs: int = 1
    epsilon: float = 4.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: Optional[float] = None
    data_distribution: str = "iid"
    dirichlet_alpha: float = 0.5
    batch_size: int = 64
    lr: float = 0.01
    use_dp: bool = True
    device: str = "cpu"
    experiment_name: str = "experiment"
    seed: int = 42


@dataclass
class RoundResult:
    round: int
    accuracy: float
    loss: float
    epsilon: float
    timestamp: float


@dataclass
class ExperimentResult:
    config: dict
    rounds: List[RoundResult] = field(default_factory=list)
    final_accuracy: float = 0.0
    final_epsilon: float = 0.0
    mia_auc: float = 0.0
    mia_accuracy: float = 0.0
    inversion_psnr: float = 0.0
    inversion_ssim: float = 0.0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "rounds": [asdict(r) for r in self.rounds],
            "final_accuracy": self.final_accuracy,
            "final_epsilon": self.final_epsilon,
            "mia_auc": self.mia_auc,
            "mia_accuracy": self.mia_accuracy,
            "inversion_psnr": self.inversion_psnr,
            "inversion_ssim": self.inversion_ssim,
            "duration_seconds": self.duration_seconds,
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def calculate_noise_multiplier(epsilon: float, delta: float, epochs: int,
                               dataset_size: int, batch_size: int,
                               max_grad_norm: float) -> float:
    """给定目标 (ε, δ)，估算所需的噪声乘数 σ。

    Opacus 的 RDP 会计会自动追踪实际 ε，此函数提供一个初始估计。
    """
    samples_per_client = dataset_size
    steps_per_epoch = samples_per_client // batch_size
    total_steps = epochs * steps_per_epoch

    # 使用 Abadi et al. 2016 的公式粗略估计
    # σ ≈ sqrt(2 * log(1.25/delta)) / ε  * (clipping_norm * total_steps^0.5 近似)
    q = batch_size / samples_per_client  # sampling ratio
    # 简化估计：σ ∝ 1/ε
    import math
    sigma_base = math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    # 在 DP-FL 场景中取一个合理的缩放
    sigma = sigma_base * 0.5

    return max(sigma, 0.1)  # 至少 0.1，避免噪声过小


class Experiment:
    """实验编排器：配置 → 训练 → 评估 → 攻击 → 保存结果"""

    def __init__(self, config: ExperimentConfig):
        self.config = config

        if config.device == "auto":
            config.device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(config.seed)

        # 加载数据
        train_set, test_set = load_dataset(config.dataset)
        sample_size = len(train_set) // config.num_clients
        self.train_loaders, self.client_indices = split_data(
            train_set, config.num_clients, config.data_distribution,
            config.dirichlet_alpha, config.batch_size
        )
        self.test_loader = get_test_loader(config.dataset)

        # 计算噪声乘数（如果未指定）
        if config.use_dp and config.noise_multiplier is None:
            config.noise_multiplier = calculate_noise_multiplier(
                config.epsilon, config.delta, config.local_epochs,
                sample_size, config.batch_size, config.max_grad_norm
            )

    def client_fn(self, cid: int) -> ClientApp:
        model = create_model(self.config.dataset)

        if self.config.use_dp:
            privacy_config = {
                "noise_multiplier": self.config.noise_multiplier,
                "max_grad_norm": self.config.max_grad_norm,
                "delta": self.config.delta,
                "local_epochs": self.config.local_epochs,
                "lr": self.config.lr,
            }
        else:
            privacy_config = {
                "noise_multiplier": 0.0,     # 不加噪声
                "max_grad_norm": float("inf"),  # 不裁剪
                "delta": self.config.delta,
                "local_epochs": self.config.local_epochs,
                "lr": self.config.lr,
            }

        client = FlowerDPClient(
            cid=cid,
            model=model,
            train_loader=self.train_loaders[cid],
            test_loader=self.test_loader,
            device=self.config.device,
            privacy_config=privacy_config,
        )
        return client.to_client()

    def run(self) -> ExperimentResult:
        """执行完整的联邦学习训练"""
        print(f"\n{'='*60}")
        print(f"实验: {self.config.experiment_name}")
        print(f"数据集: {self.config.dataset}, 客户端: {self.config.num_clients}")
        print(f"DP: {self.config.use_dp}, ε_target: {self.config.epsilon}")
        print(f"数据分布: {self.config.data_distribution}")
        print(f"设备: {self.config.device}")
        print(f"{'='*60}\n")

        result = ExperimentResult(config=asdict(self.config))
        start_time = time.time()

        # 创建 ServerApp
        server_app = create_server_app(
            self.config.dataset, self.config.num_rounds,
            self.config.num_clients
        )

        # 启动 Flower 仿真（使用 run_simulation 需要 client_fn）
        # 使用 start_client 方式逐个启动
        history = fl.simulation.start_simulation(
            client_fn=self.client_fn,
            num_clients=self.config.num_clients,
            config=fl.server.ServerConfig(num_rounds=self.config.num_rounds),
            strategy=server_app._strategy,
        )

        # 解析训练历史
        if history.metrics_centralized:
            for round_idx, (loss, metrics) in enumerate(
                    history.metrics_centralized["accuracy"], 1):
                acc = metrics if isinstance(metrics, (int, float)) else 0.0
                result.rounds.append(RoundResult(
                    round=round_idx,
                    accuracy=acc,
                    loss=0.0,
                    epsilon=0.0,
                    timestamp=time.time() - start_time,
                ))
                result.final_accuracy = acc

        result.duration_seconds = time.time() - start_time
        print(f"\n实验完成: 准确率 = {result.final_accuracy:.4f}, "
              f"耗时 = {result.duration_seconds:.1f}s")

        return result
