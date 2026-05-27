from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import flwr as fl
from flwr.common import Context, NDArrays, Scalar, parameters_to_ndarrays
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg

from dp_fl.models import create_model
from dp_fl.client import set_model_params, get_model_params


def weighted_average(metrics: List[Tuple[int, Dict[str, Scalar]]]) \
        -> Dict[str, Scalar]:
    """按客户端样本数加权平均聚合指标"""
    accuracies = [m[1]["accuracy"] * m[0] for m in metrics if "accuracy" in m[1]]
    epsilons = [m[1]["epsilon"] * m[0] for m in metrics if "epsilon" in m[1]]
    losses = [m[1]["loss"] * m[0] for m in metrics if "loss" in m[1]]
    total = sum(m[0] for m in metrics)

    result = {}
    if accuracies and total > 0:
        result["accuracy"] = sum(accuracies) / total
    if epsilons and total > 0:
        result["epsilon"] = sum(epsilons) / total
    if losses and total > 0:
        result["loss"] = sum(losses) / total
    return result


class DPFedAvg(FedAvg):
    """扩展 FedAvg：记录训练过程中的隐私预算"""

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            epsilons = [r.metrics.get("epsilon", 0.0)
                        for _, r in results]
            max_epsilon = max(epsilons) if epsilons else 0.0
            print(f"[Round {server_round}] 最大隐私预算 ε = {max_epsilon:.4f}")
        return aggregated


def gen_server_fn(model: nn.Module, num_rounds: int, min_clients: int):
    """生成 server_fn，供 ServerApp 使用"""

    def server_fn(context: Context) -> ServerAppComponents:
        strategy = DPFedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=min_clients,
            min_evaluate_clients=min_clients,
            min_available_clients=min_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
            initial_parameters=fl.common.ndarrays_to_parameters(
                get_model_params(model)),
        )
        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn


def create_server_app(dataset_name: str, num_rounds: int,
                      min_clients: int) -> ServerApp:
    model = create_model(dataset_name)
    server_fn = gen_server_fn(model, num_rounds, min_clients)
    return ServerApp(server_fn=server_fn)
