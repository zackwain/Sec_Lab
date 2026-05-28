"""Step 2: FedAvg 服务器"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import flwr as fl
from typing import List, Tuple, Dict
from flwr.common.typing import Scalar

import sys
sys.path.insert(0, ".")
from step2_baseline.model import MNISTCNN, get_parameters


def weighted_average(metrics):
    """按客户端样本数加权平均"""
    accuracies = [m["accuracy"] * n for n, m in metrics]
    losses = [m.get("loss", 0) * n for n, m in metrics]
    total = sum(n for n, _ in metrics)
    result = {}
    if total > 0:
        result["accuracy"] = sum(accuracies) / total
        result["loss"] = sum(losses) / total
    return result


def create_strategy(num_clients):
    """创建 FedAvg 策略"""
    model = MNISTCNN()
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(
            get_parameters(model)),
    )
    return strategy
