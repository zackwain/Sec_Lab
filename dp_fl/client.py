from collections import OrderedDict
from typing import Dict, Tuple

import torch
import torch.nn as nn
from flwr.common import NDArrays, Scalar, parameters_to_ndarrays
from flwr.client import NumPyClient
from opacus import PrivacyEngine

from dp_fl.models import create_model


def get_model_params(model: nn.Module) -> NDArrays:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_params(model: nn.Module, parameters: NDArrays):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class FlowerDPClient(NumPyClient):
    """带差分隐私保护的 Flower 客户端"""

    def __init__(self, cid: int, model: nn.Module, train_loader,
                 test_loader, device: str, privacy_config: dict):
        self.cid = cid
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.privacy_config = privacy_config

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=privacy_config.get("lr", 0.01),
            momentum=0.9,
            weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()

        # Opacus PrivacyEngine 包裹模型、优化器、数据加载器
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = \
            self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=privacy_config["noise_multiplier"],
                max_grad_norm=privacy_config["max_grad_norm"],
            )

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) \
            -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        set_model_params(self.model, parameters)

        local_epochs = self.privacy_config.get("local_epochs", 1)
        self.model.train()
        total_loss = 0.0
        num_samples = 0

        for _ in range(local_epochs):
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                num_samples += images.size(0)

        epsilon = self.privacy_engine.get_epsilon(
            delta=self.privacy_config["delta"])
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0

        return (get_model_params(self.model), num_samples,
                {"epsilon": epsilon, "loss": avg_loss})

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) \
            -> Tuple[float, int, Dict[str, Scalar]]:
        set_model_params(self.model, parameters)
        self.model.eval()
        correct, total = 0, 0
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += images.size(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        return avg_loss, total, {"accuracy": accuracy}
