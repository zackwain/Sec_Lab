"""Step 2: Flower 客户端（均匀 ε 分配）"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from flwr.common.typing import NDArrays, Scalar, Config, Dict
import flwr as fl
from collections import OrderedDict

import sys
sys.path.insert(0, ".")
from step2_baseline.model import MNISTCNN, get_parameters, set_parameters


class FLClient(fl.client.NumPyClient):
    """Flower 客户端：本地训练 + 评估"""
    def __init__(self, train_loader, test_loader, device="cpu"):
        self.model = MNISTCNN().to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        total_loss = 0.0
        num_samples = 0
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        return get_parameters(self.model), num_samples, {"loss": avg_loss}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()
        correct, total = 0, 0
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += images.size(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        return avg_loss, total, {"accuracy": accuracy}


def create_client_fn(train_loaders, test_loader, device="cpu"):
    """创建客户端工厂函数"""
    def client_fn(cid):
        client = FLClient(
            train_loader=train_loaders[int(cid)],
            test_loader=test_loader,
            device=device,
        )
        return client.to_client()
    return client_fn
