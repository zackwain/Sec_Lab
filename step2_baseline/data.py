"""Step 2: 数据加载与划分"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

np.random.seed(42)
torch.manual_seed(42)


def load_mnist(data_dir="./data"):
    """加载 MNIST 数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return train_set, test_set


def split_data(train_set, num_clients, distribution="iid",
               dirichlet_alpha=0.5, batch_size=64):
    """将训练数据划分给多个客户端"""
    labels = np.array([train_set[i][1] for i in range(len(train_set))])

    if distribution == "iid":
        indices = np.random.permutation(len(train_set))
        split_size = len(train_set) // num_clients
        client_indices = [indices[i * split_size:(i + 1) * split_size].tolist()
                          for i in range(num_clients)]
    elif distribution.startswith("dirichlet"):
        client_indices = _dirichlet_split(labels, num_clients, dirichlet_alpha)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    train_loaders = []
    for indices in client_indices:
        subset = Subset(train_set, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        train_loaders.append(loader)

    return train_loaders, client_indices


def _dirichlet_split(labels, num_clients, alpha):
    """用 Dirichlet 分布划分数据（Non-IID）"""
    n_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for k in range(n_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = np.cumsum(proportions)
        proportions = (proportions * len(idx_k)).astype(int)

        start = 0
        for i in range(num_clients):
            end = proportions[i]
            client_indices[i].extend(idx_k[start:end].tolist())
            start = end

    return client_indices


def get_test_loader(batch_size=256):
    """获取测试数据加载器"""
    _, test_set = load_mnist()
    return DataLoader(test_set, batch_size=batch_size, shuffle=False)


def get_client_labels(train_set, client_indices):
    """获取每个客户端的标签分布（用于 KL 散度计算）"""
    client_labels = []
    for indices in client_indices:
        labels = np.array([train_set[i][1] for i in indices])
        client_labels.append(labels)
    return client_labels
