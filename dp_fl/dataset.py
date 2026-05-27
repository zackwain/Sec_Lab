import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_transform(dataset_name: str):
    if dataset_name == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    elif dataset_name == "cifar10":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def load_dataset(dataset_name: str, data_dir: str = "./data"):
    transform = get_transform(dataset_name)
    if dataset_name == "mnist":
        train_set = datasets.MNIST(data_dir, train=True, download=True,
                                   transform=transform)
        test_set = datasets.MNIST(data_dir, train=False, download=True,
                                  transform=transform)
    elif dataset_name == "cifar10":
        train_set = datasets.CIFAR10(data_dir, train=True, download=True,
                                     transform=transform)
        test_set = datasets.CIFAR10(data_dir, train=False, download=True,
                                    transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return train_set, test_set


def _dirichlet_split(labels, num_clients, alpha):
    """用 Dirichlet 分布划分数据标签，模拟 Non-IID 场景"""
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


def split_data(train_set, num_clients: int, distribution: str = "iid",
               dirichlet_alpha: float = 0.5, batch_size: int = 64):
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


def get_test_loader(dataset_name: str, batch_size: int = 256,
                    data_dir: str = "./data"):
    _, test_set = load_dataset(dataset_name, data_dir)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False)


def get_attack_data(dataset_name: str, data_dir: str = "./data"):
    """为 MIA 攻击准备成员/非成员数据"""
    train_set, test_set = load_dataset(dataset_name, data_dir)
    return train_set, test_set
