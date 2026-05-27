"""Exp3: 数据集对比实验 (MNIST vs CIFAR-10)
研究问题：DP-FL 在不同复杂度数据集上的表现差异
"""
from dp_fl.experiment import Experiment, ExperimentConfig


def main():
    configs = [
        ExperimentConfig(
            experiment_name="Exp3_MNIST",
            use_dp=True,
            epsilon=4.0,
            dataset="mnist",
            num_clients=10,
            num_rounds=30,
        ),
        ExperimentConfig(
            experiment_name="Exp3_CIFAR10",
            use_dp=True,
            epsilon=4.0,
            dataset="cifar10",
            num_clients=10,
            num_rounds=30,
        ),
    ]

    for cfg in configs:
        exp = Experiment(cfg)
        result = exp.run()
        result.save(f"results/logs/{cfg.experiment_name}.json")
        print(f"  {cfg.dataset}: 准确率={result.final_accuracy:.4f}, "
              f"ε={result.final_epsilon:.4f}")


if __name__ == "__main__":
    main()
