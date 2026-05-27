"""Exp4: Non-IID 鲁棒性实验
研究问题：真实非均匀数据分布下 DP-FL 的表现
"""
from dp_fl.experiment import Experiment, ExperimentConfig


def main():
    distributions = [
        ("iid", None),
        ("dirichlet", 1.0),
        ("dirichlet", 0.5),
        ("dirichlet", 0.1),
    ]

    for dist, alpha in distributions:
        name = f"Exp4_{dist}" if alpha is None else f"Exp4_{dist}_{alpha}"
        cfg = ExperimentConfig(
            experiment_name=name,
            use_dp=True,
            epsilon=4.0,
            dataset="mnist",
            num_clients=10,
            num_rounds=30,
            data_distribution=dist,
            dirichlet_alpha=alpha if alpha else 0.5,
        )
        exp = Experiment(cfg)
        result = exp.run()
        result.save(f"results/logs/{cfg.experiment_name}.json")
        label = dist if alpha is None else f"{dist}(α={alpha})"
        print(f"  {label}: 准确率={result.final_accuracy:.4f}")


if __name__ == "__main__":
    main()
