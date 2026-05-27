"""Exp1: DP 开/关基线对比实验
研究问题：DP 能带来多少隐私保护？以多大精度代价？
"""
from dp_fl.experiment import Experiment, ExperimentConfig


def main():
    configs = [
        ExperimentConfig(
            experiment_name="Exp1_Baseline_NoDP",
            use_dp=False,
            dataset="mnist",
            num_clients=10,
            num_rounds=30,
        ),
        ExperimentConfig(
            experiment_name="Exp1_Baseline_DP",
            use_dp=True,
            epsilon=4.0,
            dataset="mnist",
            num_clients=10,
            num_rounds=30,
        ),
    ]

    for cfg in configs:
        exp = Experiment(cfg)
        result = exp.run()
        result.save(f"results/logs/{cfg.experiment_name}.json")
        print(f"  准确率: {result.final_accuracy:.4f}")
        print(f"  ε: {result.final_epsilon:.4f}")


if __name__ == "__main__":
    main()
