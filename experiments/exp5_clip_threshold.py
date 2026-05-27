"""Exp5: 梯度裁剪阈值 C 的影响实验
研究问题：不同梯度裁剪阈值对 DP-FL 收敛的影响
"""
from dp_fl.experiment import Experiment, ExperimentConfig


def main():
    clip_values = [0.1, 0.5, 1.0, 5.0, 10.0]

    for c in clip_values:
        cfg = ExperimentConfig(
            experiment_name=f"Exp5_Clip_C{c}",
            use_dp=True,
            epsilon=4.0,
            max_grad_norm=c,
            dataset="mnist",
            num_clients=10,
            num_rounds=30,
        )
        exp = Experiment(cfg)
        result = exp.run()
        result.save(f"results/logs/{cfg.experiment_name}.json")
        print(f"  C={c}: 准确率={result.final_accuracy:.4f}")


if __name__ == "__main__":
    main()
