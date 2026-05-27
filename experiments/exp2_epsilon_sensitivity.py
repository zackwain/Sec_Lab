"""Exp2: ε 敏感度分析实验
研究问题：不同隐私预算 ε 对模型准确率和攻击抵抗能力的影响
"""
from dp_fl.experiment import Experiment, ExperimentConfig


def main():
    epsilons = [0.5, 1.0, 2.0, 4.0, 8.0]
    results = []

    for eps in epsilons:
        cfg = ExperimentConfig(
            experiment_name=f"Exp2_Epsilon_{eps}",
            use_dp=True,
            epsilon=eps,
            dataset="mnist",
            num_clients=10,
            num_rounds=30,
        )
        exp = Experiment(cfg)
        result = exp.run()
        result.save(f"results/logs/{cfg.experiment_name}.json")
        results.append({
            "epsilon": eps,
            "accuracy": result.final_accuracy,
            "mia_auc": result.mia_auc,
        })
        print(f"  ε={eps}: 准确率={result.final_accuracy:.4f}, "
              f"MIA AUC={result.mia_auc:.4f}")

    print("\n=== Exp2 汇总 ===")
    for r in results:
        print(f"ε={r['epsilon']}: Acc={r['accuracy']:.4f}, "
              f"MIA AUC={r['mia_auc']:.4f}")


if __name__ == "__main__":
    main()
