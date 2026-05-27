"""实验结果统一绘图模块
生成全部 5+ 张核心图表到 results/figures/
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

OUTPUT_DIR = "results/figures"
LOG_DIR = "results/logs"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_log(name: str) -> dict:
    path = os.path.join(LOG_DIR, f"{name}.json")
    with open(path, "r") as f:
        return json.load(f)


# ─── 图1: DP 开/关 准确率曲线对比 ───

def plot_baseline_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"NoDP": "#2C5F2F", "DP": "#028090"}
    labels = {"NoDP": "无 DP（基线）", "DP": "DP-SGD (ε=4)"}

    for key, name in [("NoDP", "Exp1_Baseline_NoDP"), ("DP", "Exp1_Baseline_DP")]:
        try:
            data = load_log(name)
            rounds = [r["round"] for r in data["rounds"]]
            accs = [r["accuracy"] * 100 for r in data["rounds"]]
            ax.plot(rounds, accs, color=colors[key], linewidth=2,
                    label=labels[key], marker="o", markersize=4, markevery=3)
        except FileNotFoundError:
            print(f"  [跳过] {name}.json 未找到")

    ax.set_xlabel("通信轮次", fontsize=13)
    ax.set_ylabel("准确率 (%)", fontsize=13)
    ax.set_title("DP 对联邦学习准确率的影响", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig1_baseline_comparison.png")
    plt.close(fig)
    print("  ✓ fig1_baseline_comparison.png")


# ─── 图2: ε-Accuracy 权衡曲线 ───

def plot_epsilon_accuracy_tradeoff():
    epsilons = [0.5, 1.0, 2.0, 4.0, 8.0]
    accuracies = []
    mia_aucs = []

    for eps in epsilons:
        try:
            data = load_log(f"Exp2_Epsilon_{eps}")
            accuracies.append(data["final_accuracy"] * 100)
            mia_aucs.append(data.get("mia_auc", 0))
        except FileNotFoundError:
            accuracies.append(0)
            mia_aucs.append(0)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    color1 = "#028090"
    color2 = "#990011"

    line1, = ax1.plot(epsilons, accuracies, color=color1, linewidth=2.5,
                      marker="s", markersize=8, label="准确率 (%)")
    line2, = ax2.plot(epsilons, mia_aucs, color=color2, linewidth=2.5,
                      marker="^", markersize=8, linestyle="--",
                      label="MIA AUC")

    ax1.set_xlabel("隐私预算 ε", fontsize=13)
    ax1.set_ylabel("准确率 (%)", fontsize=13, color=color1)
    ax2.set_ylabel("MIA AUC", fontsize=13, color=color2)
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="随机猜测 (0.5)")
    ax1.set_title("隐私-效用权衡：ε 对准确率和攻击抵抗的影响",
                  fontsize=15, fontweight="bold")

    lines = [line1, line2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right",
               fontsize=10)

    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig2_epsilon_accuracy_tradeoff.png")
    plt.close(fig)
    print("  ✓ fig2_epsilon_accuracy_tradeoff.png")


# ─── 图3: MNIST vs CIFAR-10 对比 ───

def plot_dataset_comparison():
    fig, ax = plt.subplots(figsize=(8, 5))

    datasets = [("MNIST", "Exp3_MNIST", "#028090"),
                ("CIFAR-10", "Exp3_CIFAR10", "#990011")]

    for label, name, color in datasets:
        try:
            data = load_log(name)
            rounds = [r["round"] for r in data["rounds"]]
            accs = [r["accuracy"] * 100 for r in data["rounds"]]
            ax.plot(rounds, accs, color=color, linewidth=2, label=label,
                    marker="o", markersize=4, markevery=3)
        except FileNotFoundError:
            print(f"  [跳过] {name}.json 未找到")

    ax.set_xlabel("通信轮次", fontsize=13)
    ax.set_ylabel("准确率 (%)", fontsize=13)
    ax.set_title("DP-FL 在不同数据集上的表现 (ε=4)", fontsize=15,
                 fontweight="bold")
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig3_dataset_comparison.png")
    plt.close(fig)
    print("  ✓ fig3_dataset_comparison.png")


# ─── 图4: Non-IID 鲁棒性 ───

def plot_non_iid_robustness():
    configs = [
        ("IID", "Exp4_iid"),
        ("Dirichlet\nα=1.0", "Exp4_dirichlet_1.0"),
        ("Dirichlet\nα=0.5", "Exp4_dirichlet_0.5"),
        ("Dirichlet\nα=0.1", "Exp4_dirichlet_0.1"),
    ]

    labels = []
    accuracies = []
    colors = ["#028090", "#00A896", "#F96167", "#990011"]

    for label, name in configs:
        try:
            data = load_log(name)
            labels.append(label)
            accuracies.append(data["final_accuracy"] * 100)
        except FileNotFoundError:
            labels.append(label)
            accuracies.append(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(labels)), accuracies, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", fontsize=12, fontweight="bold")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("准确率 (%)", fontsize=13)
    ax.set_title("Non-IID 数据分布对 DP-FL 的影响 (ε=4)",
                 fontsize=15, fontweight="bold")
    ax.set_ylim(0, max(accuracies) * 1.15 if max(accuracies) > 0 else 100)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig4_non_iid_robustness.png")
    plt.close(fig)
    print("  ✓ fig4_non_iid_robustness.png")


# ─── 图5: 梯度裁剪阈值 C 影响 ───

def plot_clip_threshold():
    clip_values = [0.1, 0.5, 1.0, 5.0, 10.0]
    accuracies = []

    for c in clip_values:
        try:
            data = load_log(f"Exp5_Clip_C{c}")
            accuracies.append(data["final_accuracy"] * 100)
        except FileNotFoundError:
            accuracies.append(0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(clip_values, accuracies, color="#028090", linewidth=2.5,
            marker="D", markersize=10, markerfacecolor="white")
    ax.set_xlabel("梯度裁剪阈值 C", fontsize=13)
    ax.set_ylabel("准确率 (%)", fontsize=13)
    ax.set_title("梯度裁剪阈值 C 对 DP-FL 准确率的影响 (ε=4)",
                 fontsize=15, fontweight="bold")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig5_clip_threshold.png")
    plt.close(fig)
    print("  ✓ fig5_clip_threshold.png")


# ─── 主函数 ───

def main():
    ensure_output_dir()
    print("生成实验图表...\n")
    plot_baseline_comparison()
    plot_epsilon_accuracy_tradeoff()
    plot_dataset_comparison()
    plot_non_iid_robustness()
    plot_clip_threshold()
    print(f"\n全部图表保存至 {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
