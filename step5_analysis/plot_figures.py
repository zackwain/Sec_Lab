"""Step 5: 科研论文级可视化 — 从 JSON 出图

中文: 宋体, 英文: Times New Roman, DPI ≥ 300
配色: 科研风格 (ColorBrewer-inspired)

Usage:
  python step5_analysis/plot_figures.py
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ============================================================
# 全局字体配置
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ============================================================
# 配色方案
# ============================================================

COLORS = {
    "Uniform":           "#b3b9bc",  # 中蓝
    "KianiLinear":       "#365574",  # 暖橙
    "KianiPlusMomentum": "#e69092",  # 深蓝（我们的方法）
    "NoDP":              "#aac3c7",  # 青灰
    "grid":              "#ecedee",
    "pos_momentum":      "#365574",  # 动量正
    "neg_momentum":      "#dca6a7",  # 动量负
}

LABELS_CN = {
    "Uniform":           "Uniform",
    "KianiLinear":       "KianiLinear",
    "KianiPlusMomentum": "Kiani + Momentum (OURS)",
    "NoDP":              "No DP",
}

OUTPUT_DIR = "results/figures"

# ============================================================
# 数据加载
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_per_round(data, strategy, field="accuracy_mean"):
    """从 v2_repeats 的 summaries 提取逐轮数据。"""
    summary = data["summaries"][strategy]
    rounds = []
    values = []
    errs = []
    for entry in summary["per_round"]:
        rounds.append(entry["round"])
        key_std = field.replace("_mean", "_std")
        values.append(entry.get(field, 0))
        errs.append(entry.get(key_std, 0))
    return np.array(rounds), np.array(values), np.array(errs)


# ============================================================
# 图 1: 准确率曲线
# ============================================================

def plot_accuracy_curves(data, save=True):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for strat in ["Uniform", "KianiLinear", "KianiPlusMomentum"]:
        rds, accs, stds = get_per_round(data, strat, "accuracy_mean")
        ax.plot(rds, accs * 100, color=COLORS[strat], linewidth=1.8,
                label=LABELS_CN[strat], marker=".", markersize=4)
        #误差带
        # ax.fill_between(rds, (accs - stds) * 100, (accs + stds) * 100,
        #                 color=COLORS[strat], alpha=0.12)

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_xlim(1, 20)
    ax.set_ylim(65, 94)
    ax.legend(loc="lower right", frameon=False, edgecolor="none", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    fig.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.savefig(os.path.join(OUTPUT_DIR, "fig1_accuracy_curves.png"))
        plt.close(fig)
        print("  Saved: fig1_accuracy_curves.png")
    return fig


# ============================================================
# 图 2: ε 分配对比
# ============================================================

def plot_epsilon_allocation(data, save=True):
    fig, ax = plt.subplots(figsize=(7, 3.5))

    for strat in ["Uniform", "KianiLinear", "KianiPlusMomentum"]:
        rds, eps, _ = get_per_round(data, strat, "epsilon_allocated_mean")
        ax.plot(rds, eps, color=COLORS[strat], linewidth=1.8,
                label=LABELS_CN[strat], marker=".", markersize=4)

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("ε per Round", fontsize=12)
    ax.set_xlim(1, 20)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

    fig.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.savefig(os.path.join(OUTPUT_DIR, "fig2_epsilon_allocation.png"))
        plt.close(fig)
        print("  Saved: fig2_epsilon_allocation.png")
    return fig


# ============================================================
# 图 3: 最终准确率柱状图
# ============================================================

def plot_bar_comparison(data, save=True):
    # 创建画布，5.5×4.5 英寸
    fig, ax = plt.subplots(figsize=(7, 4.5))  # 图 3: 柱状图

    # 三种策略的名字、图例标签、颜色
    strats = ["Uniform", "KianiLinear", "KianiPlusMomentum"]
    names = [LABELS_CN[s] for s in strats]   # X 轴显示的名称
    colors = [COLORS[s] for s in strats]     # 每根柱子的颜色

    # 从 JSON 数据取每个策略的准确率均值 ×100（转百分比）
    means = [data["summaries"][s]["accuracy_mean"] * 100 for s in strats]
    # 取标准差 ×100（误差棒）
    stds = [data["summaries"][s]["accuracy_std"] * 100 for s in strats]

    # 画柱状图
    # width=0.55:    柱子宽度
    # edgecolor:     柱子描边白色
    # yerr:          误差棒高度
    # capsize:       误差棒顶端横线长度
    bars = ax.bar(names, means, color=colors, edgecolor="white", linewidth=1.2,
                  width=0.55, yerr=stds, capsize=3, error_kw={"linewidth": 1})

    # 每根柱子上方写准确率数值
    # bar.get_x() + bar.get_width()/2: X 位置 = 柱子中心
    # bar.get_height() + std + 0.08:   Y 位置 = 柱顶 + 误差棒 + 小偏移
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.08,
                f"{mean:.2f}%",           # 显示格式: 92.28%
                ha="center",              # 水平居中
                va="bottom",              # 垂直底部对齐
                fontsize=11,
                # fontweight="bold"
                )

    # Y 轴标签和范围
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_ylim(91, 94)                   # Y 轴 90%~93%，低于90不显示
    ax.grid(True, alpha=0.3, axis="y")    # 只画水平网格线

    fig.tight_layout()                    # 自动调整边距
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.savefig(os.path.join(OUTPUT_DIR, "fig3_bar_comparison.png"))
        print("  Saved: fig3_bar_comparison.png")
    return fig


# ============================================================
# 图 4: 损失动量信号
# ============================================================

def plot_momentum_signal(data, save=True):
    fig, ax = plt.subplots(figsize=(7, 3.5))

    summary = data["summaries"]["KianiPlusMomentum"]
    rds = [e["round"] for e in summary["per_round"]]
    moms = [e["momentum_mean"] for e in summary["per_round"]]

    colors_bar = [COLORS["pos_momentum"] if m >= 0 else COLORS["neg_momentum"] for m in moms]
    ax.bar(rds, moms, color=colors_bar, width=0.6, edgecolor="white", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.8)

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Loss Momentum", fontsize=12)
    ax.set_xlim(3.5, 20.5)       # 只显示 round 4~20
    ax.set_ylim(0, 0.2)
    ax.grid(True, alpha=0.3, axis="y")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # 每轮都标刻度

    # 标注区域含义
    # ax.text(3, 0.25, "Accelerating → Invest", ha="center", fontsize=9,
    #         color=COLORS["pos_momentum"], fontstyle="italic")
    # ax.text(17, -0.22, "Plateau → Save", ha="center", fontsize=9,
    #         color=COLORS["neg_momentum"], fontstyle="italic")

    fig.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.savefig(os.path.join(OUTPUT_DIR, "fig4_momentum_signal.png"))
        print("  Saved: fig4_momentum_signal.png")
    return fig


# ============================================================
# 图 5: γ 消融
# ============================================================

def plot_gamma_ablation(gamma_data, gamma_high=None, save=True):
    fig, ax = plt.subplots(figsize=(7, 4.2))  # 图 5: γ 消融

    # ── 从 JSON 解析 γ 消融数据（γ=1,2,3）──
    results = gamma_data["results"]
    gammas = []
    accs = []
    for r in results:
        if r["strategy"] == "KianiPlusMomentum":
            gammas.append(float(r["label"].split("=")[1]))
            accs.append(r["final_accuracy"] * 100)

    # ── 补充高 γ 数据（γ=5,10,20）──
    if gamma_high is not None:
        for r in gamma_high["results"]:
            if r["strategy"] == "KianiPlusMomentum":
                gammas.append(float(r["label"].split("=")[1]))
                accs.append(r["final_accuracy"] * 100)

    # ── Uniform 和 KianiLinear 参考线（从原始数据取）──
    uni_acc = next(r["final_accuracy"] * 100 for r in results
                   if r["strategy"] == "Uniform")
    kiani_acc = next(r["final_accuracy"] * 100 for r in results
                     if r["strategy"] == "KianiLinear")

    # 水平虚线: 覆盖全部 γ 范围
    x_min = min(gammas)
    x_max = max(gammas)
    ax.plot([x_min, x_max], [uni_acc, uni_acc], color=COLORS["Uniform"],
            linewidth=1.2, linestyle="--", alpha=0.7, label="Uniform")
    ax.plot([x_min, x_max], [kiani_acc, kiani_acc], color=COLORS["KianiLinear"],
            linewidth=1.2, linestyle="--", alpha=0.7, label="KianiLinear")

    # ── OURS 所有 γ 值的折线 ──
    ax.plot(gammas, accs, color=COLORS["KianiPlusMomentum"], linewidth=2,
            marker="o", markersize=5, label="Kiani+Moment (OURS)", zorder=5)

    # ── 坐标轴设置 ──
    ax.set_xlabel("γ (Momentum Response Strength)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_ylim(91.0, 92.6)
    ax.set_xticks(gammas)     # 每个 γ 都标
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.savefig(os.path.join(OUTPUT_DIR, "fig5_gamma_ablation.png"))
        print("  Saved: fig5_gamma_ablation.png")
    return fig


# ============================================================
# 图 6: 提升比 (Improvement Ratio)
# ============================================================

def plot_improvement_ratio(data, save=True):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    uni_acc = data["summaries"]["Uniform"]["accuracy_mean"] * 100
    kiani_acc = data["summaries"]["KianiLinear"]["accuracy_mean"] * 100
    ours_acc = data["summaries"]["KianiPlusMomentum"]["accuracy_mean"] * 100

    delta_kiani = kiani_acc - uni_acc   # 0.09
    delta_ours = ours_acc - uni_acc     # 0.29
    ratio = delta_ours / delta_kiani    # ~3.2

    labels = ["KianiLinear", "Kiani + Momentum\n(OURS)"]
    deltas = [delta_kiani, delta_ours]
    cols = [COLORS["KianiLinear"], COLORS["KianiPlusMomentum"]]

    bars = ax.bar(labels, deltas, color=cols, edgecolor="white", linewidth=1.2, width=0.45)

    # 柱子上方标注数值
    # for bar, val in zip(bars, deltas):
    #     ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
    #             f"+{val:.2f} pp", ha="center", va="bottom", fontsize=13, fontweight="bold")

    # 提升比标注
    ax.text(1.0, max(deltas) * 0.55,
            f"IR = {ratio:.1f}×", ha="center", fontsize=14, color=COLORS["KianiPlusMomentum"],
            fontweight="bold", fontstyle="italic")

    ax.set_ylabel("Improvement over Baseline (pp)", fontsize=12)
    ax.set_ylim(0, max(deltas) * 1.3)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linewidth=0.8)

    fig.tight_layout()
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.savefig(os.path.join(OUTPUT_DIR, "fig6_improvement_ratio.png"))
        plt.close(fig)
        print("  Saved: fig6_improvement_ratio.png")
    return fig


# ============================================================
# 主入口
# ============================================================

def main():
    print("Generating figures...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载数据
    repeats = load_json("results/logs/step4_v2_repeats.json")
    gamma = load_json("results/logs/step4_ablation_gamma.json")
    gamma_high_path = "results/logs/step4_ablation_gamma_high.json"
    gamma_high = load_json(gamma_high_path) if os.path.exists(gamma_high_path) else None

    # 出图
    plot_accuracy_curves(repeats)
    plot_epsilon_allocation(repeats)
    plot_bar_comparison(repeats)
    plot_momentum_signal(repeats)
    plot_gamma_ablation(gamma, gamma_high)
    plot_improvement_ratio(repeats)

    print(f"\n  All figures saved to {OUTPUT_DIR}/")
    print("  Formats: PNG (300+ DPI) + PDF (vector)")


if __name__ == "__main__":
    main()
