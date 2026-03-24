# -*- coding: utf-8 -*-
"""
分类任务高阶可视化：5 种全新图表
  图1: 跨任务跨模型准确率对比总览 (分组柱状图)
  图2: 多分类 ROC 曲线 + AUC (每个实验一张)
  图3: 归一化混淆矩阵 (百分比版，消除类别不平衡视觉偏差)
  图4: 多维度雷达图 (Accuracy/Precision/Recall/F1 四维对比)
  图5: F1-Score 热力图 (跨任务 × 跨模型 × 跨类别 全景总览)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
from pathlib import Path
from math import pi
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR / "Time_Series_Library" / "results"
OUT_DIR = CURRENT_DIR.parent / "输出结果" / "分类任务效果图_高阶"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_CONFIG = {
    "LondonBikeClass": {
        "class_names": ['边缘冷门站', '傍晚休闲站', '早高峰潮汐站', '双峰通勤站'],
        "task_title": "常规站点功能分类",
        "short": "常规功能",
    },
    "LondonBikePandemic": {
        "class_names": ['普通缩水区', '刚需避风港', '永恒长尾区', '脆弱通勤王'],
        "task_title": "疫情抗压演变分类",
        "short": "疫情抗压",
    },
    "LondonBikeNetFlow": {
        "class_names": ['清晨流出型', '自平衡型', '清晨汇聚型'],
        "task_title": "潮汐净流量分类",
        "short": "潮汐净流量",
    },
}


def load_all_experiments():
    """扫描 results 目录，加载所有分类实验的预测结果"""
    if not RESULTS_DIR.exists():
        print("[!] 未找到 results 目录")
        return []

    experiments = []
    exp_folders = [f for f in os.listdir(RESULTS_DIR) if f.startswith("classification_")]

    for exp in exp_folders:
        exp_path = RESULTS_DIR / exp
        pred_file = exp_path / "pred.npy"
        true_file = exp_path / "true.npy"

        if not (pred_file.exists() and true_file.exists()):
            continue

        model = "TimesNet" if "TimesNet" in exp else \
                "NS-Transformer" if "Nonstationary_Transformer" in exp else None
        if model is None:
            continue

        task_key = None
        for key in TASK_CONFIG:
            if key in exp:
                task_key = key
                break
        if task_key is None:
            continue

        raw_preds = np.load(pred_file)
        trues = np.load(true_file).reshape(-1).astype(int)

        if raw_preds.ndim > 1 and raw_preds.shape[1] > 1:
            probs = softmax(raw_preds, axis=1)
            preds = np.argmax(probs, axis=1)
        else:
            probs = None
            preds = raw_preds.reshape(-1).astype(int)

        config = TASK_CONFIG[task_key]
        acc = accuracy_score(trues, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            trues, preds, labels=range(len(config["class_names"])), zero_division=0
        )

        experiments.append({
            "folder": exp,
            "task_key": task_key,
            "model": model,
            "preds": preds,
            "trues": trues,
            "probs": probs,
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "config": config,
        })
        print(f"  已加载: [{config['short']}] {model}  Acc={acc*100:.2f}%")

    return experiments


# =====================================================================
# 图1: 跨任务跨模型准确率对比总览
# =====================================================================
def plot_accuracy_comparison(experiments):
    """分组柱状图：横轴是三大任务，每组两根柱子（两个模型）"""
    task_order = ["LondonBikeClass", "LondonBikePandemic", "LondonBikeNetFlow"]
    model_order = ["TimesNet", "NS-Transformer"]
    colors = {"TimesNet": "#3498db", "NS-Transformer": "#e67e22"}

    acc_map = {}
    for e in experiments:
        acc_map[(e["task_key"], e["model"])] = e["accuracy"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(task_order))
    width = 0.32

    for i, model in enumerate(model_order):
        vals = [acc_map.get((t, model), 0) * 100 for t in task_order]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model, color=colors[model],
                       edgecolor='white', linewidth=1.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{v:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    task_labels = [TASK_CONFIG[t]["short"] for t in task_order]
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=13)
    ax.set_ylabel('准确率 (%)', fontsize=13)
    ax.set_ylim(0, 100)
    ax.set_title('三大分类任务 × 两大模型：准确率总览对比', fontsize=17, fontweight='bold', pad=15)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> 图1 已生成: 01_accuracy_comparison.png")


# =====================================================================
# 图2: 多分类 ROC 曲线 + AUC
# =====================================================================
def plot_roc_curves(experiments):
    """每个实验绘制一张多分类 One-vs-Rest ROC 曲线图"""
    for e in experiments:
        if e["probs"] is None:
            continue

        config = e["config"]
        n_classes = len(config["class_names"])
        trues_bin = label_binarize(e["trues"], classes=range(n_classes))

        fig, ax = plt.subplots(figsize=(8, 7))
        colors_roc = plt.cm.Set2(np.linspace(0, 1, n_classes))

        all_fpr = np.linspace(0, 1, 200)
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(trues_bin[:, i], e["probs"][:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors_roc[i], linewidth=2,
                    label=f'{config["class_names"][i]}  (AUC={roc_auc:.3f})')
            mean_tpr += np.interp(all_fpr, fpr, tpr)

        mean_tpr /= n_classes
        mean_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, color='navy', linewidth=2.5, linestyle='--',
                label=f'Macro 平均  (AUC={mean_auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.set_xlabel('False Positive Rate (假阳率)', fontsize=12)
        ax.set_ylabel('True Positive Rate (真阳率)', fontsize=12)
        ax.set_title(f'多分类 ROC 曲线\n{config["task_title"]} — {e["model"]}',
                      fontsize=15, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.4)

        safe_name = f'02_ROC_{e["task_key"]}_{e["model"].replace("-", "")}.png'
        plt.tight_layout()
        plt.savefig(OUT_DIR / safe_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> 图2 已生成: {safe_name}")


# =====================================================================
# 图3: 归一化混淆矩阵 (百分比版)
# =====================================================================
def plot_normalized_confusion_matrices(experiments):
    """百分比混淆矩阵，每行归一化为100%，直观展示每个类别的识别率和混淆去向"""
    for e in experiments:
        config = e["config"]
        cm = confusion_matrix(e["trues"], e["preds"])
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100

        fig, ax = plt.subplots(figsize=(9, 7))
        cmap = 'Blues' if e["model"] == "TimesNet" else 'Oranges'

        sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap=cmap, ax=ax,
                    xticklabels=config["class_names"],
                    yticklabels=config["class_names"],
                    annot_kws={"size": 13},
                    vmin=0, vmax=100,
                    cbar_kws={'label': '百分比 (%)', 'shrink': 0.8})

        ax.set_title(f'归一化混淆矩阵 (行百分比)\n{config["task_title"]} — {e["model"]}',
                      fontsize=15, fontweight='bold')
        ax.set_xlabel('预测类别', fontsize=13)
        ax.set_ylabel('真实类别', fontsize=13)
        ax.tick_params(axis='x', rotation=15)

        safe_name = f'03_NormCM_{e["task_key"]}_{e["model"].replace("-", "")}.png'
        plt.tight_layout()
        plt.savefig(OUT_DIR / safe_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> 图3 已生成: {safe_name}")


# =====================================================================
# 图4: 多维度雷达图 (模型综合能力对比)
# =====================================================================
def plot_radar_comparison(experiments):
    """每个任务一张雷达图，在同一张图上对比两个模型的 Acc/Macro-P/Macro-R/Macro-F1"""
    task_order = ["LondonBikeClass", "LondonBikePandemic", "LondonBikeNetFlow"]
    model_colors = {"TimesNet": "#3498db", "NS-Transformer": "#e67e22"}

    for task_key in task_order:
        task_exps = [e for e in experiments if e["task_key"] == task_key]
        if not task_exps:
            continue

        config = TASK_CONFIG[task_key]
        metrics_labels = ['Accuracy', 'Macro-Precision', 'Macro-Recall', 'Macro-F1']
        N = len(metrics_labels)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(30)
        plt.xticks(angles[:-1], metrics_labels, size=12)
        plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0], ["60%", "70%", "80%", "90%", "100%"],
                   color="grey", size=9)
        plt.ylim(0.5, 1.05)

        for e in task_exps:
            values = [
                e["accuracy"],
                np.mean(e["precision"]),
                np.mean(e["recall"]),
                np.mean(e["f1"]),
            ]
            values += values[:1]
            ax.plot(angles, values, linewidth=2.2, linestyle='solid',
                    label=e["model"], color=model_colors[e["model"]])
            ax.fill(angles, values, color=model_colors[e["model"]], alpha=0.12)

        ax.set_title(f'{config["task_title"]}\n模型综合能力雷达图', fontsize=15,
                      fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=11)

        safe_name = f'04_Radar_{task_key}.png'
        plt.tight_layout()
        plt.savefig(OUT_DIR / safe_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> 图4 已生成: {safe_name}")


# =====================================================================
# 图5: F1-Score 热力图 (跨任务 × 跨模型 × 跨类别 全景总览)
# =====================================================================
def plot_f1_heatmap(experiments):
    """一张大热力图，行是「任务-模型」组合，列是各类别，色深代表F1高低"""
    row_labels = []
    f1_matrix = []

    all_class_names = []
    for task_key in ["LondonBikeClass", "LondonBikePandemic", "LondonBikeNetFlow"]:
        for model in ["TimesNet", "NS-Transformer"]:
            matched = [e for e in experiments
                       if e["task_key"] == task_key and e["model"] == model]
            if not matched:
                continue
            e = matched[0]
            config = TASK_CONFIG[task_key]
            label = f'{config["short"]} | {model}'
            row_labels.append(label)

            f1_row = list(e["f1"])
            max_classes = 4
            while len(f1_row) < max_classes:
                f1_row.append(np.nan)
            f1_matrix.append(f1_row)

            if len(config["class_names"]) > len(all_class_names):
                all_class_names = config["class_names"]

    if not f1_matrix:
        return

    col_labels = [f'类别 {i}' for i in range(len(f1_matrix[0]))]
    f1_arr = np.array(f1_matrix)

    fig, ax = plt.subplots(figsize=(10, max(5, len(row_labels) * 0.8 + 2)))

    mask = np.isnan(f1_arr)
    sns.heatmap(f1_arr, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax,
                xticklabels=col_labels, yticklabels=row_labels,
                annot_kws={"size": 12}, mask=mask,
                vmin=0.5, vmax=1.0,
                cbar_kws={'label': 'F1-Score', 'shrink': 0.7},
                linewidths=1.5, linecolor='white')

    ax.set_title('全景 F1-Score 热力图\n(行 = 任务×模型, 列 = 类别编号)',
                  fontsize=16, fontweight='bold', pad=15)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=11)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_F1_Heatmap_Panorama.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  -> 图5 已生成: 05_F1_Heatmap_Panorama.png")


# =====================================================================
# 主函数
# =====================================================================
def main():
    print("=" * 60)
    print("  分类任务高阶可视化生成器 (5 种全新图表)")
    print("=" * 60)

    experiments = load_all_experiments()
    if not experiments:
        print("\n[!] 未找到任何有效的分类实验结果 (需要 pred.npy + true.npy)")
        print("    请先运行分类训练脚本 (4_3, 4_7, 4_8, 4_9, 5_4, 5_5)，")
        print("    确保 Time_Series_Library/results/ 下生成了 pred.npy 和 true.npy")
        return

    print(f"\n  共加载 {len(experiments)} 个实验结果，开始绘制...\n")

    print("[图1] 跨任务跨模型准确率对比总览...")
    plot_accuracy_comparison(experiments)

    print("[图2] 多分类 ROC 曲线 + AUC...")
    plot_roc_curves(experiments)

    print("[图3] 归一化混淆矩阵 (百分比版)...")
    plot_normalized_confusion_matrices(experiments)

    print("[图4] 多维度雷达图 (模型综合能力对比)...")
    plot_radar_comparison(experiments)

    print("[图5] F1-Score 全景热力图...")
    plot_f1_heatmap(experiments)

    print("\n" + "=" * 60)
    print(f"  所有高阶图表已生成完毕！")
    print(f"  保存路径: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
