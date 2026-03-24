# -*- coding: utf-8 -*-
"""
阶段五：分类任务高阶四宫格性能看板生成器 (全能适配版)
自动生成包含：(a)混淆矩阵 (b)P/R/F1指标 (c)真实vs预测分布 (d)类别准确率 的综合看板
强制输出路径：D:\pythonProject\数据挖掘\数据挖掘课程设计\输出结果\分类任务效果图
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_comprehensive_dashboard(trues, preds, task_title, model_name, class_names, save_path):
    """绘制高阶四宫格分类性能看板"""
    # 计算全局指标
    acc = accuracy_score(trues, preds)
    cm = confusion_matrix(trues, preds)
    sample_size = len(trues)

    # 计算各类别详细指标
    precision, recall, f1, support = precision_recall_fscore_support(trues, preds, labels=range(len(class_names)),
                                                                     zero_division=0)
    class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-9)

    pred_counts = np.bincount(preds, minlength=len(class_names))
    true_counts = np.bincount(trues, minlength=len(class_names))

    # 智能色彩主题 (TimesNet 冷色调，NS-Transformer 暖色调)
    if "TimesNet" in model_name:
        cmap_matrix = "Blues"
        color_true_dist = "#2c3e50"
        color_pred_dist = "#5dade2"
    else:
        cmap_matrix = "Oranges"
        color_true_dist = "#2c3e50"
        color_pred_dist = "#eb984e"

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.2)
    fig.suptitle(f'{task_title} — {model_name}\n总准确率 = {acc * 100:.2f}%  |  样本数 = {sample_size}', fontsize=22,
                 fontweight='bold', y=0.96)

    # ================= (a) 混淆矩阵 =================
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_matrix, ax=ax1,
                xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 14})
    ax1.set_title('(a) 混淆矩阵', fontsize=16, pad=15)
    ax1.set_xlabel('预测类别', fontsize=14)
    ax1.set_ylabel('真实类别', fontsize=14)
    ax1.tick_params(axis='x', rotation=15)

    # ================= (b) Precision / Recall / F1 =================
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(class_names))
    width = 0.25

    rects1 = ax2.bar(x - width, precision, width, label='Precision', color='#3498db')
    rects2 = ax2.bar(x, recall, width, label='Recall', color='#e67e22')
    rects3 = ax2.bar(x + width, f1, width, label='F1-Score', color='#2ecc71')

    ax2.set_title('(b) 各类别 Precision / Recall / F1', fontsize=16, pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=15, fontsize=12)
    ax2.set_ylim(0, 1.15)
    ax2.legend(loc='upper left')
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # 在 F1 柱子上标数字
    for i, rect in enumerate(rects3):
        height = rect.get_height()
        ax2.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)

    # ================= (c) 真实 vs 预测 类别数量分布 =================
    ax3 = fig.add_subplot(gs[1, 0])
    width_dist = 0.35
    ax3.bar(x - width_dist / 2, true_counts, width_dist, label='真实分布', color=color_true_dist)
    ax3.bar(x + width_dist / 2, pred_counts, width_dist, label='预测分布', color=color_pred_dist)

    ax3.set_title('(c) 真实 vs 预测 类别数量分布', fontsize=16, pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names, rotation=15, fontsize=12)
    ax3.set_ylabel('样本数', fontsize=14)
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ================= (d) 各类别准确率 =================
    ax4 = fig.add_subplot(gs[1, 1])
    y_pos = np.arange(len(class_names))

    # 使用柔和的渐变色画横向柱状图
    colors_acc = ['#66c2a5', '#8da0cb', '#ffd92f', '#a6d854'][:len(class_names)]
    ax4.barh(y_pos, class_acc, color=colors_acc, height=0.6)

    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(class_names, fontsize=13)
    ax4.set_title('(d) 各类别准确率', fontsize=16, pad=15)
    ax4.set_xlabel('准确率', fontsize=14)
    ax4.set_xlim(0, 1.1)
    ax4.grid(axis='x', linestyle='--', alpha=0.5)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # 标上准确率数字
    for i, acc_val in enumerate(class_acc):
        ax4.text(acc_val + 0.02, i, f'{acc_val * 100:.1f}%', va='center', fontsize=12)

    # 画一条代表全局准确率的红色虚线
    ax4.axvline(acc, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=f'总准确率 {acc * 100:.2f}%')
    ax4.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    -> 绝美看板已生成: {save_path.name}")


def main():
    print("=" * 60)
    print("  深度学习分类任务：高阶四宫格性能看板生成器启动")
    print("=" * 60)

    current_dir = Path(__file__).resolve().parent
    results_dir = current_dir / "Time_Series_Library" / "results"

    # 🌟 核心修改：强制指向你截图中的绝对路径
    out_dir = Path(r"D:\pythonProject\数据挖掘\数据挖掘课程设计\输出结果\分类任务效果图")
    out_dir.mkdir(parents=True, exist_ok=True)

    task_labels = {
        "LondonBikeClass": (
        ['边缘冷门站', '傍晚休闲站', '早高峰潮汐站', '双峰通勤站'], "常规站点功能分类", "01_cls_Class"),
        "LondonBikePandemic": (
        ['普通缩水区', '刚需避风港', '永恒长尾区', '脆弱通勤王'], "疫情抗压演变分类", "02_cls_Pandemic"),
        "LondonBikeNetFlow": (['清晨流出型', '自平衡型', '清晨汇聚型'], "潮汐净流量分类", "03_cls_NetFlow")
    }

    if not results_dir.exists():
        print("[!] 未找到 results 文件夹，请确认路径。")
        return

    exp_folders = [f for f in os.listdir(results_dir) if f.startswith("classification_")]
    if not exp_folders:
        print("[!] results 文件夹下没有分类模型的结果。")
        return

    for exp in exp_folders:
        exp_path = results_dir / exp
        pred_file = exp_path / "pred.npy"
        true_file = exp_path / "true.npy"

        if not (pred_file.exists() and true_file.exists()):
            continue

        model_name = "TimesNet" if "TimesNet" in exp else "NS-Transformer" if "Nonstationary_Transformer" in exp else "Unknown"

        # 匹配任务类型
        matched_task = None
        for key in task_labels.keys():
            if key in exp:
                matched_task = key
                break

        if not matched_task:
            continue

        class_names, task_title, file_prefix = task_labels[matched_task]

        print(f"\n 正在渲染看板: [{task_title}] - {model_name}")

        preds = np.load(pred_file)
        trues = np.load(true_file)
        if preds.ndim > 1 and preds.shape[1] > 1:
            preds = np.argmax(preds, axis=1)
        trues = trues.reshape(-1)
        preds = preds.reshape(-1)

        save_name = f"{file_prefix}_{model_name}_Dashboard.png"
        plot_comprehensive_dashboard(trues, preds, task_title, model_name, class_names, out_dir / save_name)

    print("\n" + "=" * 60)
    print(f"  所有高阶分类看板均已生成完毕！")
    print(f"  请前往查收: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()