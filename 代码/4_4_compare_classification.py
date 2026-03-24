# -*- coding: utf-8 -*-
"""
阶段四+终极篇：分类任务多模型成绩对比与混淆矩阵生成器 (全能适配版)
支持自动扫描、识别三大任务，并为不同模型生成专属配色的热力图！
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def main():
    print(">>>  启动分类模型雷达...")
    current_dir = Path(__file__).resolve().parent
    results_dir = current_dir / "Time_Series_Library" / "results"

    # 🌟 核心升级：定义这几天我们打下的所有江山（三大任务标签）
    task_labels = {
        "LondonBikeClass": ['边缘冷门站 (0)', '傍晚休闲站 (1)', '早高峰潮汐站 (2)', '双峰通勤站 (3)'],
        "LondonBikePandemic": ['普通缩水区 (0)', '刚需避风港 (1)', '永恒长尾区 (2)', '脆弱通勤王 (3)'],
        # 👇 终极篇新增：潮汐三大门派
        "LondonBikeNetFlow": ['清晨流出型 (0)', '自平衡型 (1)', '清晨汇聚型 (2)']
    }

    if not results_dir.exists():
        print("[!] 结果目录不存在。")
        return

    exp_folders = [f for f in os.listdir(results_dir) if f.startswith("classification_")]
    if not exp_folders:
        print("[!] 未找到任何分类任务结果，请检查模型是否跑完。")
        return

    for exp in exp_folders:
        # 智能判断这是哪个大模型的成绩
        model_name = "TimesNet" if "TimesNet" in exp else "NS-Transformer" if "Nonstationary_Transformer" in exp else "Unknown"

        # 智能匹配这是哪一场考试
        current_class_names = None
        task_title = ""
        if "LondonBikeNetFlow" in exp:
            current_class_names = task_labels["LondonBikeNetFlow"]
            task_title = "潮汐净流量"
        elif "LondonBikePandemic" in exp:
            current_class_names = task_labels["LondonBikePandemic"]
            task_title = "疫情抗压演变"
        elif "LondonBikeClass" in exp:
            current_class_names = task_labels["LondonBikeClass"]
            task_title = "常规功能"
        else:
            continue

        exp_path = results_dir / exp
        pred_file = exp_path / "pred.npy"
        true_file = exp_path / "true.npy"

        if pred_file.exists() and true_file.exists():
            print(f"\n{'-' * 60}")
            print(f" 正在分析: [{task_title}] - : {model_name}")

            # 读取数据
            preds = np.load(pred_file)
            trues = np.load(true_file)
            if preds.ndim > 1 and preds.shape[1] > 1:
                preds = np.argmax(preds, axis=1)
            trues = trues.reshape(-1)
            preds = preds.reshape(-1)

            # 算分
            acc = accuracy_score(trues, preds)
            print(f"     终极准确率: {acc * 100:.2f}%")

            # 画图
            cm = confusion_matrix(trues, preds)
            plt.figure(figsize=(10, 8))

            # 🎨 视觉彩蛋：TimesNet 用冷色调蓝，NS-Transformer 用暖色调橙
            cmap_color = 'Blues' if model_name == 'TimesNet' else 'Oranges'

            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_color,
                        xticklabels=current_class_names,
                        yticklabels=current_class_names,
                        annot_kws={"size": 14})

            plt.title(f'伦敦单车【{task_title}】分类混淆矩阵\n大模型: {model_name} | 准确率: {acc * 100:.2f}%',
                      fontsize=16, pad=20)
            plt.xlabel('AI 预测的类型', fontsize=14)
            plt.ylabel('真实的类型', fontsize=14)
            plt.xticks(rotation=15)
            plt.yticks(rotation=0)
            plt.tight_layout()

            # 保存
            out_dir = current_dir / "figures"
            out_dir.mkdir(exist_ok=True)
            save_path = out_dir / f"cm_{exp}.png"
            plt.savefig(save_path, dpi=300)
            print(f"    热力图已生成！")
            plt.show()


if __name__ == "__main__":
    main()