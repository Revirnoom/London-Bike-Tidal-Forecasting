# -*- coding: utf-8 -*-
"""
阶段三 (附加)：多模型终极成绩单对比查看器
修复版：自动过滤分类任务，只看回归预测的 MAE 和 MSE 成绩
"""
import os
import numpy as np
from pathlib import Path


def check_results():
    print(">>>  正在启动大模型成绩对比雷达...\n")

    current_dir = Path(__file__).resolve().parent

    model_dirs = {
        "Non-stationary Transformer (NeurIPS 2022)": current_dir / "Nonstationary_Transformers" / "results",
        "TimesNet (ICLR 2023)": current_dir / "Time_Series_Library" / "results"
    }

    for model_name, results_path in model_dirs.items():
        print("-" * 70)
        print(f" 模型阵营: {model_name}")

        if not results_path.exists():
            print("  [!] 暂未找到该模型的 results 文件夹，可能还未运行。")
            continue

        exp_folders = [f for f in os.listdir(results_path) if (results_path / f).is_dir()]

        if not exp_folders:
            print("  [!] 文件夹为空，暂无实验记录。")
            continue

        valid_count = 0
        for exp in exp_folders:
            # 🌟 核心修复：直接跳过所有分类任务的文件夹，只看回归任务！
            if "classification" in exp:
                continue

            metrics_file = results_path / exp / "metrics.npy"
            if metrics_file.exists():
                valid_count += 1
                metrics = np.load(metrics_file)
                mae = metrics[0]
                mse = metrics[1]

                short_exp_name = exp[:45] + "..." if len(exp) > 45 else exp
                print(f"   实验名称: {short_exp_name}")
                print(f"      MAE (平均绝对误差): {mae:.4f}")
                print(f"      MSE (均方误差):     {mse:.4f}\n")
            else:
                print(f"   实验名称: {exp} \n     [!] 未找到 metrics.npy 成绩单")

        if valid_count == 0:
            print("  [!] 未找到回归任务的成绩单 (可能只跑了分类任务)。")

    print("-" * 70)
    print(">>>  提示回顾：")
    print(">>> MAE 和 MSE 数值越小，说明模型的预测越精准！")


if __name__ == "__main__":
    check_results()