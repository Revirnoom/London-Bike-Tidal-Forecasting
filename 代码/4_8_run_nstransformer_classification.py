# -*- coding: utf-8 -*-
"""
阶段四：Non-stationary Transformer 常规站点功能分类任务启动器
与 TimesNet 进行同台竞技，对比分类准确率
"""
import os
from pathlib import Path


def main():
    print(">>>  准备唤醒 Non-stationary Transformer 执行 [常规功能分类任务]...")
    print(">>>  擂台对手：TimesNet (之前成绩 81.26%)")

    current_dir = Path(__file__).resolve().parent
    model_dir = current_dir / "Time_Series_Library"

    # 强制将 Windows 路径转为 Linux 友好的正斜杠
    dataset_root = str(model_dir / "dataset").replace("\\", "/")

    os.chdir(model_dir)
    print(f">>> 当前工作目录已切换至: {os.getcwd()}")

    # ==========================================
    # Non-stationary Transformer 分类任务专属超参数
    # ==========================================
    args = [
        "python run.py",
        "--task_name classification",
        "--is_training 1",
        f'--root_path "{dataset_root}"',
        "--model_id LondonBikeClass",  # 🌟 修复：改回最初的数据集名字！不要加后缀了
        "--data_path LondonBikeClass",  # 🌟 保持一致
        "--model Nonstationary_Transformer",  # 🤖 依然用 NS-Transformer 大脑
        "--data UEA",
        "--seq_len 96",
        "--c_out 4",
        "--e_layers 2",
        "--enc_in 1",
        "--d_model 128",
        "--d_ff 256",
        "--p_hidden_dims 128 128",
        "--p_hidden_layers 2",
        "--des 'Exp'",
        "--itr 1",
        "--train_epochs 5",
        "--batch_size 16",
        "--learning_rate 0.001",
        "--num_workers 0"
    ]

    command = " ".join(args)
    print("\n" + "=" * 50)
    print(" 即将执行的 NS-Transformer 分类指令:")
    print(command)
    print("=" * 50 + "\n")

    # 执行指令
    os.system(command)


if __name__ == "__main__":
    main()