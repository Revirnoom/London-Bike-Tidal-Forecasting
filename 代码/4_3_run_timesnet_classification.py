# -*- coding: utf-8 -*-
"""
阶段四：TimesNet 站点功能分类任务启动器 (一键读档测试版)
跳过漫长的训练，直接读取之前最好的 AI 大脑，并输出 pred.npy 结果！
"""
import os
from pathlib import Path


def main():
    print(">>>  准备唤醒 TimesNet 执行 [分类测试与结果提取]...")

    current_dir = Path(__file__).resolve().parent
    model_dir = current_dir / "Time_Series_Library"

    # 强制将 Windows 路径转为 Linux 友好的正斜杠，防止 404
    dataset_root = str(model_dir / "dataset").replace("\\", "/")

    os.chdir(model_dir)
    print(f">>> 当前工作目录已切换至: {os.getcwd()}")

    # ==========================================
    # TimesNet 分类任务专属超参数配置
    # ==========================================
    args = [
        "python run.py",
        "--task_name classification",
        "--is_training 0",  # 🌟 修复1：明确设置为 0！直接进行期末考，不要再训练了！
        f'--root_path "{dataset_root}"',
        "--model_id LondonBikeClass",
        "--data_path LondonBikeClass",
        "--model TimesNet",
        "--data UEA",
        "--seq_len 96",
        "--c_out 4",
        "--e_layers 2",
        "--enc_in 1",
        "--d_model 128",
        "--d_ff 256",
        "--des 'Exp'",
        "--itr 1",
        "--train_epochs 5",
        "--batch_size 16",
        "--learning_rate 0.001",
        "--num_workers 0"  # 🌟 修复2：强制锁死为 0，彻底根除刚才的 Windows 多进程报错！
    ]

    command = " ".join(args)
    print("\n" + "=" * 50)
    print(" 即将执行的深度学习分类读档指令:")
    print(command)
    print("=" * 50 + "\n")

    # 执行指令
    os.system(command)


if __name__ == "__main__":
    main()