# -*- coding: utf-8 -*-
"""
阶段四 (高阶分支)：TimesNet 疫情演变分类大考启动器
使用全新生成的 LondonBikePandemic 数据集进行高难度分类训练
"""
import os
from pathlib import Path


def main():
    print(">>>  准备唤醒 TimesNet 执行 [高阶抗压分类任务]...")
    print(">>>  挑战目标：仅看 96 小时的单车波动，精准猜出站点的疫情抗压类型！")

    current_dir = Path(__file__).resolve().parent
    model_dir = current_dir / "Time_Series_Library"

    # 强制将 Windows 路径转为 Linux 友好的正斜杠，防止 404 (继承之前的终极防坑术)
    dataset_root = str(model_dir / "dataset").replace("\\", "/")

    os.chdir(model_dir)
    print(f">>> 当前工作目录已切换至: {os.getcwd()}")

    # ==========================================
    # TimesNet 疫情高阶分类任务专属超参数配置
    # ==========================================
    args = [
        "python run.py",
        "--task_name classification",
        "--is_training 1",  # 🌟 设为 1，开启全新的训练大考！
        f'--root_path "{dataset_root}"',
        "--model_id LondonBikePandemic",  # 🌟 核心修改：指向全新的疫情演变数据集
        "--data_path LondonBikePandemic",  # 🌟 核心修改：指向全新的疫情演变数据集
        "--model TimesNet",
        "--data UEA",
        "--seq_len 96",
        "--c_out 4",
        "--e_layers 2",
        "--enc_in 1",
        "--d_model 128",  # 🛡️ OOM护盾：保持核心维度在 128
        "--d_ff 256",  # 🛡️ OOM护盾：保持全连接层在 256
        "--des 'Exp'",
        "--itr 1",
        "--train_epochs 5",  # 同样跑 5 轮看看实力
        "--batch_size 16",
        "--learning_rate 0.001",
        "--num_workers 0"  # 🛡️ Windows 防崩溃护盾
    ]

    command = " ".join(args)
    print("\n" + "=" * 50)
    print(" 即将执行的深度学习分类指令 (疫情高阶版):")
    print(command)
    print("=" * 50 + "\n")

    # 执行指令
    os.system(command)


if __name__ == "__main__":
    main()