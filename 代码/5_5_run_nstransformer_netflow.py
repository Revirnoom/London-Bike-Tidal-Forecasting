# -*- coding: utf-8 -*-
"""
终极篇 步骤五：Non-stationary Transformer 潮汐净流量盲测大考
"""
import os
from pathlib import Path


def main():
    print(">>>  准备唤醒 NS-Transformer 执行 [终极潮汐分类任务]...")

    current_dir = Path(__file__).resolve().parent
    model_dir = current_dir / "Time_Series_Library"
    dataset_root = str(model_dir / "dataset").replace("\\", "/")

    os.chdir(model_dir)
    print(f">>> 当前工作目录已切换至: {os.getcwd()}")

    args = [
        "python run.py",
        "--task_name classification",
        "--is_training 1",
        f'--root_path "{dataset_root}"',
        "--model_id LondonBikeNetFlow",  # 🌟 同样的潮汐考卷
        "--data_path LondonBikeNetFlow",
        "--model Nonstationary_Transformer",  # 🌟 换成去平稳化 Transformer 大脑
        "--data UEA",
        "--seq_len 96",
        "--c_out 3",  # 🌟 3 大门派
        "--e_layers 2",
        "--enc_in 1",
        "--d_model 128",
        "--d_ff 256",
        "--p_hidden_dims 128 128",  # NS-Transformer 专属投影参数
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
    print(" 即将执行 NS-Transformer 潮汐分类指令:")
    print(command)
    print("=" * 50 + "\n")
    os.system(command)


if __name__ == "__main__":
    main()