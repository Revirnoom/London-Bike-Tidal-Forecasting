# -*- coding: utf-8 -*-
"""
终极篇 步骤四：TimesNet 潮汐净流量盲测大考
"""
import os
from pathlib import Path


def main():
    print(">>>  准备唤醒 TimesNet 执行 [潮汐分类任务]...")

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
        "--model_id LondonBikeNetFlow",  # 🌟 指向我们刚做好的潮汐考卷
        "--data_path LondonBikeNetFlow",
        "--model TimesNet",
        "--data UEA",
        "--seq_len 96",
        "--c_out 3",  # 🌟 改为 3，因为这次只有 3 大门派
        "--e_layers 2",
        "--enc_in 1",
        "--d_model 128",
        "--d_ff 256",
        "--des 'Exp'",
        "--itr 1",
        "--train_epochs 5",
        "--batch_size 16",
        "--learning_rate 0.001",
        "--num_workers 0"
    ]

    command = " ".join(args)
    print("\n" + "=" * 50)
    print(" 即将执行 TimesNet 潮汐分类指令:")
    print(command)
    print("=" * 50 + "\n")
    os.system(command)


if __name__ == "__main__":
    main()