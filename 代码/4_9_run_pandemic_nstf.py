# -*- coding: utf-8 -*-
"""
阶段四 (高阶分支)：Non-stationary Transformer 疫情演变分类任务启动器
使用 LondonBikePandemic 数据集，与 TimesNet 进行同台竞技
核心特性：de-stationary projector 参数用于处理非平稳时间序列
"""
import os
from pathlib import Path


def main():
    print(">>> 准备唤醒 Non-stationary Transformer 执行 [疫情抗压分类任务]...")
    print(">>> 挑战目标：仅看 96 小时的单车波动，精准猜出站点的疫情抗压类型！")
    print(">>> 擂台对手：TimesNet 疫情分类 (4_7)")

    current_dir = Path(__file__).resolve().parent
    model_dir = current_dir / "Time_Series_Library"
    dataset_root = str(model_dir / "dataset").replace("\\", "/")

    os.chdir(model_dir)
    print(f">>> 当前工作目录已切换至: {os.getcwd()}")

    # ==========================================
    # Non-stationary Transformer 疫情分类专属超参数
    # ==========================================
    args = [
        "python run.py",
        "--task_name classification",
        "--is_training 1",
        f'--root_path "{dataset_root}"',
        "--model_id LondonBikePandemic",
        "--data_path LondonBikePandemic",
        "--model Nonstationary_Transformer",
        "--data UEA",
        "--seq_len 96",
        "--c_out 4",
        "--e_layers 2",
        "--enc_in 1",
        "--d_model 128",
        "--d_ff 256",
        "--p_hidden_dims 128 128",
        "--p_hidden_layers 2",
        "--des Exp",
        "--itr 1",
        "--train_epochs 5",
        "--batch_size 16",
        "--learning_rate 0.001",
        "--num_workers 0",
    ]

    command = " ".join(args)
    print("\n" + "=" * 50)
    print(" 即将执行的 NS-Transformer 疫情分类指令:")
    print(command)
    print("=" * 50 + "\n")

    os.system(command)


if __name__ == "__main__":
    main()