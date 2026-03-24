# -*- coding: utf-8 -*-
"""
阶段三 (附加)：TimesNet 模型启动器
调用最新的 Time-Series-Library 框架，运行 ICLR 2023 顶会模型
"""
import os
import sys
from pathlib import Path


def main():
    print(">>>  准备唤醒 ICLR 2023 顶会模型: TimesNet...")

    current_dir = Path(__file__).resolve().parent
    # 指向我们新下载的 TSlib 库
    model_dir = current_dir / "Time_Series_Library"
    data_root_path = model_dir / "dataset" / "london_bike"
    data_file = "bike_station_14.csv"

    os.chdir(model_dir)
    print(f">>> 当前工作目录已切换至: {os.getcwd()}")

    # ==========================================
    # TimesNet 核心超参数配置
    # ==========================================
    args = [
        "python run.py",
        "--task_name long_term_forecast",  # 最新框架要求指定任务类型：长序列预测
        "--is_training 1",
        "--root_path " + str(data_root_path) + "\\",
        "--data_path " + data_file,
        "--model_id LondonBike_TimesNet",
        "--model TimesNet",  # 🌟 指定使用 TimesNet 模型
        "--data custom",
        "--features MS",
        "--target start_count",
        "--freq h",
        "--seq_len 96",  # 依然是回溯 96 小时
        "--label_len 48",
        "--pred_len 24",  # 预测未来 24 小时
        "--e_layers 2",  # TimesBlock 层数
        "--enc_in 5",  # 输入 5 个特征
        "--dec_in 5",
        "--c_out 1",  # 输出 1 个预测值
        "--des 'Exp'",
        "--itr 1",
        "--train_epochs 3",
        "--batch_size 16",  #
        "--learning_rate 0.0001",
        "--d_model 128",  # 🌟 新增：将模型核心维度从 512 压缩到 128
        "--d_ff 256",  # 🌟 新增：将全连接层维度从 2048 压缩到 256
        "--num_workers 0"
    ]

    command = " ".join(args)
    print("\n" + "=" * 50)
    print(" 即将执行的深度学习指令:")
    print(command)
    print("=" * 50 + "\n")

    # 执行指令
    os.system(command)


if __name__ == "__main__":
    main()