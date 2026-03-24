# -*- coding: utf-8 -*-
"""
阶段三 (附加)：Non-stationary Transformer 模型启动器 (Windows友好版)
自动配置超参数，一键启动深度学习模型的训练与测试
"""
import os
import sys
from pathlib import Path


def main():
    print(">>>  准备唤醒 Non-stationary Transformer 大模型...")

    # 获取各种绝对路径
    current_dir = Path(__file__).resolve().parent
    model_dir = current_dir / "Nonstationary_Transformers"
    data_root_path = model_dir / "dataset" / "london_bike"
    data_file = "bike_station_14.csv"

    # 切换工作目录到模型所在的文件夹，防止相对路径报错
    os.chdir(model_dir)
    print(f">>> 当前工作目录已切换至: {os.getcwd()}")

    # ==========================================
    # 核心超参数配置 (Hyperparameters)
    # ==========================================
    # 这里我们使用模型自带的 custom 数据集加载器
    args = [
        "python run.py",
        "--is_training 1",  # 1表示进行训练，0表示只测试
        "--root_path " + str(data_root_path) + "\\",  # 数据集文件夹路径
        "--data_path " + data_file,  # 数据集文件名
        "--model_id LondonBike_96_24",  # 本次实验的名字 (自己起)
        "--model ns_Transformer",           # 使用的具体模型名称 (作者写的缩写)
        "--data custom",  # 数据类型：自定义表格数据
        "--features MS",  # 多变量输入，单变量输出 (Multiple to Single)
        "--target start_count",  # 我们要预测的目标列名
        "--freq h",  # 数据频率：h表示小时级
        "--seq_len 96",  # Look-back window: 模型往前看 96 个小时 (4天)
        "--label_len 48",  # Decoder token: 解码器引导长度
        "--pred_len 24",  # Forecast horizon: 模型向后预测 24 个小时 (1天)
        "--e_layers 2",  # 编码器层数 (调小一点防止普通电脑跑不动)
        "--d_layers 1",  # 解码器层数
        "--factor 3",  # 注意力机制因子
        "--enc_in 5",  # 输入特征的维度数量 (temp, precip, weekend, holiday, start_count 共5个)
        "--dec_in 5",  # 解码器输入特征维度
        "--c_out 1",  # 输出维度 (只预测 start_count 1个值)
        "--des 'Exp'",  # 实验描述
        "--itr 1",  # 实验重复次数 (跑1次即可)
        "--p_hidden_dims 16 16",  # Projector 隐藏层维度 (Non-stationary特有参数)
        "--p_hidden_layers 2",  # Projector 层数
        "--train_epochs 3",  # 训练轮数 (Epoch)，课设演示设置 3 轮即可快速看到结果
        "--batch_size 32",  # 批次大小
        "--learning_rate 0.0001", # 学习率
        "--itr 1",  # 实验重复次数 (跑1次即可)
        "--num_workers 0",  # 🌟 新增这一行：强制在 Windows 下使用单进程读取数据，彻底解决多进程报错！
        "--p_hidden_dims 16 16"  # Projector 隐藏层维度
    ]

    # 组装完整的命令行指令
    command = " ".join(args)
    print("\n" + "=" * 50)
    print(" 即将执行的深度学习指令:")
    print(command)
    print("=" * 50 + "\n")

    print(">>>  模型即将开始训练，这可能需要几分钟到十几分钟的时间，请耐心等待...")
    print(">>>  如果看到 Epoch: 1, cost time: ... 说明训练正在顺利进行！\n")

    # 执行指令
    os.system(command)


if __name__ == "__main__":
    main()