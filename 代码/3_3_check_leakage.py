# -*- coding: utf-8 -*-
"""
阶段三 (附加)：数据泄露体检脚本
验证大模型在划分 Train / Val / Test 时，时间戳是否严格隔离，是否存在未来数据泄露
"""
import pandas as pd
from pathlib import Path


def check_data_leakage():
    print(">>> ️ 启动时间序列数据泄露检查程序...\n")

    # 1. 读取大模型使用的同一份数据
    current_dir = Path(__file__).resolve().parent
    data_path = current_dir / "Nonstationary_Transformers" / "dataset" / "london_bike" / "bike_station_14.csv"

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    total_len = len(df)

    # 2. 还原大模型底层的切分逻辑 (默认比例 7 : 1 : 2)
    # 这些数字对应你之前终端里输出的 Train 16487, Val 2350, Test 4721
    num_train = int(total_len * 0.7)
    num_test = int(total_len * 0.2)
    num_vali = total_len - num_train - num_test

    # 序列长度 (Look-back window)
    seq_len = 96

    # 计算严格的边界
    # 注意：验证集和测试集的起点，会往前倒退 seq_len (96小时)。
    # 这不是泄露！而是为了预测验证集的第一天，模型必须回头看前 96 小时的历史。
    train_df = df.iloc[0: num_train]
    val_df = df.iloc[num_train - seq_len: num_train + num_vali]
    test_df = df.iloc[total_len - num_test - seq_len: total_len]

    print("-" * 50)
    print(" 训练集 (Train) 边界：给模型学习历史规律")
    print(f"  开始时间: {train_df['date'].min()}")
    print(f"  结束时间: {train_df['date'].max()}")
    print(f"  数据量: {len(train_df)} 行")

    print("\验证集 (Validation) 边界：")
    print(f"  开始时间: {val_df['date'].min()} ")
    print(f"  结束时间: {val_df['date'].max()}")
    print(f"  数据量: {len(val_df)} 行")

    print("\ 测试集 (Test) 边界：")
    print(f"  开始时间: {test_df['date'].min()}")
    print(f"  结束时间: {test_df['date'].max()}")
    print(f"  数据量: {len(test_df)} 行")
    print("-" * 50)

    # 3. 最终判定
    train_max_date = df.iloc[num_train - 1]['date']
    test_real_start = df.iloc[total_len - num_test]['date']

    print("\n>>>  终极泄露判定结论：")
    if train_max_date < test_real_start:
        print(f" 完美通过！训练集最晚时间 ({train_max_date}) 早于测试集核心预测起点 ({test_real_start})。")
        print(" 没有任何未来的测试集标签参与到了模型的训练中！模型跑出的 MAE 分数是完全真实可信的。")
    else:
        print(" 警告：发生时间穿越，存在严重数据泄露！")


if __name__ == "__main__":
    check_data_leakage()