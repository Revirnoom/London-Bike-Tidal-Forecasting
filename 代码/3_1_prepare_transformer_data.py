# -*- coding: utf-8 -*-
"""
阶段三 (附加)：为 Non-stationary Transformer 准备专属格式的数据集
把 14 号超级站点的特征，转换为标准时间序列格式
"""
import pandas as pd
from pathlib import Path
import os

# 1. 路径配置
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
DATA_IN_PATH = ROOT_DIR / "处理后的数据集" / "02_hourly_start_count.csv"

# 为模型新建一个存放数据的专属目录
TRANSFORMER_DIR = CURRENT_DIR / "Nonstationary_Transformers"
MODEL_DATA_DIR = TRANSFORMER_DIR / "dataset" / "london_bike"
MODEL_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV_PATH = MODEL_DATA_DIR / "bike_station_14.csv"


def prepare_data_for_transformer():
    print(">>> 1. 正在读取我们聚合好的千万级底表...")
    df = pd.read_csv(DATA_IN_PATH)

    # 锁定我们在 v1.0 中找到的最繁忙站点：14号站
    target_station = 14
    print(f">>> 2. 正在提取 {target_station} 号站点的所有数据...")
    df_top = df[df['start_station_id'] == target_station].copy()

    # --- 核心改造：组装 Transformer 要求的连续时间戳 ---
    print(">>> 3. 正在重组时间序列，迎合大模型格式要求...")
    # 把 date_str 和 hour 拼起来，变成 2017-01-01 08:00:00 这种标准格式
    df_top['date'] = pd.to_datetime(df_top['date_str']) + pd.to_timedelta(df_top['hour'], unit='h')

    # 按时间严格排序
    df_top = df_top.sort_values('date').reset_index(drop=True)

    # 挑选出大模型真正需要的列，并调整顺序（习惯上时间在最前，目标值在最后）
    # 特征：气温、降水、是否周末、是否假日
    # 目标：start_count
    final_cols = ['date', 'mean_temp', 'precipitation', 'is_weekend', 'is_holiday', 'start_count']
    df_final = df_top[final_cols]

    # 检查是否有缺失值并填充（Transformer 对 NaN 极其敏感）
    if df_final.isnull().sum().sum() > 0:
        print(" 发现缺失值，正在进行前向填充修复...")
        df_final = df_final.fillna(method='ffill').fillna(0)

    # 保存给大模型享用
    df_final.to_csv(OUTPUT_CSV_PATH, index=False)

    print("-" * 50)
    print(f" 专属数据集已制作完毕！")
    print(f" 存放路径：{OUTPUT_CSV_PATH}")
    print(f" 数据维度：共 {len(df_final)} 行，{len(df_final.columns)} 列。")
    print("预览前三行：")
    print(df_final.head(3))
    print("-" * 50)


if __name__ == "__main__":
    prepare_data_for_transformer()