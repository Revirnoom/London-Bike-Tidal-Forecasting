# -*- coding: utf-8 -*-
"""
终极篇 步骤三：构建大模型潮汐预测专属数据集 (.ts)
提取 96 小时的连续净流量波形，贴上三大门派标签，打包送入考场
"""
import pandas as pd
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def write_ts_file(data_list, labels, file_path, seq_len):
    """严格按照 sktime 变态标准生成 .ts 文件"""
    with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write("@problemName LondonBikeNetFlow\n")
        f.write(f"@timeSeriesLength {seq_len}\n")
        f.write("@missing false\n")
        f.write("@univariate true\n")
        f.write("@dimensions 1\n")
        f.write("@equalLength true\n")
        f.write("@timestamps false\n")
        # 🌟 核心：这次只有 0, 1, 2 三大门派！
        f.write("@classLabel true 0 1 2\n")
        f.write("@data\n")

        for seq, label in zip(data_list, labels):
            # 净流量包含负数，直接保留符号转字符串
            seq_str = ",".join(map(lambda x: str(round(x, 2)), seq))
            f.write(f"{seq_str}:{label}\n")


def main():
    print(">>>  准备打包潮汐预测数据集 (LondonBikeNetFlow)...")

    current_dir = Path(__file__).resolve().parent
    # 🌟 核心数据源：03_hourly_net_flow.csv (净流量表)
    data_path = current_dir.parent / "处理后的数据集" / "03_hourly_net_flow.csv"
    # 🌟 核心标签源：刚才生成的三大门派标签
    label_path = current_dir / "分类数据准备_潮汐净流量" / "station_labels_netflow.csv"

    dataset_name = "LondonBikeNetFlow"
    output_dir = current_dir / "Time_Series_Library" / "dataset" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(">>>  正在加载潮汐数据与 3 大门派标签...")
    df_raw = pd.read_csv(data_path, usecols=['station_id', 'net_flow'])
    df_labels = pd.read_csv(label_path)

    # 建立站点 ID 到三大门派标签的映射字典
    label_dict = dict(zip(df_labels['station_id'], df_labels['cluster_label']))

    seq_len = 96  # 依然让 AI 看连续 4 天 (96小时) 的波形
    samples_per_station = 50

    all_sequences = []
    all_labels = []

    print(f">>>  正在切分正负交错的潮汐波形序列 (Seq_Len = {seq_len})...")
    grouped = df_raw.groupby('station_id')

    for station_id, group in tqdm(grouped, desc="处理站点"):
        if station_id not in label_dict:
            continue

        # 🌟 核心特征：提取的是 net_flow (净流量)！
        counts = group['net_flow'].values
        if len(counts) < seq_len:
            continue

        label = label_dict[station_id]
        max_start = len(counts) - seq_len
        num_samples = min(samples_per_station, max_start)

        start_indices = random.sample(range(max_start), num_samples)

        for start_idx in start_indices:
            seq = counts[start_idx: start_idx + seq_len]
            all_sequences.append(seq)
            all_labels.append(label)

    print(f"\n>>>  共生成 {len(all_sequences)} 个有效潮汐盲测样本。")

    # 打乱并切分训练集(80%)和测试集(20%)
    combined = list(zip(all_sequences, all_labels))
    random.shuffle(combined)
    split_idx = int(len(combined) * 0.8)

    train_data = combined[:split_idx]
    test_data = combined[split_idx:]

    print(f">>>  正在生成 TSlib 框架格式文件...")
    write_ts_file([item[0] for item in train_data], [item[1] for item in train_data],
                  output_dir / f"{dataset_name}_TRAIN.ts", seq_len)
    write_ts_file([item[0] for item in test_data], [item[1] for item in test_data],
                  output_dir / f"{dataset_name}_TEST.ts", seq_len)

    print(f">>>  完美！终极潮汐考卷制作完毕，已存放至: {output_dir}")


if __name__ == "__main__":
    main()