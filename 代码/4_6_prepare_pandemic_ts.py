# -*- coding: utf-8 -*-
"""
阶段四 (高阶)：构建 TimesNet 专用的疫情演变分类数据集 (.ts)
基于新生成的疫情抗压标签进行打包
"""
import pandas as pd
import numpy as np
import random
from pathlib import Path
import os
from tqdm import tqdm


def write_ts_file(data_list, labels, file_path, seq_len):
    """将数据写入标准的 .ts 格式文件 (完全满足 sktime 的变态审核标准)"""
    with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write("@problemName LondonBikePandemic\n")
        f.write(f"@timeSeriesLength {seq_len}\n")
        f.write("@missing false\n")
        f.write("@univariate true\n")
        f.write("@dimensions 1\n")
        f.write("@equalLength true\n")
        f.write("@timestamps false\n")
        f.write("@classLabel true 0 1 2 3\n")
        f.write("@data\n")

        for seq, label in zip(data_list, labels):
            seq_str = ",".join(map(lambda x: str(round(x, 2)), seq))
            f.write(f"{seq_str}:{label}\n")


def main():
    print(">>>  准备打包 TimesNet 高阶分类数据集 (LondonBikePandemic)...")

    current_dir = Path(__file__).resolve().parent
    data_path = current_dir.parent / "处理后的数据集" / "02_hourly_start_count.csv"
    # 🌟 注意这里：读取我们刚刚生成的 疫情演变 标签表
    label_path = current_dir / "分类数据准备_疫情演变" / "station_labels_pandemic.csv"

    dataset_name = "LondonBikePandemic"
    output_dir = current_dir / "Time_Series_Library" / "dataset" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(">>>  正在加载数据与疫情抗压标签...")
    df_raw = pd.read_csv(data_path, usecols=['start_station_id', 'start_count'])
    df_labels = pd.read_csv(label_path)

    label_dict = dict(zip(df_labels['start_station_id'], df_labels['cluster_label']))

    seq_len = 96
    samples_per_station = 50

    all_sequences = []
    all_labels = []

    print(f">>> ️ 正在切分时间序列 (Seq_Len = {seq_len})...")
    grouped = df_raw.groupby('start_station_id')

    for station_id, group in tqdm(grouped, desc="处理站点"):
        if station_id not in label_dict:
            continue

        counts = group['start_count'].values
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

    print(f"\n>>>  共生成 {len(all_sequences)} 个有效样本序列。")

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

    print(f">>>  完美！高阶分类数据集制作完毕，已存放至: {output_dir}")


if __name__ == "__main__":
    main()