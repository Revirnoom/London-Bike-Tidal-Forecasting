# -*- coding: utf-8 -*-
"""
阶段四：构建 TimesNet 分类任务专用的 UEA 格式 (.ts) 数据集 (防 sktime 报错终极加强版)
将原始的 1 维时间序列切分为指定长度的片段，并贴上聚类得到的标签。
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
        f.write("@problemName LondonBikeClass\n")
        f.write(f"@timeSeriesLength {seq_len}\n")
        f.write("@missing false\n")
        f.write("@univariate true\n")
        f.write("@timestamps false\n")  # 🌟 修复：就是缺了这一行！必须明确告诉它没有时间戳
        f.write("@classLabel true 0 1 2 3\n")
        f.write("@data\n")

        for seq, label in zip(data_list, labels):
            # 将数值转为逗号分隔的字符串
            seq_str = ",".join(map(lambda x: str(round(x, 2)), seq))
            # 格式: 特征序列:标签
            f.write(f"{seq_str}:{label}\n")

def main():
    print(">>>  准备打包 TimesNet 专用的分类数据集 (修复 sktime 强迫症版)...")

    current_dir = Path(__file__).resolve().parent
    data_path = current_dir.parent / "处理后的数据集" / "02_hourly_start_count.csv"
    label_path = current_dir / "分类数据准备" / "station_labels.csv"

    # TSlib 框架要求分类数据集存放在特定结构的目录中
    dataset_name = "LondonBikeClass"
    output_dir = current_dir / "Time_Series_Library" / "dataset" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(">>>  正在加载数据与标签...")
    df_raw = pd.read_csv(data_path, usecols=['start_station_id', 'start_count'])
    df_labels = pd.read_csv(label_path)

    # 建立 站点ID -> 标签 的字典
    label_dict = dict(zip(df_labels['start_station_id'], df_labels['cluster_label']))

    seq_len = 96  # 截取连续 96 小时 (4天) 作为一段供模型学习的样本
    samples_per_station = 50  # 为了防止数据量太大撑爆内存，每个站点随机抽取 50 个时间段片段

    all_sequences = []
    all_labels = []

    print(f">>> ✂ 正在切分时间序列并打标签 (Seq_Len = {seq_len})...")
    grouped = df_raw.groupby('start_station_id')

    for station_id, group in tqdm(grouped, desc="处理站点"):
        if station_id not in label_dict:
            continue

        counts = group['start_count'].values
        if len(counts) < seq_len:
            continue

        label = label_dict[station_id]

        # 提取滑动窗口数据
        max_start = len(counts) - seq_len
        # 如果数据足够多，进行随机采样；如果不够 50 段，则全取
        num_samples = min(samples_per_station, max_start)
        start_indices = random.sample(range(max_start), num_samples)

        for start_idx in start_indices:
            seq = counts[start_idx: start_idx + seq_len]
            all_sequences.append(seq)
            all_labels.append(label)

    print(f"\n>>>  共生成 {len(all_sequences)} 个有效样本序列。")

    # 划分训练集和测试集 (80% 训练, 20% 测试)
    combined = list(zip(all_sequences, all_labels))
    random.shuffle(combined)
    split_idx = int(len(combined) * 0.8)

    train_data = combined[:split_idx]
    test_data = combined[split_idx:]

    # 写入 .ts 文件
    print(f">>>  正在生成 TSlib 框架格式文件...")
    write_ts_file([item[0] for item in train_data], [item[1] for item in train_data],
                  output_dir / f"{dataset_name}_TRAIN.ts", seq_len)
    write_ts_file([item[0] for item in test_data], [item[1] for item in test_data],
                  output_dir / f"{dataset_name}_TEST.ts", seq_len)

    print(f">>>  完美！分类数据集制作完毕，已存放至: {output_dir}")
    print(">>> (包含 _TRAIN.ts 和 _TEST.ts 两个文件)")


if __name__ == "__main__":
    main()