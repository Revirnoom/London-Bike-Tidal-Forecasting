# -*- coding: utf-8 -*-
"""
终极篇 步骤一：处理 5GB 原始骑行数据，提取潮汐净流量 (Net Flow)
内存优化版 (Chunking)，防止 OOM 内存爆炸
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm


def main():
    print(">>>  启动 潮汐净流量特征提取工程...")

    current_dir = Path(__file__).resolve().parent
    raw_data_path = current_dir.parent / "数据集" / "london.csv"
    output_dir = current_dir.parent / "处理后的数据集"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "03_hourly_net_flow.csv"

    # 建立两个字典来在内存中不断累加计数，极其省内存！
    # 格式: {(date_str, hour, station_id): count}
    start_counter = {}
    end_counter = {}

    chunk_size = 1000000  # 每次只读 100 万行，绝不撑爆内存

    print(f">>>  正在分块解析 5GB 超大文件: {raw_data_path}")
    print(">>>  这可能需要几分钟，请耐心等待...")

    start_time = time.time()

    try:
        # 使用 pandas 分块读取
        chunk_iter = pd.read_csv(raw_data_path,
                                 usecols=['start_rental_date_time', 'start_station_id',
                                          'end_rental_date_time', 'end_station_id'],
                                 chunksize=chunk_size,
                                 dtype={'start_station_id': str, 'end_station_id': str})

        for chunk_idx, chunk in enumerate(chunk_iter):
            print(f"  --> 正在处理第 {chunk_idx + 1} 个 100万行数据块...")

            # 清理缺失值
            chunk = chunk.dropna()

            # 1. 统计借出量 (Start)
            # 字符串切片提取日期和小时，比 pd.to_datetime 快 10 倍！
            chunk['start_date'] = chunk['start_rental_date_time'].str[:10]
            chunk['start_hour'] = chunk['start_rental_date_time'].str[11:13]

            start_grouped = chunk.groupby(['start_date', 'start_hour', 'start_station_id']).size()
            for (d, h, sid), count in start_grouped.items():
                start_counter[(d, h, sid)] = start_counter.get((d, h, sid), 0) + count

            # 2. 统计还入量 (End)
            chunk['end_date'] = chunk['end_rental_date_time'].str[:10]
            chunk['end_hour'] = chunk['end_rental_date_time'].str[11:13]

            end_grouped = chunk.groupby(['end_date', 'end_hour', 'end_station_id']).size()
            for (d, h, sid), count in end_grouped.items():
                end_counter[(d, h, sid)] = end_counter.get((d, h, sid), 0) + count

    except FileNotFoundError:
        print(f"[!] 找不到文件，请确认 london.csv 是否在这个路径: {raw_data_path}")
        return

    print(f">>>  5GB 数据读取完毕！耗时: {time.time() - start_time:.2f} 秒。")
    print(">>>  正在合并借出与还入数据，计算潮汐净流量 (Net Flow)...")

    # 收集所有的键 (date, hour, station_id)
    all_keys = set(start_counter.keys()).union(set(end_counter.keys()))

    results = []
    for key in tqdm(all_keys, desc="计算净流量"):
        d, h, sid = key
        start_c = start_counter.get(key, 0)
        end_c = end_counter.get(key, 0)
        net_flow = end_c - start_c  # 核心公式：净流量 = 还车数 - 借车数

        results.append({
            'date_str': d,
            'hour': int(h),
            'station_id': sid,
            'start_count': start_c,
            'end_count': end_c,
            'net_flow': net_flow
        })

    print(">>>  正在将结果保存为全新的 CSV 表格...")
    df_final = pd.DataFrame(results)
    # 按时间和站点排序，让数据更整洁
    df_final = df_final.sort_values(by=['station_id', 'date_str', 'hour'])
    df_final.to_csv(output_file, index=False)

    print(f">>>  包含净流量的高阶特征表已生成: {output_file}")


if __name__ == "__main__":
    main()