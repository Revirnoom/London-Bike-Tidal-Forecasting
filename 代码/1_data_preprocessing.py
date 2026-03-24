# -*- coding: utf-8 -*-
"""
阶段一：数据清洗与特征融合
处理逻辑：分块读取 5GB 数据 -> 严格时间过滤 -> 清洗异常 -> 特征融合 -> 输出最终底表与聚合表
输出路径：../处理后的数据集/
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time

# ================= 1. 路径与全局配置 =================
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
DATA_IN_DIR = ROOT_DIR / "数据集"
# 更新了输出路径：指向你新建的 "处理后的数据集" 文件夹
DATA_OUT_DIR = ROOT_DIR / "处理后的数据集"
DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

# 课设严格要求的时间区间
START_DATE = pd.to_datetime("2017-01-01")
END_DATE = pd.to_datetime("2020-07-31 23:59:59")
CHUNK_SIZE = 1000000  # 每次处理 100 万行，防止内存溢出

# 输出文件路径及命名
CLEANED_CSV_PATH = DATA_OUT_DIR / "01_rentals_cleaned_with_features.csv"
HOURLY_CSV_PATH = DATA_OUT_DIR / "02_hourly_start_count.csv"


# ================= 2. 加载外部数据 =================
def load_external_data():
    print(">>> 开始加载外部特征数据...")
    # 1. 站点表
    df_stations = pd.read_csv(DATA_IN_DIR / "london_stations.csv")
    valid_station_ids = set(df_stations["station_id"].astype(int))
    print(f"成功加载站点表，共 {len(valid_station_ids)} 个合法站点。")

    # 2. 天气表
    df_weather = pd.read_csv(DATA_IN_DIR / "london_weather.csv")
    df_weather['date_str'] = pd.to_datetime(df_weather['date'].astype(str), format='mixed', errors='coerce').dt.date
    df_weather = df_weather[['date_str', 'mean_temp', 'precipitation']]
    print(f"成功加载天气表。")

    # 3. 节假日表
    df_holiday = pd.read_csv(DATA_IN_DIR / "UK_holiday.csv")
    hol_col = 'Date' if 'Date' in df_holiday.columns else 'date'
    df_holiday['date_str'] = pd.to_datetime(df_holiday[hol_col], format='mixed', errors='coerce').dt.date
    df_holiday['is_holiday'] = 1
    df_holiday = df_holiday[['date_str', 'is_holiday']].drop_duplicates()
    print(f"成功加载节假日表。")

    return valid_station_ids, df_weather, df_holiday


# ================= 3. 核心分块处理 =================
def process_full_data():
    v_ids, df_weather, df_holiday = load_external_data()
    london_csv_path = DATA_IN_DIR / "london.csv"

    print(f"\n>>> 开始分块清洗 5GB 数据，输出至 [{DATA_OUT_DIR.name}] 目录，请耐心等待...")
    start_time = time.time()

    chunk_hourly_stats = []
    total_processed = 0
    total_kept = 0
    is_first_chunk = True

    # 使用 chunksize 逐块读取
    for chunk in pd.read_csv(london_csv_path, chunksize=CHUNK_SIZE, low_memory=False):
        total_processed += len(chunk)

        # 1. 统一列名查找
        start_col = "start_rental_date_time" if "start_rental_date_time" in chunk.columns else "Start_rental_date_time"
        dur_col = "duration" if "duration" in chunk.columns else "Duration"
        start_id_col = "start_station_id" if "start_station_id" in chunk.columns else "Start_station_id"

        # 2. 时间解析与严格过滤 (2017-01-01 ~ 2020-07-31)
        chunk[start_col] = pd.to_datetime(chunk[start_col], errors="coerce")
        chunk = chunk.dropna(subset=[start_col])
        chunk = chunk[(chunk[start_col] >= START_DATE) & (chunk[start_col] <= END_DATE)]

        # 3. 清洗异常 Duration (60秒到24小时)
        chunk = chunk[(chunk[dur_col] >= 60) & (chunk[dur_col] <= 24 * 3600)]

        # 4. 清洗不在字典里的非法站点
        chunk = chunk[chunk[start_id_col].astype(int).isin(v_ids)]

        # 5. 提取衍生时间特征
        chunk['date_str'] = chunk[start_col].dt.date
        chunk['hour'] = chunk[start_col].dt.hour
        chunk['weekday'] = chunk[start_col].dt.weekday
        chunk['is_weekend'] = chunk['weekday'].isin([5, 6]).astype(int)

        # 6. 特征融合 (天气、节假日)
        chunk = chunk.merge(df_weather, on='date_str', how='left')
        chunk = chunk.merge(df_holiday, on='date_str', how='left')
        chunk['is_holiday'] = chunk['is_holiday'].fillna(0).astype(int)

        # 7. 统计这一个 chunk 里每个站点每小时的单量
        hourly = chunk.groupby(['date_str', 'hour', start_id_col, 'mean_temp', 'precipitation', 'is_weekend',
                                'is_holiday']).size().reset_index(name='start_count')
        chunk_hourly_stats.append(hourly)

        # 8. 追加写入 CSV (第一块写表头，后面不写)
        chunk.to_csv(CLEANED_CSV_PATH, mode='w' if is_first_chunk else 'a', index=False, header=is_first_chunk)

        total_kept += len(chunk)
        is_first_chunk = False
        print(f"  已扫描 {total_processed / 10000} 万行... 本块保留 {len(chunk)} 行")

    # ================= 4. 全局聚合处理 =================
    print("\n>>> 正在合并聚合小时级统计特征表...")
    final_hourly = pd.concat(chunk_hourly_stats, ignore_index=True)
    # 按多维度进行全局汇总
    final_hourly = \
    final_hourly.groupby(['date_str', 'hour', start_id_col, 'mean_temp', 'precipitation', 'is_weekend', 'is_holiday'])[
        'start_count'].sum().reset_index()
    # 按照时间排序
    final_hourly = final_hourly.sort_values(by=['date_str', 'hour']).reset_index(drop=True)
    final_hourly.to_csv(HOURLY_CSV_PATH, index=False)

    cost_time = (time.time() - start_time) / 60
    print("-" * 50)
    print(f" 第一阶段数据清洗与融合全部完成！耗时约 {cost_time:.1f} 分钟。")
    print(f" 原始处理总行数: {total_processed}")
    print(f" 清洗后有效行数: {total_kept}")
    print(f" 明细底表保存至: 处理后的数据集/{CLEANED_CSV_PATH.name}")
    print(f" 小时聚合表保存至: 处理后的数据集/{HOURLY_CSV_PATH.name} (共 {len(final_hourly)} 行)")
    print("-" * 50)


if __name__ == "__main__":
    process_full_data()