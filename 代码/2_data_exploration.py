# -*- coding: utf-8 -*-
"""
阶段二：数据探索与可视化 (EDA - 全维度终极版)
输出路径已更新为: 输出结果/数据分析可视化图像/
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 路径配置 =================
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
RAW_DATA_DIR = ROOT_DIR / "数据集"
PROCESSED_DATA_DIR = ROOT_DIR / "处理后的数据集"

# 🌟 修改点：更新为你要求的数据分析可视化输出路径
FIGURES_OUT_DIR = ROOT_DIR / "输出结果" / "数据分析可视化图像"
FIGURES_OUT_DIR.mkdir(parents=True, exist_ok=True)

HOURLY_CSV_PATH = PROCESSED_DATA_DIR / "02_hourly_start_count.csv"
CLEANED_CSV_PATH = PROCESSED_DATA_DIR / "01_rentals_cleaned_with_features.csv"
STATIONS_CSV_PATH = RAW_DATA_DIR / "london_stations.csv"

def load_data():
    print(">>> 正在加载轻量级聚合表与站点表...")
    df_hourly = pd.read_csv(HOURLY_CSV_PATH)
    df_hourly['date_str'] = pd.to_datetime(df_hourly['date_str'])
    df_hourly['weekday'] = df_hourly['date_str'].dt.weekday
    df_hourly['is_weekend'] = df_hourly['weekday'].apply(lambda x: 1 if x >= 5 else 0)

    df_stations = pd.read_csv(STATIONS_CSV_PATH)
    return df_hourly, df_stations

def plot_station_frequency(df):
    print(">>> 绘制: 站点使用频率分布图...")
    station_counts = df.groupby('start_station_id')['start_count'].sum().sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    top_20 = station_counts.head(20)
    sns.barplot(x=top_20.values, y=[str(int(i)) for i in top_20.index], ax=axes[0], palette="viridis")
    axes[0].set_title('Top 20 最繁忙出发站点')
    sns.histplot(station_counts.values, bins=50, kde=True, ax=axes[1], color="coral")
    axes[1].set_title('全网站点使用频次分布')
    plt.tight_layout()
    plt.savefig(FIGURES_OUT_DIR / "01_station_frequency.png", dpi=300)
    plt.close()

def plot_hourly_trend_by_daytype(df):
    print(">>> 绘制: 24小时潮汐图...")
    hourly_trend = df.groupby(['hour', 'is_weekend'])['start_count'].sum().reset_index()
    days_count = df.groupby('is_weekend')['date_str'].nunique()
    hourly_trend.loc[hourly_trend['is_weekend'] == 0, 'avg_count'] = hourly_trend['start_count'] / days_count[0]
    hourly_trend.loc[hourly_trend['is_weekend'] == 1, 'avg_count'] = hourly_trend['start_count'] / days_count[1]
    hourly_trend['Day Type'] = hourly_trend['is_weekend'].map({0: '工作日', 1: '周末'})

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=hourly_trend, x='hour', y='avg_count', hue='Day Type', marker='o', palette='Set1')
    plt.title('24小时平均使用潮汐规律对比')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_OUT_DIR / "02_hourly_tide_pattern.png", dpi=300)
    plt.close()

def plot_spatial_distribution(df_hourly, df_stations):
    print(">>> 绘制: 空间热点差异图...")
    df = df_hourly.copy()
    df['time_period'] = '其他'
    df.loc[(df['hour'] >= 7) & (df['hour'] <= 9), 'time_period'] = '早高峰 (7-9点)'
    df.loc[(df['hour'] >= 17) & (df['hour'] <= 19), 'time_period'] = '晚高峰 (17-19点)'
    df.loc[(df['hour'] >= 22) | (df['hour'] <= 6), 'time_period'] = '夜间 (22-6点)'

    peak_df = df[df['time_period'] != '其他']
    station_period_counts = peak_df.groupby(['start_station_id', 'time_period'])['start_count'].sum().reset_index()
    merged_sp = station_period_counts.merge(df_stations[['station_id', 'longitude', 'latitude']],
                                            left_on='start_station_id', right_on='station_id', how='inner')
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    periods = ['早高峰 (7-9点)', '晚高峰 (17-19点)', '夜间 (22-6点)']
    for ax, period in zip(axes, periods):
        subset = merged_sp[merged_sp['time_period'] == period]
        if len(subset) == 0: continue
        sc = ax.scatter(subset['longitude'], subset['latitude'],
                        s=subset['start_count'] / subset['start_count'].max() * 200 + 10,
                        c=subset['start_count'], cmap="YlOrRd", alpha=0.7)
        ax.set_title(f'{period} 出发热点分布')
        plt.colorbar(sc, ax=ax, label='总租借次数')
    plt.tight_layout()
    plt.savefig(FIGURES_OUT_DIR / "03_spatial_peak_distribution.png", dpi=300)
    plt.close()

def plot_weather_impact(df):
    print(">>> 绘制: 气象特征影响分析图...")
    daily_df = df.groupby(['date_str', 'mean_temp', 'precipitation'])['start_count'].sum().reset_index()
    daily_df = daily_df.dropna(subset=['mean_temp'])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.regplot(data=daily_df, x='mean_temp', y='start_count', ax=axes[0],
                scatter_kws={'alpha': 0.5, 'color': '#2ecc71'}, line_kws={'color': '#e74c3c'})
    axes[0].set_title('日均气温与单车使用量关系')
    daily_df['Rain Level'] = daily_df['precipitation'].apply(lambda x: '无雨' if x==0 else ('小雨' if x<=5 else '中大雨'))
    sns.boxplot(data=daily_df, x='Rain Level', y='start_count', ax=axes[1], order=['无雨', '小雨', '中大雨'])
    axes[1].set_title('降水级别对单车使用量的影响')
    plt.tight_layout()
    plt.savefig(FIGURES_OUT_DIR / "04_weather_impact.png", dpi=300)
    plt.close()

def plot_daily_trend(df):
    print(">>> 绘制: 长期宏观趋势与季节性图...")
    daily_df = df.groupby('date_str')['start_count'].sum().reset_index()
    plt.figure(figsize=(15, 5))
    sns.lineplot(data=daily_df, x='date_str', y='start_count', color='#3498db', linewidth=1)
    daily_df['rolling_30'] = daily_df['start_count'].rolling(window=30).mean()
    sns.lineplot(data=daily_df, x='date_str', y='rolling_30', color='#e74c3c', linewidth=2, label='30天移动平均')
    plt.title('每日总租借量趋势')
    plt.tight_layout()
    plt.savefig(FIGURES_OUT_DIR / "05_daily_trend_seasonality.png", dpi=300)
    plt.close()

def main():
    if not HOURLY_CSV_PATH.exists():
        print("找不到输入数据文件。")
        return
    df_hourly, df_stations = load_data()
    plot_station_frequency(df_hourly)
    plot_hourly_trend_by_daytype(df_hourly)
    plot_spatial_distribution(df_hourly, df_stations)
    plot_weather_impact(df_hourly)
    plot_daily_trend(df_hourly)
    print(">>> 阶段二：数据探索可视化完成！图表已存入数据分析可视化图像文件夹。")

if __name__ == "__main__":
    main()