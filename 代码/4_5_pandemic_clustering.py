# -*- coding: utf-8 -*-
"""
阶段四 (高阶分支)：基于疫情前后动态演变的站点抗压性聚类
验证“站点功能在时间上是否保持不变”，并提取新的分类标签
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def main():
    print(">>>  启动 [高阶方案] 疫情动态抗压画像提取...")

    current_dir = Path(__file__).resolve().parent
    data_path = current_dir.parent / "处理后的数据集" / "02_hourly_start_count.csv"
    output_dir = current_dir / "分类数据准备_疫情演变"
    output_dir.mkdir(exist_ok=True)

    # 1. 读取数据并转换时间格式
    print(">>>  正在读取全网站点历史数据...")
    df = pd.read_csv(data_path, usecols=['date_str', 'start_station_id', 'hour', 'start_count'])
    df['date'] = pd.to_datetime(df['date_str'])

    # 2. 划定时间结界：太平盛世 vs 疫情封控
    print(">>> ️ 正在进行时间线切割 (2019 太平盛世 vs 2020 疫情冲击)...")
    mask_pre = (df['date'] >= '2019-01-01') & (df['date'] <= '2019-12-31')
    # 英国大约 2020 年 3 月下旬实施全国封锁，我们取 4月-12月 作为严格受影响期
    mask_covid = (df['date'] >= '2020-04-01') & (df['date'] <= '2020-12-31')

    df_pre = df[mask_pre]
    df_covid = df[mask_covid]

    # 3. 分别计算两个时空的 24 小时平均画像
    print(">>>  正在提取双重时空特征...")
    pre_profile = df_pre.groupby(['start_station_id', 'hour'])['start_count'].mean().unstack().fillna(0)
    covid_profile = df_covid.groupby(['start_station_id', 'hour'])['start_count'].mean().unstack().fillna(0)

    # 为列名加上后缀，防止混淆
    pre_profile.columns = [f'pre_{h}' for h in range(24)]
    covid_profile.columns = [f'covid_{h}' for h in range(24)]

    # 4. 寻找“穿越者”：只有在 2019 和 2020 都一直存在的站点才参与对比
    combined_profiles = pd.concat([pre_profile, covid_profile], axis=1).dropna()
    print(f">>>  成功锁定 {len(combined_profiles)} 个经历了疫情完整周期的站点。")

    # 5. K-Means 高阶聚类 (基于 48 维特征：24h和平 + 24h疫情)
    n_clusters = 4
    print(f">>>  正在启动 K-Means 洞察演变规律 (K={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    combined_profiles['cluster_label'] = kmeans.fit_predict(combined_profiles.values)

    # 保存新标签
    label_save_path = output_dir / "station_labels_pandemic.csv"
    combined_profiles[['cluster_label']].to_csv(label_save_path)
    print(f">>>  疫情动态演变标签已保存至: {label_save_path}")

    # 6. 绘制高分双子对比可视化图表！
    print(">>>  正在生成绝美的时空对比折线图...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    cluster_centers = combined_profiles.groupby('cluster_label').mean()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    for cluster_id in range(n_clusters):
        count = sum(combined_profiles["cluster_label"] == cluster_id)
        # 提取 2019 和 2020 的曲线
        y_pre = cluster_centers.loc[cluster_id, [f'pre_{h}' for h in range(24)]].values
        y_covid = cluster_centers.loc[cluster_id, [f'covid_{h}' for h in range(24)]].values

        label_str = f'类别 {cluster_id} (共 {count} 站)'
        ax1.plot(range(24), y_pre, marker=markers[cluster_id], color=colors[cluster_id], linewidth=2, label=label_str)
        ax2.plot(range(24), y_covid, marker=markers[cluster_id], color=colors[cluster_id], linewidth=2, label=label_str)

    # 图表装饰
    ax1.set_title(' 疫情前 (2019): 正常的城市脉搏', fontsize=14, fontweight='bold')
    ax2.set_title(' 疫情中 (2020): 封锁下的抗压底色', fontsize=14, fontweight='bold')

    for ax in [ax1, ax2]:
        ax.set_xlabel('一天中的时间 (0点 - 23点)', fontsize=12)
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=11)

    ax1.set_ylabel('平均每小时借车量 (人次)', fontsize=12)
    fig.suptitle('伦敦单车站点功能演变对比：从"太平盛世"到"疫情冲击"', fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()

    # 保存并显示
    fig_path = output_dir / "pandemic_evolution_profiles.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()