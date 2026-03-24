# -*- coding: utf-8 -*-
"""
阶段四：站点功能化聚类分析
利用 K-Means 算法对站点的 24 小时骑行画像进行无监督聚类，自动生成分类标签
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体，防止图表中的中文显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def main():
    print(">>>  启动站点画像提取与聚类程序...")

    # 1.
    current_dir = Path(__file__).resolve().parent
    data_path = current_dir.parent / "处理后的数据集" / "02_hourly_start_count.csv"
    output_dir = current_dir / "分类数据准备"
    output_dir.mkdir(exist_ok=True)

    # 2. 读取数据 (文件较大，指定读取需要的列以节省内存)
    print(f">>>  正在读取全网站点数据 (这可能需要十几秒，请稍候): {data_path.name}")
    df = pd.read_csv(data_path, usecols=['start_station_id', 'hour', 'start_count'])

    # 3. 提取特征：计算每个站点在 0-23 点的平均借车量
    print(">>>  正在计算各个站点的 24 小时平均画像...")
    # 聚合得到每个站点每小时的平均值
    hourly_avg = df.groupby(['start_station_id', 'hour'])['start_count'].mean().reset_index()
    # 透视表：行是站点 ID，列是 0-23 小时，值是平均借车量
    station_profiles = hourly_avg.pivot(index='start_station_id', columns='hour', values='start_count').fillna(0)

    print(f">>>  成功提取了 {len(station_profiles)} 个站点的特征向量。")

    # 4. 聚类分析 (K-Means)
    # 根据任务书要求，我们分为 4 类 (例如：早高峰通勤、晚高峰通勤、综合休闲、边缘低流量)
    n_clusters = 4
    print(f">>>  正在启动 K-Means 无监督机器学习 (K={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    station_profiles['cluster_label'] = kmeans.fit_predict(station_profiles.values)

    # 5. 保存带有标签的数据 (这将作为下一个大模型的“标准答案”)
    label_save_path = output_dir / "station_labels.csv"
    station_profiles.to_csv(label_save_path)
    print(f">>>  标签数据已保存至: {label_save_path}")

    # 6. 绘制高分可视化图表
    print(">>>  正在生成各类别 24 小时特征画像曲线图...")
    plt.figure(figsize=(12, 6))

    # 计算每个类别的平均曲线
    cluster_centers = station_profiles.groupby('cluster_label').mean()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    for cluster_id in range(n_clusters):
        # 提取该类别的 24 小时数据
        y_values = cluster_centers.loc[cluster_id].values
        # 画线
        plt.plot(range(24), y_values, marker=markers[cluster_id],
                 linewidth=2, color=colors[cluster_id],
                 label=f'类别 {cluster_id} (共 {sum(station_profiles["cluster_label"] == cluster_id)} 个站)')

    plt.title('伦敦单车各类型站点的 24 小时平均使用画像 (聚类结果)', fontsize=16, fontweight='bold')
    plt.xlabel('一天中的时间 (0点 - 23点)', fontsize=12)
    plt.ylabel('平均每小时借车量 (人次)', fontsize=12)
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    # 保存图片
    fig_path = output_dir / "cluster_profiles.png"
    plt.savefig(fig_path, dpi=300)
    print(f">>> 🖼 漂亮的可视化图表已保存至: {fig_path}")

    # 在屏幕上显示出来
    plt.show()


if __name__ == "__main__":
    main()