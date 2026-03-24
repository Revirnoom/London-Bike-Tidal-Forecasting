# -*- coding: utf-8 -*-
"""
阶段四：站点的功能化描述 (无监督聚类)
提取 802 个站点的业务特征，使用三种聚类算法进行功能划分，并生成雷达图与地图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from math import pi

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 路径与配置 =================
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
DATA_IN_PATH = ROOT_DIR / "处理后的数据集" / "02_hourly_start_count.csv"
STATIONS_CSV_PATH = ROOT_DIR / "数据集" / "london_stations.csv"
FIGURES_OUT_DIR = ROOT_DIR / "输出结果" / "figures"
FIGURES_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ================= 2. 特征工程 =================
def extract_station_features():
    print(">>> 1. 正在从千万级底表中提取站点画像特征...")
    df = pd.read_csv(DATA_IN_PATH)

    # 定义时段
    df['is_morning_peak'] = df['hour'].apply(lambda x: 1 if 7 <= x <= 9 else 0)
    df['is_evening_peak'] = df['hour'].apply(lambda x: 1 if 17 <= x <= 19 else 0)

    # 计算每个站点的总指标
    station_stats = df.groupby('start_station_id').agg(
        total_starts=('start_count', 'sum'),
        morning_starts=('start_count', lambda x: df.loc[x.index, 'is_morning_peak'].dot(x)),
        evening_starts=('start_count', lambda x: df.loc[x.index, 'is_evening_peak'].dot(x)),
        weekend_starts=('start_count', lambda x: df.loc[x.index, 'is_weekend'].dot(x))
    ).reset_index()

    # 构造相对比例特征 (消除站点绝对规模差异带来的影响)
    station_stats['morning_ratio'] = station_stats['morning_starts'] / station_stats['total_starts']
    station_stats['evening_ratio'] = station_stats['evening_starts'] / station_stats['total_starts']
    station_stats['weekend_ratio'] = station_stats['weekend_starts'] / station_stats['total_starts']

    # 绝对规模取对数，防止极值拉伸
    station_stats['log_total'] = np.log1p(station_stats['total_starts'])

    # 过滤掉总单量极少的死站 (少于100单)
    station_stats = station_stats[station_stats['total_starts'] > 100].reset_index(drop=True)

    features = ['morning_ratio', 'evening_ratio', 'weekend_ratio', 'log_total']
    return station_stats, features


# ================= 3. 聚类实验 =================
def run_clustering_experiments(station_stats, feature_cols):
    print(">>> 2. 正在进行三种机器学习算法的聚类实验...")
    X = station_stats[feature_cols]

    # 标准化 (让所有特征在同一个量纲下)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_clusters = 4  # 预设 4 种业务角色

    # 算法 1: K-Means (主模型)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_kmeans = kmeans.fit_predict(X_scaled)
    score_kmeans = silhouette_score(X_scaled, labels_kmeans)

    # 算法 2: Gaussian Mixture Model (GMM)
    gmm = GaussianMixture(n_components=n_clusters, random_state=42)
    labels_gmm = gmm.fit_predict(X_scaled)
    score_gmm = silhouette_score(X_scaled, labels_gmm)

    # 算法 3: Agglomerative Clustering (层次聚类)
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    labels_agg = agg.fit_predict(X_scaled)
    score_agg = silhouette_score(X_scaled, labels_agg)

    print("-" * 40)
    print(" 聚类算法轮廓系数 (Silhouette Score) 对比：")
    print(f"  [K-Means] : {score_kmeans:.4f} (选为主力模型)")
    print(f"  [GMM]     : {score_gmm:.4f}")
    print(f"  [层次聚类] : {score_agg:.4f}")
    print("-" * 40)

    # 将主力模型的结果贴回原表
    station_stats['Cluster'] = labels_kmeans

    # 业务重命名 (根据聚类中心自动判断)
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=feature_cols)
    cluster_names = {}
    for i in range(n_clusters):
        c_data = centroids.iloc[i]
        if c_data['log_total'] < centroids['log_total'].quantile(0.25):
            cluster_names[i] = "边缘冷门站 (Edge)"
        elif c_data['weekend_ratio'] > centroids['weekend_ratio'].quantile(0.6):
            cluster_names[i] = "周末休闲站 (Leisure)"
        elif c_data['morning_ratio'] > c_data['evening_ratio']:
            cluster_names[i] = "早峰输出站 (Morning Commute)"
        else:
            cluster_names[i] = "晚峰输出站 (Evening Commute)"

    station_stats['Cluster_Name'] = station_stats['Cluster'].map(cluster_names)
    print("各类站点数量分布：")
    print(station_stats['Cluster_Name'].value_counts())

    return station_stats, centroids, cluster_names


# ================= 4. 酷炫可视化 =================
def plot_radar_chart(centroids, cluster_names):
    print(">>> 3. 正在绘制站点功能雷达图...")
    features = ['早高峰占比', '晚高峰占比', '周末流量占比', '整体规模(Log)']
    N = len(features)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], features, size=12)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
    plt.ylim(0, 1)

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    # 对 centroid 数据进行 0-1 归一化以方便雷达图展示
    norm_centroids = (centroids - centroids.min()) / (centroids.max() - centroids.min() + 1e-5)

    for i in range(len(centroids)):
        values = norm_centroids.iloc[i].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=cluster_names[i], color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('伦敦共享单车：四大功能站点的特征画像 (雷达图)', size=16, y=1.1)
    plt.tight_layout()
    plt.savefig(FIGURES_OUT_DIR / "10_cluster_radar.png", dpi=300)
    plt.close()


def plot_spatial_map(station_stats):
    print(">>> 4. 正在绘制地理空间分布图...")
    stations_geo = pd.read_csv(STATIONS_CSV_PATH)
    merged = station_stats.merge(stations_geo[['station_id', 'latitude', 'longitude']],
                                 left_on='start_station_id', right_on='station_id', how='inner')

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=merged, x='longitude', y='latitude', hue='Cluster_Name',
                    palette=['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'],
                    size='total_starts', sizes=(20, 400), alpha=0.8, edgecolor='white')

    plt.title('伦敦站点的城市功能分布地图', fontsize=18)
    plt.xlabel('经度')
    plt.ylabel('纬度')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="站点功能分类")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_OUT_DIR / "11_cluster_spatial_map.png", dpi=300)
    plt.close()


def main():
    stats, feature_cols = extract_station_features()
    stats_clustered, centroids, c_names = run_clustering_experiments(stats, feature_cols)
    plot_radar_chart(centroids, c_names)
    plot_spatial_map(stats_clustered)

    print("雷达图和地理分布图在figures中")


if __name__ == "__main__":
    main()