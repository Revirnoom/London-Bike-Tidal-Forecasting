# -*- coding: utf-8 -*-
"""
终极篇 步骤二：高级层次聚类与潮汐画像提取
基于站点的 24 小时平均净流量进行聚类，并生成树状图与折线图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def main():
    print(">>>  启动高级层次聚类 (Hierarchical Clustering)...")

    current_dir = Path(__file__).resolve().parent
    # 读取我们刚才跑出来的潮汐净流量表
    data_path = current_dir.parent / "处理后的数据集" / "03_hourly_net_flow.csv"
    output_dir = current_dir / "分类数据准备_潮汐净流量"
    output_dir.mkdir(exist_ok=True)

    # 1. 读取净流量数据
    print(">>> 正在加载潮汐净流量数据...")
    df = pd.read_csv(data_path)

    # 2. 计算每个站点在 24 小时内的平均净流量画像
    print(">>>  正在提取每个站点的 24 维净流量特征 (24小时均值)...")
    # 宽表转换：行是 station_id，列是 hour (0-23)，值是 net_flow 的平均值
    profile = df.groupby(['station_id', 'hour'])['net_flow'].mean().unstack().fillna(0)

    # 3. 层次聚类 (使用 Ward 最小方差法)
    print(">>>  正在构建层次聚类族谱 (计算距离矩阵)...")
    Z = linkage(profile.values, method='ward')

    # 4. 绘制并保存极其震撼的【树状图 Dendrogram】
    plt.figure(figsize=(12, 6))
    plt.title('伦敦单车站点潮汐净流量 - 层次聚类树状图 (Dendrogram)', fontsize=16)
    plt.xlabel('站点聚类分支', fontsize=12)
    plt.ylabel('Ward 距离 (差异度)', fontsize=12)
    # truncate_mode='lastp' 可以让几百个站点折叠成最后几十个核心分支，避免糊成一团
    dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=12., show_contracted=True)
    plt.tight_layout()
    plt.savefig(output_dir / "01_dendrogram.png", dpi=300)
    plt.show()  # 弹出的第一张图！

    # 5. 切割树，强行提取出 4 个终极大类
    n_clusters = 3
    print(f"\n>>>  正在将树状图从最顶层切分为 {n_clusters} 大类...")
    labels = fcluster(Z, t=n_clusters, criterion='maxclust')

    # 这里的 labels 默认是从 1 开始的，我们把它转成从 0 开始，保持和大模型规范一致
    profile['cluster_label'] = labels - 1

    # 保存大模型的“标准答案”标签
    label_save_path = output_dir / "station_labels_netflow.csv"
    profile[['cluster_label']].to_csv(label_save_path)
    print(f">>>  潮汐分类标签 (标准答案) 已成功保存至: {label_save_path}")

    # 6. 绘制 4 大类别的 24 小时潮汐净流量曲线
    print(">>>  正在生成终极潮汐画像图表...")
    plt.figure(figsize=(12, 6))
    cluster_centers = profile.groupby('cluster_label').mean()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']

    for i in range(n_clusters):
        count = sum(profile['cluster_label'] == i)
        y_values = cluster_centers.loc[i, range(24)].values
        plt.plot(range(24), y_values, marker=markers[i], color=colors[i],
                 linewidth=2.5, label=f'类别 {i} (共 {count} 站)')

    plt.title('伦敦单车四大潮汐门派：24小时平均净流量画像 (Net Flow)', fontsize=16, fontweight='bold')
    plt.xlabel('一天中的时间 (0点 - 23点)', fontsize=12)
    plt.ylabel('净流量 (正数=大量还入/爆仓，负数=大量借出/掏空)', fontsize=12)
    plt.xticks(range(0, 24, 2))
    # 🌟 画一条极其关键的“0水位线”（自平衡线）
    plt.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "02_netflow_profiles.png", dpi=300)
    plt.show()  # 弹出的第二张图！


if __name__ == "__main__":
    main()