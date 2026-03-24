# -*- coding: utf-8 -*-
"""
全量骑行流向地图 (50万+ OD 路线)
使用轻量 PolyLine + 低透明度叠加策略：
  路线越密集的走廊自然越亮，形成"流量热辉光"效果
  线宽和透明度均与行程数成正比
"""
import pandas as pd
import numpy as np
import folium
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
RENTALS_PATH = ROOT_DIR / "处理后的数据集" / "01_rentals_cleaned_with_features.csv"
STATIONS_PATH = ROOT_DIR / "数据集" / "london_stations.csv"
OUTPUT_DIR = ROOT_DIR / "输出结果" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def flow_color(ratio):
    """ratio 0~1 → 冷蓝 → 暖黄 → 亮红 连续渐变"""
    if ratio < 0.5:
        t = ratio / 0.5
        r = int(80 + (255 - 80) * t)
        g = int(180 + (200 - 180) * t)
        b = int(255 + (50 - 255) * t)
    else:
        t = (ratio - 0.5) / 0.5
        r = int(255)
        g = int(200 - (200 - 50) * t)
        b = int(50 - (50 - 30) * t)
    return f'#{max(0,min(255,r)):02x}{max(0,min(255,g)):02x}{max(0,min(255,b)):02x}'


def main():
    print("=" * 55)
    print("  全量骑行流向地图生成器 (PolyLine 热辉光版)")
    print("=" * 55)

    # ===== 1. 读取站点坐标 =====
    print("\n>>> 读取站点坐标表...")
    df_stations = pd.read_csv(STATIONS_PATH)
    coord_dict = {}
    for _, row in df_stations.iterrows():
        coord_dict[int(row['station_id'])] = (float(row['latitude']), float(row['longitude']))
    print(f"  已加载 {len(coord_dict)} 个站点坐标")

    # ===== 2. 聚合 OD 流量 =====
    print(">>> 正在读取行程数据并聚合 OD 流量 (约 3700 万条，请耐心等待)...")
    df = pd.read_csv(RENTALS_PATH, usecols=['start_station_id', 'end_station_id'])
    df = df.dropna(subset=['start_station_id', 'end_station_id'])
    df['start_station_id'] = df['start_station_id'].astype(int)
    df['end_station_id'] = df['end_station_id'].astype(int)
    df = df[df['start_station_id'] != df['end_station_id']]

    od_counts = df.groupby(['start_station_id', 'end_station_id']).size().reset_index(name='trip_count')
    total_routes = len(od_counts)
    print(f"  共 {total_routes:,} 条不同的 OD 路线")

    # 取流量排名前 10% 的路线（过滤掉大量低频噪声，保留主干流向）
    top_ratio = 0.005
    top_count = int(len(od_counts) * top_ratio)
    od_filtered = od_counts.head(top_count).copy()
    print(f"  取 Top {top_ratio:.0%} 共 {len(od_filtered):,} 条路线 "
          f"(覆盖 {od_filtered['trip_count'].sum() / od_counts['trip_count'].sum() * 100:.1f}% 的总行程)")

    # 用对数尺度压缩流量范围，避免极端值主导视觉
    od_filtered['log_count'] = np.log1p(od_filtered['trip_count'])
    log_max = od_filtered['log_count'].max()
    log_min = od_filtered['log_count'].min()
    od_filtered['ratio'] = (od_filtered['log_count'] - log_min) / (log_max - log_min + 1e-9)

    # 按流量从低到高排序，让高流量画在最上层
    od_filtered = od_filtered.sort_values('trip_count', ascending=True)

    max_count = od_filtered['trip_count'].max()
    print(f"  最大单条路线行程数: {max_count:,}")

    # ===== 3. 绘制全量流向地图 =====
    print(f">>> 正在绘制 {len(od_filtered):,} 条流向线 (可能需要 1~2 分钟)...")
    london_map = folium.Map(location=[51.5074, -0.1278], zoom_start=13, tiles='CartoDB dark_matter')

    # 打全部涉及站点的底色
    all_stations = set(od_filtered['start_station_id']) | set(od_filtered['end_station_id'])
    for sid in all_stations:
        if sid not in coord_dict:
            continue
        lat, lon = coord_dict[sid]
        folium.CircleMarker(
            location=[lat, lon], radius=2,
            color='#ffffff', weight=0,
            fill=True, fill_color='#ffffff', fill_opacity=0.25,
        ).add_to(london_map)

    drawn = 0
    for _, row in od_filtered.iterrows():
        s_id = int(row['start_station_id'])
        e_id = int(row['end_station_id'])
        if s_id not in coord_dict or e_id not in coord_dict:
            continue

        s_lat, s_lon = coord_dict[s_id]
        e_lat, e_lon = coord_dict[e_id]
        ratio = row['ratio']

        weight = 0.4 + 3.6 * ratio
        opacity = 0.06 + 0.54 * ratio
        color = flow_color(ratio)

        folium.PolyLine(
            locations=[[s_lat, s_lon], [e_lat, e_lon]],
            color=color,
            weight=weight,
            opacity=opacity,
        ).add_to(london_map)
        drawn += 1

    print(f"  已绘制 {drawn:,} 条流向线")

    # 图例
    legend_html = f'''
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:rgba(0,0,0,0.8);padding:14px 18px;border-radius:8px;
                font-size:13px;color:white;font-family:SimHei,sans-serif;line-height:1.8;">
        <b>全量骑行流向图 ({drawn:,} 条路线)</b><br>
        <span style="color:#50b4ff;">━━</span> 低流量路线<br>
        <span style="color:#ffc832;">━━</span> 中流量路线<br>
        <span style="color:#ff3c3c;">━━</span> 高流量路线<br>
        <span style="font-size:11px;color:#aaa;">线越密 = 走廊越繁忙 | 透明度叠加 = 热辉光</span>
    </div>
    '''
    london_map.get_root().html.add_child(folium.Element(legend_html))

    save_path = OUTPUT_DIR / "London_Bike_Flow_Map_Full.html"
    london_map.save(save_path)
    print(f"\n  全量流向地图已保存: {save_path}")
    print("  (文件较大，浏览器打开可能需要几秒加载)")


if __name__ == "__main__":
    main()
