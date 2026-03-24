# -*- coding: utf-8 -*-
"""
站点间骑行流向交互式地图
从 3690 万条行程中聚合 OD (Origin-Destination) 流量，
用动画蚂蚁线 (AntPath) 在暗黑底图上展示 Top 流向。
支持：线宽 = 流量大小 / 颜色 = 流量强度 / 动画方向 = 骑行方向
"""
import pandas as pd
import numpy as np
import folium
from folium.plugins import AntPath
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
RENTALS_PATH = ROOT_DIR / "处理后的数据集" / "01_rentals_cleaned_with_features.csv"
STATIONS_PATH = ROOT_DIR / "数据集" / "london_stations.csv"
OUTPUT_DIR = ROOT_DIR / "输出结果" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def flow_color(count, max_count):
    """根据流量大小返回从蓝到红的渐变色"""
    ratio = min(count / max_count, 1.0)
    if ratio < 0.33:
        r, g, b = 80, 180, 255
    elif ratio < 0.66:
        r, g, b = 255, 200, 50
    else:
        r, g, b = 255, 60, 60
    return f'#{r:02x}{g:02x}{b:02x}'


def main():
    print("=" * 55)
    print("  站点间骑行流向地图生成器 (动画蚂蚁线)")
    print("=" * 55)

    # ===== 1. 读取站点坐标 =====
    print("\n>>> 读取站点坐标表...")
    df_stations = pd.read_csv(STATIONS_PATH)
    coord_dict = {}
    for _, row in df_stations.iterrows():
        coord_dict[int(row['station_id'])] = (float(row['latitude']), float(row['longitude']))
    print(f"  已加载 {len(coord_dict)} 个站点坐标")

    # ===== 2. 聚合 OD 流量 =====
    print(">>> 正在读取行程数据并聚合 OD 流量 (数据量约 3700 万，请耐心等待)...")
    df = pd.read_csv(RENTALS_PATH, usecols=['start_station_id', 'end_station_id'])
    df = df.dropna(subset=['start_station_id', 'end_station_id'])
    df['start_station_id'] = df['start_station_id'].astype(int)
    df['end_station_id'] = df['end_station_id'].astype(int)

    # 排除同站借还 (原地还车没有流向意义)
    df = df[df['start_station_id'] != df['end_station_id']]

    od_counts = df.groupby(['start_station_id', 'end_station_id']).size().reset_index(name='trip_count')
    od_counts = od_counts.sort_values('trip_count', ascending=False)
    print(f"  共 {len(od_counts)} 条不同的 OD 路线")

    # ===== 3. 选取 Top N 流向 =====
    top_n = 150
    od_top = od_counts.head(top_n).copy()
    max_count = od_top['trip_count'].max()
    min_count = od_top['trip_count'].min()
    print(f"  取 Top {top_n} 条流向 (最多 {max_count} 次, 最少 {min_count} 次)")

    # ===== 4. 绘制地图 =====
    print(">>> 正在绘制流向动画地图...")
    london_map = folium.Map(location=[51.5074, -0.1278], zoom_start=13, tiles='CartoDB dark_matter')

    # 先打站点底色点 (半透明白色小点)
    drawn_stations = set()
    for _, row in od_top.iterrows():
        for sid in [int(row['start_station_id']), int(row['end_station_id'])]:
            if sid in drawn_stations or sid not in coord_dict:
                continue
            drawn_stations.add(sid)
            lat, lon = coord_dict[sid]
            folium.CircleMarker(
                location=[lat, lon], radius=3,
                color='#ffffff', weight=0.5,
                fill=True, fill_color='#ffffff', fill_opacity=0.4,
            ).add_to(london_map)

    # 画流向蚂蚁线
    for _, row in od_top.iterrows():
        s_id, e_id, cnt = int(row['start_station_id']), int(row['end_station_id']), int(row['trip_count'])
        if s_id not in coord_dict or e_id not in coord_dict:
            continue

        s_lat, s_lon = coord_dict[s_id]
        e_lat, e_lon = coord_dict[e_id]

        weight = 1.5 + 4.5 * (cnt - min_count) / (max_count - min_count + 1)
        color = flow_color(cnt, max_count)

        AntPath(
            locations=[[s_lat, s_lon], [e_lat, e_lon]],
            color=color,
            weight=weight,
            opacity=0.7,
            delay=1200,
            dash_array=[10, 20],
            pulse_color='#FFFFFF',
            popup=f"<b>流向:</b> 站{s_id} → 站{e_id}<br><b>行程数:</b> {cnt:,} 次",
        ).add_to(london_map)

    # 添加图例
    legend_html = '''
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:rgba(0,0,0,0.8);padding:14px 18px;border-radius:8px;
                font-size:13px;color:white;font-family:SimHei,sans-serif;line-height:1.8;">
        <b>骑行流向图 (Top 150)</b><br>
        <span style="color:#50b4ff;">━━</span> 低流量<br>
        <span style="color:#ffc832;">━━</span> 中流量<br>
        <span style="color:#ff3c3c;">━━</span> 高流量<br>
        <span style="font-size:11px;color:#aaa;">线宽 ∝ 行程数 | 动画方向 = 骑行方向</span>
    </div>
    '''
    london_map.get_root().html.add_child(folium.Element(legend_html))

    save_path = OUTPUT_DIR / "London_Bike_Flow_Map.html"
    london_map.save(save_path)
    print(f"\n  流向地图已保存: {save_path}")
    print("  请用浏览器打开 .html 文件查看动画效果！")


if __name__ == "__main__":
    main()
