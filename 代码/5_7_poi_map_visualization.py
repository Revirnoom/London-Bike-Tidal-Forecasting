# -*- coding: utf-8 -*-
"""
三大分类任务的站点空间投影地图 (Folium 暗黑底图)
  地图1: 潮汐净流量分类 — 清晨流出型 / 自平衡型 / 清晨汇聚型
  地图2: 常规站点功能分类 — 边缘冷门站 / 傍晚休闲站 / 早高峰潮汐站 / 双峰通勤站
  地图3: 疫情抗压演变分类 — 普通缩水区 / 刚需避风港 / 永恒长尾区 / 脆弱通勤王
"""
import pandas as pd
import folium
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
STATIONS_PATH = ROOT_DIR / "数据集" / "london_stations.csv"
OUTPUT_DIR = ROOT_DIR / "输出结果" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_and_merge(labels_path, station_id_col):
    """读取标签文件并与官方站点坐标合并"""
    if not labels_path.exists():
        print(f"  [!] 找不到标签文件：{labels_path}")
        return None

    df_stations = pd.read_csv(STATIONS_PATH)
    df_labels = pd.read_csv(labels_path)

    label_col = 'cluster_label' if 'cluster_label' in df_labels.columns else df_labels.columns[-1]

    df_labels[station_id_col] = df_labels[station_id_col].astype(int)
    df_stations['station_id'] = df_stations['station_id'].astype(int)

    df_merged = pd.merge(df_stations, df_labels[[station_id_col, label_col]],
                         left_on='station_id', right_on=station_id_col, how='inner')
    print(f"  成功匹配到 {len(df_merged)} 个站点")
    return df_merged, label_col


def _build_map(df_merged, label_col, color_map, class_names, map_title):
    """在暗黑底图上打彩色站点并添加图例"""
    london_map = folium.Map(location=[51.5074, -0.1278], zoom_start=12, tiles='CartoDB dark_matter')

    for _, row in df_merged.iterrows():
        cluster_id = int(row[label_col])
        color = color_map.get(cluster_id, '#FFFFFF')
        class_label = class_names.get(cluster_id, f"类别 {cluster_id}")

        lat, lon = float(row['latitude']), float(row['longitude'])
        name = str(row['station_name']).replace("'", "`")

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=color,
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=f"<b>{name}</b><br>类别 {cluster_id}: {class_label}",
            tooltip=f"{class_label}"
        ).add_to(london_map)

    legend_items = "".join(
        f'<li><span style="background:{c};width:12px;height:12px;'
        f'display:inline-block;margin-right:6px;border-radius:50%;"></span>'
        f'{class_names[k]}</li>'
        for k, c in sorted(color_map.items())
    )
    legend_html = f'''
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:rgba(0,0,0,0.75);padding:12px 16px;border-radius:8px;
                font-size:13px;color:white;font-family:SimHei,sans-serif;">
        <b>{map_title}</b>
        <ul style="list-style:none;padding:4px 0;margin:0;">{legend_items}</ul>
    </div>
    '''
    london_map.get_root().html.add_child(folium.Element(legend_html))
    return london_map


# =====================================================================
# 地图1: 潮汐净流量分类 (3 类)
# =====================================================================
def create_tidal_map():
    print("\n>>> [地图1] 潮汐净流量分类空间投影...")
    labels_path = CURRENT_DIR / "分类数据准备_潮汐净流量" / "station_labels_netflow.csv"
    result = _load_and_merge(labels_path, 'station_id')
    if result is None:
        return

    df_merged, label_col = result
    color_map = {0: '#FF3333', 1: '#33CCFF', 2: '#33FF33'}
    class_names = {0: '清晨流出型 (住宅区)', 1: '自平衡型 (普通区)', 2: '清晨汇聚型 (商业区)'}

    m = _build_map(df_merged, label_col, color_map, class_names, "潮汐净流量分类")
    save_path = OUTPUT_DIR / "London_Ultimate_Tidal_Map.html"
    m.save(save_path)
    print(f"  地图已保存: {save_path}")


# =====================================================================
# 地图2: 常规站点功能分类 (4 类)
# =====================================================================
def create_function_map():
    print("\n>>> [地图2] 常规站点功能分类空间投影...")
    labels_path = CURRENT_DIR / "分类数据准备" / "station_labels.csv"
    result = _load_and_merge(labels_path, 'start_station_id')
    if result is None:
        return

    df_merged, label_col = result
    color_map = {
        0: '#3498DB',
        1: '#F39C12',
        2: '#E74C3C',
        3: '#2ECC71',
    }
    class_names = {
        0: '边缘冷门站',
        1: '傍晚休闲站',
        2: '早高峰潮汐站',
        3: '双峰通勤站',
    }

    m = _build_map(df_merged, label_col, color_map, class_names, "常规站点功能分类")
    save_path = OUTPUT_DIR / "London_Station_Function_Map.html"
    m.save(save_path)
    print(f"  地图已保存: {save_path}")


# =====================================================================
# 地图3: 疫情抗压演变分类 (4 类)
# =====================================================================
def create_pandemic_map():
    print("\n>>> [地图3] 疫情抗压演变分类空间投影...")
    labels_path = CURRENT_DIR / "分类数据准备_疫情演变" / "station_labels_pandemic.csv"
    result = _load_and_merge(labels_path, 'start_station_id')
    if result is None:
        return

    df_merged, label_col = result
    color_map = {
        0: '#95A5A6',
        1: '#27AE60',
        2: '#8E44AD',
        3: '#E74C3C',
    }
    class_names = {
        0: '普通缩水区',
        1: '刚需避风港',
        2: '永恒长尾区',
        3: '脆弱通勤王',
    }

    m = _build_map(df_merged, label_col, color_map, class_names, "疫情抗压演变分类")
    save_path = OUTPUT_DIR / "London_Pandemic_Resilience_Map.html"
    m.save(save_path)
    print(f"  地图已保存: {save_path}")


# =====================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  三大分类任务站点空间投影地图生成器")
    print("=" * 50)
    create_tidal_map()
    create_function_map()
    create_pandemic_map()
    print("\n 全部地图生成完毕！")
