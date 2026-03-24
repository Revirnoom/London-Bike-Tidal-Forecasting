# -*- coding: utf-8 -*-
"""
阶段三：LightGBM 模型专属测试脚本
黄金视距版：截取 3~7 天平滑历史背景 + 完美拼接 1h/6h/24h 预测，彻底解决数据挤压重叠问题
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import lightgbm as lgb

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
DATA_IN_PATH = ROOT_DIR / "处理后的数据集" / "02_hourly_start_count.csv"
FIGURES_OUT_DIR = ROOT_DIR / "输出结果" / "回归任务效果图"
FIGURES_OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    df = pd.read_csv(DATA_IN_PATH)
    top_station = df.groupby('start_station_id')['start_count'].sum().idxmax()
    df_top = df[df['start_station_id'] == top_station].copy()
    df_top['datetime'] = pd.to_datetime(df_top['date_str']) + pd.to_timedelta(df_top['hour'], unit='h')
    df_top = df_top.sort_values('datetime').reset_index(drop=True)

    target = 'start_count'
    df_top['lag_1'] = df_top[target].shift(1)
    df_top['lag_24'] = df_top[target].shift(24)
    df_top['lag_168'] = df_top[target].shift(168)
    df_top = df_top.dropna().reset_index(drop=True)
    features_all = ['hour', 'is_weekend', 'is_holiday', 'lag_1', 'lag_24', 'lag_168', 'mean_temp', 'precipitation']
    return df_top, target, features_all


def plot_zoomed_forecast(hist_dt, hist_y, test_dt, true_y, pred_y, horizon, model_name, save_name):
    """黄金视距绘图：展示连续的潮汐波浪，强化单点/短序列的视觉锚点"""
    plt.figure(figsize=(16, 6))

    # 获取历史数据的最后一个点，作为“分叉”的起点
    last_hist_dt = hist_dt[-1]
    last_hist_y = hist_y[-1]

    # 🌟 核心修复：把真实值和预测值，都往回连到历史数据的最后一个点上！
    plot_dt_true = np.insert(test_dt[:horizon], 0, last_hist_dt)
    plot_y_true = np.insert(true_y[:horizon], 0, last_hist_y)

    plot_dt_pred = np.insert(test_dt[:horizon], 0, last_hist_dt)
    plot_y_pred = np.insert(pred_y[:horizon], 0, last_hist_y)

    # 画蓝色的历史背景线
    plt.plot(hist_dt, hist_y, label='History (Context)', color='tab:blue', linewidth=1.8)

    # 🌟 核心修复：加上 marker='o' (圆点) 和 marker='X' (叉号)！
    # 这样哪怕是 1h 预测（只有1个点），也能在图上亮起一颗星星！
    plt.plot(plot_dt_true, plot_y_true, label='True', color='tab:orange',
             linewidth=2.0, marker='o', markersize=5)
    plt.plot(plot_dt_pred, plot_y_pred, label='Predicted', color='tab:green',
             linewidth=2.0, marker='X', markersize=6, alpha=0.9)

    plt.title(f'{model_name} Prediction (Next {horizon}h Horizon) - Zoomed View', fontsize=17, fontweight='bold')
    plt.xlabel('Date Time', fontsize=13)
    plt.ylabel('Rental Count', fontsize=13)
    plt.legend(loc='upper left', fontsize=12)

    # 高亮未来的预测区域
    plt.axvspan(last_hist_dt, test_dt[horizon - 1] if horizon > 1 else test_dt[0], color='yellow', alpha=0.1)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_OUT_DIR / save_name, dpi=300)
    plt.close()
    print(f"    -> 生成完毕: {save_name}")


def run_experiments():
    df, target, features_full = load_and_prepare_data()

    total_len = len(df)
    num_train = int(total_len * 0.7)
    num_test = int(total_len * 0.2)

    train_df = df.iloc[:num_train]
    test_df = df.iloc[-num_test:]

    X_train, y_train = train_df[features_full], train_df[target]
    X_test, y_test = test_df[features_full], test_df[target]

    print(">>> 正在训练 LightGBM 模型...")
    lgbm = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
    lgbm.fit(X_train, y_train)
    preds_lgbm = lgbm.predict(X_test)

    # 🌟 自动雷达：寻找测试集中包含最大波峰的那一天
    peak_idx = y_test.values.argmax()
    best_start = max(0, peak_idx - 12)  # 把高峰放在预测窗口的中间或偏后位置，视觉最震撼

    print(">>> 正在生成黄金视距预测图像 (1h, 6h, 24h)...")
    for h in [1, 6, 24]:
        # 🌟 动态历史窗口：最少保留 3 天 (72小时) 的历史，足够展示潮汐周期
        context_len = max(72, h * 3)

        global_start = len(df) - len(y_test) + best_start
        hist_df = df.iloc[global_start - context_len: global_start]

        test_seg = test_df.iloc[best_start: best_start + h]
        test_pred_seg = preds_lgbm[best_start: best_start + h]

        plot_zoomed_forecast(
            hist_df['datetime'].values, hist_df[target].values,
            test_seg['datetime'].values, test_seg[target].values, test_pred_seg,
            h, "LightGBM", f"LGBM_Advanced_{h}h.png"
        )
    print(">>> 生成完毕")


if __name__ == "__main__":
    run_experiments()