# -*- coding: utf-8 -*-
"""
精简版：全模型独立高级可视化生成器 (长短周期全能版)
包含：短周期 (1h/6h/24h) 黄金视距 + 长周期 (1周/1月) 滚动拼接宏观视距
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
DATA_IN_PATH = ROOT_DIR / "处理后的数据集" / "02_hourly_start_count.csv"
REG_FIG_DIR = ROOT_DIR / "输出结果" / "回归任务效果图"
REG_FIG_DIR.mkdir(parents=True, exist_ok=True)

TSLIB_RESULTS = CURRENT_DIR / "Time_Series_Library" / "results"
NSTF_RESULTS = CURRENT_DIR / "Nonstationary_Transformers" / "results"


def load_and_prepare_data():
    df = pd.read_csv(DATA_IN_PATH)
    top_station = df.groupby('start_station_id')['start_count'].sum().idxmax()
    df_top = df[df['start_station_id'] == top_station].copy()
    df_top['datetime'] = pd.to_datetime(df_top['date_str']) + pd.to_timedelta(df_top['hour'], unit='h')
    df_top = df_top.sort_values('datetime').reset_index(drop=True)
    return df_top


def plot_zoomed_forecast(hist_dt, hist_y, test_dt, true_y, pred_y, horizon, model_name, save_name):
    """自适应视距绘图器：自动处理超长序列的样式渲染"""
    # 针对 1 个月的超长图，拉长画布，防止挤在一起
    fig_width = 24 if horizon > 200 else 16
    plt.figure(figsize=(fig_width, 6))

    last_hist_dt = hist_dt[-1]
    last_hist_y = hist_y[-1]

    plot_dt_true = np.insert(test_dt[:horizon], 0, last_hist_dt)
    plot_y_true = np.insert(true_y[:horizon], 0, last_hist_y)

    plot_dt_pred = np.insert(test_dt[:horizon], 0, last_hist_dt)
    plot_y_pred = np.insert(pred_y[:horizon], 0, last_hist_y)

    # 历史背景线
    plt.plot(hist_dt, hist_y, label='History (Context)', color='tab:blue', linewidth=1.8)

    # 🌟 智能样式：如果是超长周期 (1周/1个月)，取消 marker 点，调细线条，防止图糊成一团
    msize_true = 5 if horizon <= 48 else 0
    msize_pred = 6 if horizon <= 48 else 0
    lw = 2.0 if horizon <= 48 else 1.2
    alpha_pred = 0.9 if horizon <= 48 else 0.75

    plt.plot(plot_dt_true, plot_y_true, label='True', color='tab:orange',
             linewidth=lw, marker='o', markersize=msize_true)
    plt.plot(plot_dt_pred, plot_y_pred, label='Predicted', color='tab:green',
             linewidth=lw, marker='X', markersize=msize_pred, alpha=alpha_pred)

    # 标题自动适应
    horizon_str = f"{horizon}h ({horizon // 24} Days)" if horizon >= 24 else f"{horizon}h"
    plt.title(f'{model_name} Prediction ({horizon_str} Horizon) - Macro View', fontsize=17, fontweight='bold')
    plt.xlabel('Date Time', fontsize=13)
    plt.ylabel('Rental Count', fontsize=13)
    plt.legend(loc='upper left', fontsize=12)

    # 高亮未来的预测区域
    plt.axvspan(last_hist_dt, test_dt[horizon - 1] if horizon > 1 else test_dt[0], color='yellow', alpha=0.1)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(REG_FIG_DIR / save_name, dpi=300)
    plt.close()
    print(f"    -> 生成完毕: {save_name}")


def _find_forecast_result(results_dir, keyword):
    if not results_dir.exists(): return None, None
    for d in results_dir.iterdir():
        if d.is_dir() and keyword in d.name and (d / 'pred.npy').exists():
            return np.load(d / 'pred.npy'), np.load(d / 'true.npy')
    return None, None


def stitch_long_term_sequence(dl_pred, dl_true, start_chunk_idx, target_horizon, chunk_size=24):
    """
    🌟 核心魔改魔法：滚动拼接术！
    把多个连续的 24h 预测块，像接火车车厢一样拼成 1周 或 1个月 的超长序列。
    """
    pred_stitched = []
    true_stitched = []
    num_chunks = target_horizon // chunk_size

    for i in range(num_chunks):
        # 每次跨越 24 个小时去取下一个完全不重叠的预测块
        current_idx = start_chunk_idx + (i * chunk_size)
        if current_idx >= len(dl_pred):
            print(f"      [!] 警告: 测试集数据不足以支撑拼接 {target_horizon}h，在 {i * chunk_size}h 处截断。")
            break
        pred_stitched.extend(dl_pred[current_idx, :, 0])
        true_stitched.extend(dl_true[current_idx, :, 0])

    return np.array(pred_stitched), np.array(true_stitched)


def main():
    print("=" * 60)
    print("  深度学习：长短周期全尺寸预测可视化 ")
    print("=" * 60)

    df = load_and_prepare_data()
    num_train = int(len(df) * 0.7)
    num_test = int(len(df) * 0.2)

    # 🌟 核心修复：计算训练集的均值和标准差，用于还原数据
    train_df = df.iloc[:num_train]
    train_mean = train_df['start_count'].mean()
    train_std = train_df['start_count'].std()

    test_df = df.iloc[-num_test:]
    seq_len = 96

    models_to_check = [
        ("TimesNet", TSLIB_RESULTS, 'long_term_forecast_LondonBike_TimesNet'),
        ("NS-Transformer", NSTF_RESULTS, 'ns_Transformer'),
        ("Mamba", TSLIB_RESULTS, 'Mamba')
    ]

    for model_name, folder, keyword in models_to_check:
        print(f"\n>>> 正在处理模型: {model_name}")
        dl_pred, dl_true = _find_forecast_result(folder, keyword)

        if dl_pred is not None:
            # 1. 寻找合适的起跳点 (找一个有明显波峰的早期位置，确保后面有足够的数据拼 1 个月)
            search_range = min(1000, len(dl_true) - 720)  # 留出 720h 的余量
            best_chunk_idx = np.argmax(np.max(dl_true[:search_range, :, 0], axis=1))

            start_idx_in_test = seq_len + best_chunk_idx

            # ================= (A) 绘制短期预测 (原逻辑) =================
            print("  -> [A] 生成短期视距 (1h, 6h, 24h) ...")
            true_24h = dl_true[best_chunk_idx, :, 0]
            pred_24h = dl_pred[best_chunk_idx, :, 0]

            for h in [1, 6, 24]:
                context_len = max(72, h * 3)
                global_start = len(df) - len(test_df) + start_idx_in_test
                hist_df = df.iloc[global_start - context_len: global_start]
                test_dt_seg = test_df['datetime'].iloc[start_idx_in_test: start_idx_in_test + h].values

                plot_zoomed_forecast(
                    hist_df['datetime'].values, hist_df['start_count'].values,
                    test_dt_seg, true_24h, pred_24h, h,
                    model_name, f"{model_name.replace('-', '')}_Short_{h}h.png"
                )

            # ================= (B) 绘制超长宏观预测 (滚动拼接) =================
                # ================= (B) 绘制超长宏观预测 (滚动拼接) =================
            print("  -> [B] 正在执行滚动拼接，生成宏观视距 (1周, 1个月) ...")
            for h in [168, 720]:
                pred_long, true_long = stitch_long_term_sequence(dl_pred, dl_true, best_chunk_idx, h)

                    # 🌟 修复：反归一化还原为真实单车数量
                pred_long = pred_long * train_std + train_mean
                true_long = true_long * train_std + train_mean

                actual_h = len(pred_long)

                # 宏观视距的历史背景保留 7 天 (168小时)
                context_len_macro = 168
                global_start = len(df) - len(test_df) + start_idx_in_test
                hist_df = df.iloc[global_start - context_len_macro: global_start]

                test_dt_seg = test_df['datetime'].iloc[start_idx_in_test: start_idx_in_test + actual_h].values

                plot_zoomed_forecast(
                    hist_df['datetime'].values, hist_df['start_count'].values,
                    test_dt_seg, true_long, pred_long, actual_h,
                    model_name, f"{model_name.replace('-', '')}_Macro_{h}h.png"
                )

        else:
            print(f"  [!] 未找到 {model_name} 的预测数据，跳过。")


if __name__ == "__main__":
    main()