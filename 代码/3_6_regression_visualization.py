# -*- coding: utf-8 -*-
"""
阶段三：回归模型核心可视化生成器 (大模型专注版 + LightGBM 深度看板)
重点聚焦：全模型性能分析看板与对比
输出路径：硬编码指定为 D:\pythonProject\数据挖掘\数据挖掘课程设计\输出结果\回归任务效果图
数据划分：严格 7:1:2
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings

from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# =================== 核心路径配置 ===================
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
DATA_IN_PATH = ROOT_DIR / "处理后的数据集" / "02_hourly_start_count.csv"

FIGURES_OUT_DIR = Path(r"D:\pythonProject\数据挖掘\数据挖掘课程设计\输出结果\回归任务效果图")
FIGURES_OUT_DIR.mkdir(parents=True, exist_ok=True)

TSLIB_RESULTS_DIR = CURRENT_DIR / "Time_Series_Library" / "results"
NSTF_RESULTS_DIR = CURRENT_DIR / "Nonstationary_Transformers" / "results"


# =================== 数据与模型准备 ===================

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


def train_lightgbm_baseline(df, target, features_all):
    total_len = len(df)
    num_train = int(total_len * 0.7)
    num_test = int(total_len * 0.2)

    train_df = df.iloc[:num_train]
    test_df = df.iloc[-num_test:]

    X_train, y_train = train_df[features_all], train_df[target]
    X_test, y_test = test_df[features_all], test_df[target]

    lgbm = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
    lgbm.fit(X_train, y_train)
    preds = lgbm.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    metrics = {'MAE': mae, 'RMSE': rmse}

    # 🌟 修改点：除了指标，也把 1D 的真实值和预测值返回
    return metrics, y_test.values, preds


def convert_1d_to_3d_tensor(y_true_1d, y_pred_1d, horizon=24):
    """🌟 核心降维打击魔法：把 LightGBM 的一维预测切片组装成大模型标准 3D 张量"""
    N = len(y_true_1d) - horizon + 1
    ts_pred = np.zeros((N, horizon, 1))
    ts_true = np.zeros((N, horizon, 1))

    # 使用滑动窗口切分
    for i in range(N):
        ts_pred[i, :, 0] = y_pred_1d[i: i + horizon]
        ts_true[i, :, 0] = y_true_1d[i: i + horizon]

    # 计算 3D 张量的全局评估指标
    mae = np.mean(np.abs(ts_pred - ts_true))
    mse = np.mean((ts_pred - ts_true) ** 2)
    rmse = np.sqrt(mse)

    epsilon = 1e-8
    mape = np.mean(np.abs((ts_pred - ts_true) / (ts_true + epsilon))) * 100
    mspe = np.mean(np.square((ts_pred - ts_true) / (ts_true + epsilon))) * 100

    metrics = [mae, mse, rmse, mape, mspe]
    return ts_pred, ts_true, metrics


def load_dl_results(results_dir, keyword):
    if not results_dir.exists(): return None, None, None
    for d in results_dir.iterdir():
        if d.is_dir() and keyword in d.name and "classification" not in d.name:
            pred_path = d / 'pred.npy'
            met_path = d / 'metrics.npy'
            if pred_path.exists() and met_path.exists():
                return np.load(pred_path), np.load(d / 'true.npy'), np.load(met_path)
    return None, None, None


# =================== 图表生成模块 ===================

def plot_dl_comprehensive_card(ts_pred, ts_true, ts_metrics, model_name, save_name, main_color):
    """通用的性能看版画图器（现在也兼容 LightGBM 了）"""
    print(f">>> 绘制: {model_name} 专属性能看板 ...")
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])
    np.random.seed(42)
    sample_ids = np.random.choice(len(ts_pred), size=6, replace=False)
    sample_ids.sort()

    pred_concat, true_concat, boundaries, offset = [], [], [], 0
    for sid in sample_ids:
        pred_concat.extend(ts_pred[sid, :, 0])
        true_concat.extend(ts_true[sid, :, 0])
        boundaries.append(offset)
        offset += len(ts_pred[sid, :, 0])

    x_axis = np.arange(len(pred_concat))
    ax1.plot(x_axis, true_concat, color='black', linewidth=2, label='真实值', zorder=5)
    ax1.plot(x_axis, pred_concat, color=main_color, linewidth=1.8, linestyle='--', label=f'{model_name} 预测',
             alpha=0.9)
    for b in boundaries[1:]:
        ax1.axvline(b, color='grey', linestyle=':', alpha=0.5)
    ax1.set_title(f'{model_name} 24小时预测轨迹 (6段随机测试样本无缝拼接)', fontsize=15, fontweight='bold')
    ax1.set_xlabel('预测步长时间轴 (小时)', fontsize=12)
    ax1.set_ylabel('归一化单车需求量' if model_name != "LightGBM" else '单车需求量', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2 = fig.add_subplot(gs[1, 0])
    step_mae = [np.mean(np.abs(ts_pred[:, h, 0] - ts_true[:, h, 0])) for h in range(ts_pred.shape[1])]

    ax2.bar(range(1, 25), step_mae, color='#bdc3c7', edgecolor='white', alpha=0.5)
    ax2.plot(range(1, 25), step_mae, color=main_color, marker='o', markersize=5, linewidth=2.0)
    ax2.set_title('预测误差的远期衰减 (长序列抗压能力)', fontsize=15, fontweight='bold')
    ax2.set_xlabel('预测步长 (h+1 到 h+24)', fontsize=12)
    ax2.set_ylabel('平均绝对误差 (MAE)', fontsize=12)
    ax2.set_xticks(range(1, 25, 2))
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')
    metric_names = ['MAE (平绝对误差)', 'MSE (均方误差)', 'RMSE (均方根误差)', 'MAPE (平绝对百分比)', 'MSPE']
    metric_vals = ts_metrics[:5]
    card_y_positions = np.linspace(0.85, 0.15, len(metric_names))
    card_colors = [main_color, '#34495e', '#7f8c8d', '#95a5a6', '#bdc3c7']

    for mn, mv, cy, cc in zip(metric_names, metric_vals, card_y_positions, card_colors):
        ax3.add_patch(plt.Rectangle((0.05, cy - 0.06), 0.9, 0.12, facecolor=cc, alpha=0.1, edgecolor=cc, linewidth=2,
                                    transform=ax3.transAxes, clip_on=False))
        ax3.text(0.10, cy, mn, fontsize=15, fontweight='bold', color=cc, transform=ax3.transAxes, va='center')
        ax3.text(0.80, cy, f'{mv:.4f}', fontsize=18, fontweight='bold', color='#2c3e50', transform=ax3.transAxes,
                 va='center', ha='center')

    ax3.set_title(f'{model_name} 核心评估指标一览', fontsize=15, fontweight='bold', pad=15)
    fig.suptitle(
        f'机器学习基线：{model_name} 回归性能深度分析' if model_name == "LightGBM" else f'深度学习前沿模型：{model_name} 回归性能深度分析',
        fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_OUT_DIR / save_name, dpi=300, bbox_inches='tight')
    plt.close()


def plot_dl_mae_decay_comparison(timesnet_pred, timesnet_true, nstf_pred, nstf_true, mamba_pred, mamba_true):
    print(">>> 绘制: 三大顶会模型预测衰减对抗图 ...")
    plt.figure(figsize=(14, 7))
    t_mae = [np.mean(np.abs(timesnet_pred[:, h, 0] - timesnet_true[:, h, 0])) for h in range(24)]
    n_mae = [np.mean(np.abs(nstf_pred[:, h, 0] - nstf_true[:, h, 0])) for h in range(24)]
    m_mae = [np.mean(np.abs(mamba_pred[:, h, 0] - mamba_true[:, h, 0])) for h in range(24)]

    plt.plot(range(1, 25), t_mae, label='TimesNet (ICLR 2023)', color='#3498db', marker='s', markersize=6,
             linewidth=2.5)
    plt.plot(range(1, 25), n_mae, label='NS-Transformer (NeurIPS 2022)', color='#e74c3c', marker='^', markersize=6,
             linewidth=2.5)
    plt.plot(range(1, 25), m_mae, label='Mamba (SSM Latest)', color='#27ae60', marker='o', markersize=6, linewidth=2.5)

    plt.title('24小时长序列抗衰减能力对抗：TimesNet vs NS-Transformer vs Mamba', fontsize=16, fontweight='bold')
    plt.xlabel('预测步长 (未来第 N 个小时)', fontsize=13)
    plt.ylabel('平均绝对误差 (MAE) - 越低越好', fontsize=13)
    plt.xticks(range(1, 25))
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_OUT_DIR / "04_DL_MAE_Decay_Comparison.png", dpi=300)
    plt.close()


def plot_overall_metrics_bar(lgbm_metrics, timesnet_metrics, nstf_metrics, mamba_metrics):
    print(">>> 绘制: 全模型宏观指标横向对比图 ...")
    models = ['LightGBM\n(机器学习基线)', 'NS-Transformer\n(NeurIPS 2022)', 'TimesNet\n(ICLR 2023)',
              'Mamba\n(状态空间模型)']
    mae_vals = [lgbm_metrics['MAE'], nstf_metrics[0], timesnet_metrics[0], mamba_metrics[0]]
    rmse_vals = [lgbm_metrics['RMSE'], nstf_metrics[2], timesnet_metrics[2], mamba_metrics[2]]

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(13, 7))
    rects1 = ax.bar(x - width / 2, mae_vals, width, label='MAE (平均绝对误差)', color='#2ecc71', edgecolor='white')
    rects2 = ax.bar(x + width / 2, rmse_vals, width, label='RMSE (均方根误差)', color='#f39c12', edgecolor='white')

    ax.set_ylabel('误差数值 (越小越好)', fontsize=13)
    ax.set_title('伦敦共享单车预测：全模型性能终极对决', fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 5),
                        textcoords="offset points", ha='center', va='bottom', fontsize=12, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    plt.savefig(FIGURES_OUT_DIR / "05_Overall_Metrics_Comparison.png", dpi=300)
    plt.close()


# =================== 主函数 ===================

def main():
    print("=" * 60)
    print("  回归全模型性能可视化生成器启动")
    print(f"  强制输出目录: {FIGURES_OUT_DIR}")
    print("=" * 60)

    df, target, features_all = load_and_prepare_data()
    print("\n[阶段 1/3] 正在后台计算基线模型 (LightGBM) 得分并转换 3D 张量 ...")
    lgbm_base_metrics, lgbm_y_true, lgbm_y_pred = train_lightgbm_baseline(df, target, features_all)

    # 🌟 将 LGBM 转换为与 DL 模型相同的 3D 格式
    l_pred_3d, l_true_3d, l_met_3d = convert_1d_to_3d_tensor(lgbm_y_true, lgbm_y_pred)

    print("\n[阶段 2/3] 正在加载大模型成绩单 ...")
    t_pred, t_true, t_met = load_dl_results(TSLIB_RESULTS_DIR, 'TimesNet')
    n_pred, n_true, n_met = load_dl_results(NSTF_RESULTS_DIR, 'ns_Transformer')
    m_pred, m_true, m_met = load_dl_results(TSLIB_RESULTS_DIR, 'Mamba')

    if t_pred is None or n_pred is None or m_pred is None:
        print("\n [!] 警告：未能找齐深度学习数据，部分图表将无法生成。")
        return

    print("\n[阶段 3/3] 正在生成专属深度分析图表 ...")
    # 🌟 为 LightGBM 绘制专属性能看板 (使用橙色高亮)
    plot_dl_comprehensive_card(l_pred_3d, l_true_3d, l_met_3d, "LightGBM", "00_LightGBM_Deep_Analysis.png", "#f39c12")

    plot_dl_comprehensive_card(t_pred, t_true, t_met, "TimesNet", "01_TimesNet_Deep_Analysis.png", "#3498db")
    plot_dl_comprehensive_card(n_pred, n_true, n_met, "NS-Transformer", "02_NSTransformer_Deep_Analysis.png", "#e74c3c")
    plot_dl_comprehensive_card(m_pred, m_true, m_met, "Mamba", "03_Mamba_Deep_Analysis.png", "#27ae60")

    plot_dl_mae_decay_comparison(t_pred, t_true, n_pred, n_true, m_pred, m_true)
    plot_overall_metrics_bar(lgbm_base_metrics, t_met, n_met, m_met)

    print("正在加载 Mamba 消融实验成绩单 ...")
    # 🌟 修改点：精准搜索消融实验的关键词
    std_pred, std_true, std_met = load_dl_results(TSLIB_RESULTS_DIR, 'bike_Ablation_Standard')
    bi_pred, bi_true, bi_met = load_dl_results(TSLIB_RESULTS_DIR, 'bike_Ablation_Bidirectional')

    if std_pred is None or bi_pred is None:
        print("\n [!] 警告：未能找齐消融实验数据，请检查 results 文件夹名是否包含关键字。")
        return

    print("正在生成消融实验对比图表 ...")

    # 绘制标准版的分析看板
    plot_dl_comprehensive_card(std_pred, std_true, std_met, "Mamba-Standard", "06_Mamba_Standard_Analysis.png",
                               "#3498db")

    # 绘制双向改进版的分析看板
    plot_dl_comprehensive_card(bi_pred, bi_true, bi_met, "Mamba-Bidirectional", "07_Mamba_Bidir_Analysis.png",
                               "#e74c3c")

    # 绘制两者的误差衰减对比图 (利用你原有的对比函数，稍作参数调整即可)
    # 你可以修改一下 plot_dl_mae_decay_comparison 的调用，或者直接对比这两个

    print("\n" + "=" * 60)
    print("  大功告成！全系图表已成功生成。")
    print(f"  请前往查收您的报告素材：{FIGURES_OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()