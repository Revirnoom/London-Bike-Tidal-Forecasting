# London-Bike-Tidal-Forecasting

**A Deep Learning Based Tidal Dispatch and Demand Forecasting System for Urban Bike-Sharing**

London-Bike-Tidal-Forecasting is an end-to-end data mining system that addresses the tidal congestion problem in metropolitan bike-sharing networks. It processes **5 GB / 37 million** cycling trip records through a complete pipeline — from data cleaning to deep learning forecasting to geospatial validation.

Language: **English** | [中文](#中文文档)

---

## Major Features

- **Large-scale data engineering.** Chunk-based streaming architecture processes 5 GB raw CSV in 1M-row blocks, preventing OOM crashes while fusing weather and holiday features.

- **Multi-perspective unsupervised clustering.** Three clustering dimensions (basic 24h profile, pandemic resilience, tidal net flow) using K-Means and hierarchical Ward linkage to reveal hidden station functional types.

- **State-of-the-art time-series models.** Integrates TimesNet (ICLR 2023) and Non-stationary Transformer (NeurIPS 2022) for both regression (demand forecasting) and classification (station type identification) tasks.

- **Comprehensive evaluation dashboards.** Generates confusion matrices, ROC curves, radar charts, F1 heatmaps, MAE decay curves, and multi-model comparison bar charts.

- **Geospatial POI validation.** Projects clustering results onto real London maps via Folium, confirming algorithmic labels match physical urban geography (CBD vs. residential vs. mixed-use).

---

## Model Zoo

### Regression (Short-Term Demand Forecasting)

| Model | Conference | Config |
|:---:|:---:|:---:|
| TimesNet | ICLR 2023 | [3_4_run_timesnet.py](代码/3_4_run_timesnet.py) |
| NS-Transformer | NeurIPS 2022 | [3_2_run_transformer.py](代码/3_2_run_transformer.py) |
| Mamba | arXiv 2024 | [3_short_term_prediction.py](代码/3_short_term_prediction.py) |
| LightGBM (Baseline) | — | [3_6_regression_visualization.py](代码/3_6_regression_visualization.py) |

### Classification (Station Functional Type)

| Task | TimesNet | NS-Transformer | Clustering |
|:---:|:---:|:---:|:---:|
| Basic 24h Profile (4-class) | [run](代码/4_3_run_timesnet_classification.py) | [run](代码/4_8_run_nstransformer_classification.py) | K-Means K=4 |
| Pandemic Resilience (4-class) | [run](代码/4_7_run_pandemic_classification.py) | [run](代码/4_9_run_pandemic_nstf.py) | K-Means K=4 |
| Tidal Net Flow (3-class) | [run](代码/5_4_run_timesnet_netflow.py) | [run](代码/5_5_run_nstransformer_netflow.py) | Hierarchical Ward K=3 |

---

## Pipeline Overview

```
Phase 1                Phase 2              Phase 3                Phase 4                Phase 5
┌──────────┐     ┌──────────────┐     ┌───────────────┐     ┌───────────────┐     ┌──────────────────┐
│   Data   │     │  Exploratory │     │  Regression   │     │Classification │     │  Tidal Net Flow  │
│Preprocess│────▶│    Data      │────▶│  Forecasting  │────▶│  (Station     │────▶│  & Geospatial    │
│(5GB CSV) │     │  Analysis    │     │  (TimesNet /  │     │   Functional  │     │  POI Validation  │
│          │     │  (EDA)       │     │   NS-Trans.)  │     │   Typing)     │     │  (Folium Maps)   │
└──────────┘     └──────────────┘     └───────────────┘     └───────────────┘     └──────────────────┘
```

---

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 1.13

### Setup

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
London-Bike-Tidal-Forecasting/
│
├── 代码/                                    # Core source code
│   ├── 1_data_preprocessing.py              # Phase 1: Chunked data cleaning & feature fusion
│   ├── 2_data_exploration.py                # Phase 2: EDA visualization
│   ├── 3_1_prepare_transformer_data.py      # Phase 3: Data formatting for models
│   ├── 3_2_run_transformer.py               #          NS-Transformer launcher
│   ├── 3_3_check_leakage.py                 #          Data leakage audit
│   ├── 3_4_run_timesnet.py                  #          TimesNet launcher
│   ├── 3_5_compare_results.py               #          Multi-model comparison
│   ├── 3_6_regression_visualization.py      #          Regression dashboard generator
│   ├── 4_1_station_clustering.py            # Phase 4: K-Means station profiling
│   ├── 4_2_prepare_ts_dataset.py            #          UEA-format dataset builder
│   ├── 4_3_run_timesnet_classification.py   #          TimesNet classification
│   ├── 4_4_compare_classification.py        #          Confusion matrix generator
│   ├── 4_5_pandemic_clustering.py           #          COVID-19 resilience clustering
│   ├── 4_6_prepare_pandemic_ts.py           #          Pandemic dataset builder
│   ├── 4_7_run_pandemic_classification.py   #          TimesNet pandemic classification
│   ├── 4_8_run_nstransformer_classification.py  #      NS-Transformer classification
│   ├── 5_1_prepare_net_flow.py              # Phase 5: Net flow extraction (chunked)
│   ├── 5_2_advanced_clustering.py           #          Hierarchical clustering
│   ├── 5_3_prepare_netflow_ts.py            #          Net flow dataset builder
│   ├── 5_4_run_timesnet_netflow.py          #          TimesNet tidal classification
│   ├── 5_5_run_nstransformer_netflow.py     #          NS-Transformer tidal classification
│   ├── 5_6_classification_dashboard.py      #          4-panel performance dashboard
│   ├── 5_7_poi_map_visualization.py         #          Interactive POI map generator
│   ├── 5_8_advanced_classification_viz.py   #          ROC / Radar / F1 heatmap
│   ├── 5_10_flow_map_full.py                #          OD flow heatglow map
│   ├── Time_Series_Library/                 #          TSLib framework (TimesNet backbone)
│   └── Nonstationary_Transformers/          #          NS-Transformer framework
│
├── 输出结果/                                # Output visualizations & interactive maps
├── .gitignore
├── LICENSE
├── README.md                                # Documentation (this file)
└── requirements.txt
```

---

## Usage

### Step 1: Data Preprocessing
```bash
python 代码/1_data_preprocessing.py
```

### Step 2: Exploratory Data Analysis
```bash
python 代码/2_data_exploration.py
```

### Step 3: Regression Forecasting
```bash
python 代码/3_1_prepare_transformer_data.py
python 代码/3_2_run_transformer.py          # NS-Transformer
python 代码/3_4_run_timesnet.py             # TimesNet
python 代码/3_3_check_leakage.py            # Data leakage audit
python 代码/3_5_compare_results.py
python 代码/3_6_regression_visualization.py
```

### Step 4: Station Classification
```bash
python 代码/4_1_station_clustering.py
python 代码/4_2_prepare_ts_dataset.py
python 代码/4_3_run_timesnet_classification.py
python 代码/4_5_pandemic_clustering.py
python 代码/4_6_prepare_pandemic_ts.py
python 代码/4_7_run_pandemic_classification.py
python 代码/4_8_run_nstransformer_classification.py
python 代码/4_4_compare_classification.py
```

### Step 5: Tidal Net Flow & Geospatial Validation
```bash
python 代码/5_1_prepare_net_flow.py
python 代码/5_2_advanced_clustering.py
python 代码/5_3_prepare_netflow_ts.py
python 代码/5_4_run_timesnet_netflow.py
python 代码/5_5_run_nstransformer_netflow.py
python 代码/5_6_classification_dashboard.py
python 代码/5_7_poi_map_visualization.py
python 代码/5_8_advanced_classification_viz.py
python 代码/5_10_flow_map_full.py
```

---

## Data

| Dataset | Description | Size | Source |
|:---:|:---|:---:|:---:|
| `london.csv` | Raw cycling transaction records (start/end stations, timestamps, duration) | ~37M rows, 5 GB | [TfL Open Data](https://cycling.data.tfl.gov.uk/) |
| `london_stations.csv` | Station dictionary (ID, name, latitude, longitude) | 800+ stations | TfL |
| `london_weather.csv` | Daily weather (mean temperature, precipitation) | 2017–2020 | London Datastore |
| `UK_holiday.csv` | UK public holiday calendar | 2017–2020 | UK Gov |

---

## Clustering Perspectives

| Clustering Task | Feature Space | Algorithm | Classes |
|:---|:---|:---:|:---|
| **Basic 24h Profile** | 24-dim hourly avg departure | K-Means (K=4) | Fringe / Evening-leisure / Morning-peak / Dual-peak |
| **Pandemic Resilience** | 48-dim (24h pre-COVID + 24h during-COVID) | K-Means (K=4) | Normal-shrink / Essential-haven / Long-tail / Fragile |
| **Tidal Net Flow** | 24-dim hourly avg net flow | Hierarchical Ward (K=3) | Morning-outflow / Self-balanced / Morning-inflow |

---

## Acknowledgement

This project integrates the following open-source frameworks as model backbones:

- **[Time-Series-Library](https://github.com/thuml/Time-Series-Library)** (Tsinghua University, MIT License) — TimesNet, Nonstationary Transformer, Mamba, and other SOTA time-series models.
- **[Nonstationary Transformers](https://github.com/thuml/Nonstationary_Transformers)** (Tsinghua University, MIT License) — De-stationary Attention mechanism for non-stationary time series.

---

## License

This project is released under the [MIT License](LICENSE).

Copyright (c) 2025. All rights reserved. Submitted for Computer Software Copyright Registration.

---
---

<a name="中文文档"></a>

# 中文文档

# 基于深度学习与时空特征的伦敦共享单车潮汐调度预测系统

Language: [English](#london-bike-tidal-forecasting) | **中文**

---

## 项目简介

本项目旨在解决超大城市共享单车系统中的核心痛点："早高峰住宅区无车可借，商业区无位可还"的潮汐拥堵问题。项目从 **5GB** 的伦敦官方骑行流水底表出发，经过严苛的抗 OOM 数据清洗与多源时空特征融合，构建了完备的数据分析流水线。

在算法层面，项目不仅深入探索了数据分布规律（EDA），还引入了目前时序预测领域的顶会大模型（**TimesNet** 与 **Non-stationary Transformer**），完成了从"短时需求预测（回归）"到"动态站点分类（分类）"，再到"终极物理空间验证（POI）"的全业务闭环。

---

## 核心技术亮点

- **亿级数据防爆工程** — 面对 5GB 级别的原始 CSV 流水，通过 Pandas chunksize 技术实现低内存占用下的分块清洗与特征融合（气象、节假日特征提取）。

- **多维高阶聚类挖掘** — 不仅仅停留在基础聚合，项目设计了"疫情前后抗压对比图谱"与"终极潮汐净流量（Net Flow）波形"，结合 K-Means 与层次聚类，精准勾勒出站点的物理隐性属性（如：通勤商业区、外围住宅区、自平衡区）。

- **顶会时序大模型实战验证** — 抛弃传统的 ARIMA 或基础 LSTM，系统性引入并验证了清华大学开源的时序顶会框架（基于 2D 张量变换的 TimesNet 以及针对非平稳数据优化的 Non-stationary Transformer），在大规模盲测中实现了对潮汐类别的精准分类预测。

- **算法结果的真实世界闭环（POI 空间验证）** — 采用 Folium 库生成交互式 HTML 城市地图。将纯粹基于"时间序列算法"聚类出的无监督标签，强行与真实的"GPS 物理空间"进行碰撞。完美证实了算法识别出的"清晨汇聚点"高度重合于伦敦金融城等核心 CBD，为物理世界的车辆调度路线提供了无可辩驳的数学支撑。

---

## 项目架构

```text
数据挖掘课程设计/
│
├── 代码/                                    # 核心算法工程区
│   ├── 1_data_preprocessing.py              # 阶段一：分块数据清洗与特征融合
│   ├── 2_data_exploration.py                # 阶段二：探索性数据分析 (EDA)
│   ├── 3_1 ~ 3_7                            # 阶段三：短时需求预测（回归任务）
│   ├── 4_1 ~ 4_9                            # 阶段四：站点功能分类（分类任务）
│   ├── 5_1 ~ 5_10                           # 阶段五：潮汐净流量 & 地理空间验证
│   ├── Time_Series_Library/                 # 清华大学时序大模型框架 (TimesNet)
│   └── Nonstationary_Transformers/          # NS-Transformer 框架
│
├── 输出结果/                                # 可视化图表与交互式地图
│   ├── figures/                             # 交互式 Folium 地图 (HTML)
│   ├── 分类任务效果图/                       # 混淆矩阵、P/R/F1 看板
│   ├── 分类任务效果图_高阶/                   # ROC曲线、雷达图、F1热力图
│   └── 回归任务效果图/                       # 模型性能对比、MAE衰减曲线
│
├── .gitignore
├── LICENSE
├── README.md                                # 项目文档（本文件）
└── requirements.txt                         # Python 依赖清单
```

---

## 聚类分析维度

| 聚类任务 | 特征空间 | 算法 | 类别 |
|:---|:---|:---:|:---|
| **基础24小时画像** | 24维 每小时平均借车量 | K-Means (K=4) | 边缘冷门站 / 傍晚休闲站 / 早高峰潮汐站 / 双峰通勤站 |
| **疫情抗压演变** | 48维 (24小时疫情前 + 24小时疫情中) | K-Means (K=4) | 普通缩水区 / 刚需避风港 / 永恒长尾区 / 脆弱通勤王 |
| **潮汐净流量** | 24维 每小时平均净流量 | 层次聚类 Ward (K=3) | 清晨流出型 / 自平衡型 / 清晨汇聚型 |

---

## 致谢

本项目集成了以下开源框架作为模型骨架：

- **[Time-Series-Library](https://github.com/thuml/Time-Series-Library)** (清华大学, MIT 许可证) — TimesNet、Non-stationary Transformer、Mamba 等前沿时序模型
- **[Nonstationary Transformers](https://github.com/thuml/Nonstationary_Transformers)** (清华大学, MIT 许可证) — 面向非平稳时间序列的去平稳注意力机制
