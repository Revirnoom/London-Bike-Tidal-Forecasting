# London Shared Bicycle Tidal Dispatch and Demand Forecasting System

**Based on Deep Learning and Spatio-Temporal Feature Mining**

> A full-pipeline intelligent analytics system for urban bike-sharing demand forecasting and station functional classification, leveraging state-of-the-art Transformer-based time-series models and multi-dimensional clustering algorithms on large-scale London cycling trip data.

---

## 1. Software Overview

This software system addresses a critical pain point in metropolitan bike-sharing operations — the **tidal congestion problem**, where residential areas run out of bicycles during morning rush hours while commercial districts overflow with returns. Starting from over **5 GB** of official London cycling transaction records (approximately 37 million trips), the system implements a complete data mining pipeline spanning data preprocessing, exploratory analysis, short-term demand forecasting, station functional classification, and geospatial validation.

The system integrates two cutting-edge deep learning architectures from top-tier machine learning conferences:

- **TimesNet** (ICLR 2023) — a 2D tensor transformation model for time-series analysis
- **Non-stationary Transformer** (NeurIPS 2022) — a Transformer variant optimized for non-stationary temporal distributions

These models are applied to both **regression** (bike demand forecasting) and **classification** (station functional type identification) tasks. The final outputs are validated through interactive geospatial map projections using real GPS coordinates, closing the loop between algorithmic insights and physical-world urban planning.

---

## 2. System Architecture

```
Project Root
│
├── Code/                              # Core algorithm and engineering scripts
│   ├── 1_data_preprocessing.py        # Phase 1: Chunked data cleaning & feature fusion
│   ├── 2_data_exploration.py          # Phase 2: Exploratory data analysis & visualization
│   ├── 3_1_prepare_transformer_data.py    # Phase 3: Time-series data formatting
│   ├── 3_2_run_transformer.py             # NS-Transformer model launcher
│   ├── 3_3_check_leakage.py               # Data leakage audit script
│   ├── 3_4_run_timesnet.py                # TimesNet model launcher
│   ├── 3_5_compare_results.py             # Multi-model performance comparison
│   ├── 3_6_regression_visualization.py    # Regression dashboard generator
│   ├── 3_7_individual_model_plots.py      # Individual model analysis plots
│   ├── 4_1_station_clustering.py          # Phase 4: K-Means station profiling
│   ├── 4_2_prepare_ts_dataset.py          # UEA-format classification dataset builder
│   ├── 4_3_run_timesnet_classification.py # TimesNet classification launcher
│   ├── 4_4_compare_classification.py      # Classification confusion matrix generator
│   ├── 4_5_pandemic_clustering.py         # COVID-19 resilience clustering
│   ├── 4_6_prepare_pandemic_ts.py         # Pandemic classification dataset builder
│   ├── 4_7_run_pandemic_classification.py # TimesNet pandemic classification
│   ├── 4_8_run_nstransformer_classification.py  # NS-Transformer classification
│   ├── 4_9_run_pandemic_nstf.py           # NS-Transformer pandemic classification
│   ├── 5_1_prepare_net_flow.py            # Phase 5: Net flow extraction (chunked)
│   ├── 5_2_advanced_clustering.py         # Hierarchical clustering & dendrogram
│   ├── 5_3_prepare_netflow_ts.py          # Net flow classification dataset builder
│   ├── 5_4_run_timesnet_netflow.py        # TimesNet tidal net flow classification
│   ├── 5_5_run_nstransformer_netflow.py   # NS-Transformer tidal classification
│   ├── 5_6_classification_dashboard.py    # 4-panel classification performance dashboard
│   ├── 5_7_poi_map_visualization.py       # Interactive geospatial POI map generator
│   ├── 5_8_advanced_classification_viz.py # Advanced classification visualizations (ROC, radar, heatmap)
│   ├── 5_9_flow_map_visualization.py      # OD flow corridor visualization
│   ├── 5_10_flow_map_full.py              # Full-volume bike flow heatglow map
│   ├── Time_Series_Library/               # TSLib framework (TimesNet backbone)
│   └── Nonstationary_Transformers/        # NS-Transformer framework
│
├── Processed Data/                    # Intermediate data products
│   ├── 01_rentals_cleaned_with_features.csv   # Cleaned transaction table with weather & holiday features
│   ├── 02_hourly_start_count.csv              # Hourly aggregated departure counts
│   └── 03_hourly_net_flow.csv                 # Hourly tidal net flow (returns − departures)
│
└── Output/                            # Final visualization and report assets
    └── figures/
        ├── 01_station_frequency.png               # Station usage frequency distribution
        ├── 02_hourly_tide_pattern.png             # 24-hour tidal pattern (weekday vs weekend)
        ├── 03_spatial_peak_distribution.png       # Morning/evening/night spatial heatmaps
        ├── 04_weather_impact.png                  # Weather impact correlation analysis
        ├── 05_daily_trend_seasonality.png         # Long-term daily trend with 30-day moving average
        ├── London_Ultimate_Tidal_Map.html         # Interactive tidal classification map
        ├── London_Station_Function_Map.html       # Interactive station function map
        ├── London_Pandemic_Resilience_Map.html    # Interactive pandemic resilience map
        └── London_Bike_Flow_Map_Full.html         # Full OD flow corridor heatglow map
```

---

## 3. Technical Highlights

### 3.1 Memory-Safe Large-Scale Data Processing

The raw London cycling dataset exceeds **5 GB** and contains over **37 million** records. The system employs a **chunk-based streaming architecture** (`pandas chunksize = 1,000,000`) to process the data in 1-million-row blocks, preventing out-of-memory (OOM) crashes. Each chunk undergoes time filtering (2017-01-01 to 2020-07-31), duration anomaly removal (60s–24h), invalid station ID filtering, temporal feature extraction (hour, weekday, weekend flag), and multi-source feature fusion (weather temperature/precipitation, UK public holidays) via left-join operations.

### 3.2 Multi-Dimensional Unsupervised Clustering

The system designs three distinct clustering perspectives for station functional profiling:

| Clustering Task | Feature Space | Algorithm | Classes |
|---|---|---|---|
| **Basic 24h Profile** | 24-dim hourly average departure counts | K-Means (K=4) | Fringe/Evening-leisure/Morning-peak/Dual-peak commuter |
| **Pandemic Resilience** | 48-dim (24h pre-COVID + 24h during-COVID) | K-Means (K=4) | Normal-shrink/Essential-haven/Long-tail/Fragile-commuter |
| **Tidal Net Flow** | 24-dim hourly average net flow (returns − departures) | Hierarchical Ward (K=3) | Morning-outflow/Self-balanced/Morning-inflow |

### 3.3 State-of-the-Art Deep Learning Models

Two top-conference time-series architectures are integrated for both regression and classification:

- **TimesNet (ICLR 2023)**: Converts 1D time series into 2D tensors via FFT-based period detection, then applies 2D convolution kernels (Inception blocks) to capture both intra-period and inter-period temporal patterns. Configuration: `seq_len=96, pred_len=24, d_model=128, d_ff=256, e_layers=2`.

- **Non-stationary Transformer (NeurIPS 2022)**: Extends the standard Transformer with De-stationary Attention and Series Stationarization modules to handle distribution shifts in real-world time series. Configuration: `seq_len=96, pred_len=24, d_model=512, d_ff=2048, e_layers=2, d_layers=1, p_hidden_dims=[16,16]`.

- **LightGBM Baseline**: A gradient boosting decision tree model with engineered lag features (lag-1h, lag-24h, lag-168h) serves as the machine learning baseline for regression comparison.

### 3.4 Data Leakage Audit

A dedicated integrity check script (`3_3_check_leakage.py`) verifies that the chronological train/validation/test split (70:10:20) maintains strict temporal isolation. The script confirms that no future test-set labels leak into the training set, ensuring that all reported metrics are trustworthy.

### 3.5 Comprehensive Evaluation & Visualization

The system generates a rich set of evaluation artifacts:

- **Regression**: MAE, MSE, RMSE, MAPE, MSPE metrics; 24-step MAE decay curves; multi-model performance comparison bar charts; prediction trajectory overlays.
- **Classification**: Confusion matrices (both absolute and row-normalized percentage); per-class Precision/Recall/F1 bar charts; multi-class ROC curves with AUC; radar charts for model capability comparison; cross-task F1 score heatmaps; comprehensive 4-panel performance dashboards.

### 3.6 Geospatial Validation (POI Map Projection)

The system uses **Folium** to render interactive HTML maps on a CartoDB dark-matter basemap. Unsupervised clustering labels are projected onto real London GPS coordinates, enabling visual verification that:
- "Morning-inflow" stations concentrate around the City of London financial district (CBD)
- "Morning-outflow" stations correspond to residential suburbs
- "Self-balanced" stations appear in mixed-use neighborhoods

This closes the analytical loop by validating purely time-series-based algorithmic results against physical urban geography.

---

## 4. Development Environment

| Component | Version / Specification |
|---|---|
| Programming Language | Python 3.11 |
| Operating System | Windows 10/11 |
| Deep Learning Framework | PyTorch ≥ 1.13 |
| Time-Series Library | Time-Series-Library (TSLib) — Tsinghua University open-source framework |
| Data Processing | Pandas, NumPy |
| Machine Learning | scikit-learn, LightGBM |
| Visualization | Matplotlib, Seaborn, Folium |
| Geospatial Mapping | Folium (Leaflet.js backend) |
| Hierarchical Clustering | SciPy (`scipy.cluster.hierarchy`) |

### Key Dependencies

```
torch
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
folium
scipy
tqdm
```

---

## 5. Functional Modules

### Module 1 — Data Preprocessing Engine

**Entry Script**: `1_data_preprocessing.py`

Reads the 5 GB raw CSV transaction log in 1-million-row chunks. Applies temporal boundary filtering, trip duration anomaly removal, station ID validation against the official station dictionary, and performs multi-source left-join feature fusion (weather data: mean temperature & precipitation; UK public holiday calendar). Outputs a cleaned detail table and an hourly aggregated feature table.

### Module 2 — Exploratory Data Analysis (EDA)

**Entry Script**: `2_data_exploration.py`

Generates seven categories of analytical visualizations: station usage frequency distribution (Top-20 bar chart + network-wide histogram), 24-hour tidal wave pattern (weekday vs. weekend comparison), spatial hotspot distribution for morning/evening/night peaks (scatter plot with GPS coordinates), weather impact analysis (temperature regression + precipitation boxplot), and long-term daily trend with 30-day rolling average.

### Module 3 — Short-Term Demand Forecasting (Regression)

**Entry Scripts**: `3_1` through `3_7`

Formats the busiest station's (Station #14) multivariate time series into sliding-window tensors for supervised learning. Launches and trains three deep learning models (TimesNet, Non-stationary Transformer, Mamba) and one machine learning baseline (LightGBM). Includes a data leakage audit to verify temporal isolation. Generates per-model performance dashboards and cross-model comparison charts with MAE/MSE/RMSE metrics.

### Module 4 — Station Functional Classification

**Entry Scripts**: `4_1` through `4_9`

Extracts 24-hour average departure profiles for all stations and applies K-Means clustering (K=4) to identify station functional types. Extends the analysis with pandemic-era resilience profiling by comparing 2019 (pre-COVID) and 2020 (during-COVID) 48-dimensional feature vectors. Packages clustering labels into UEA-standard `.ts` format datasets and launches TimesNet and NS-Transformer classification models. Generates confusion matrices, precision/recall/F1 reports, and 4-panel performance dashboards.

### Module 5 — Tidal Net Flow Analysis & Geospatial Validation

**Entry Scripts**: `5_1` through `5_10`

Computes hourly tidal net flow (bike returns minus departures) for every station using chunk-based dictionary accumulation on the raw 5 GB dataset. Applies Ward-linkage hierarchical clustering with dendrogram visualization to identify three tidal archetypes. Packages labels and runs deep learning classification models. Generates interactive Folium HTML maps that project clustering results onto the real London street map for POI-level spatial validation. Additionally produces a full OD flow corridor heatglow map visualizing the top cycling routes across the entire network.

---

## 6. Execution Guide

### Step 1: Data Preprocessing
```bash
python Code/1_data_preprocessing.py
```
Processes the raw 5 GB CSV file and outputs cleaned data to the `Processed Data/` directory.

### Step 2: Exploratory Data Analysis
```bash
python Code/2_data_exploration.py
```
Generates EDA visualization charts in the `Output/` directory.

### Step 3: Regression Model Training & Evaluation
```bash
python Code/3_1_prepare_transformer_data.py
python Code/3_2_run_transformer.py
python Code/3_4_run_timesnet.py
python Code/3_3_check_leakage.py
python Code/3_5_compare_results.py
python Code/3_6_regression_visualization.py
```

### Step 4: Classification Model Training & Evaluation
```bash
python Code/4_1_station_clustering.py
python Code/4_2_prepare_ts_dataset.py
python Code/4_3_run_timesnet_classification.py
python Code/4_5_pandemic_clustering.py
python Code/4_6_prepare_pandemic_ts.py
python Code/4_7_run_pandemic_classification.py
python Code/4_8_run_nstransformer_classification.py
python Code/4_4_compare_classification.py
```

### Step 5: Tidal Net Flow & Geospatial Validation
```bash
python Code/5_1_prepare_net_flow.py
python Code/5_2_advanced_clustering.py
python Code/5_3_prepare_netflow_ts.py
python Code/5_4_run_timesnet_netflow.py
python Code/5_5_run_nstransformer_netflow.py
python Code/5_6_classification_dashboard.py
python Code/5_7_poi_map_visualization.py
python Code/5_8_advanced_classification_viz.py
python Code/5_10_flow_map_full.py
```

---

## 7. Input Data Description

| Dataset | Description | Source |
|---|---|---|
| `london.csv` | Raw cycling transaction records (~37M rows, 5 GB) containing start/end station IDs, timestamps, and trip duration | Transport for London (TfL) Open Data |
| `london_stations.csv` | Station dictionary with station ID, name, latitude, and longitude | Transport for London (TfL) |
| `london_weather.csv` | Daily weather observations including mean temperature and precipitation | London Datastore |
| `UK_holiday.csv` | UK public holiday calendar | UK Government Open Data |

---

## 8. Output Artifacts

### Processed Intermediate Data
- `01_rentals_cleaned_with_features.csv` — Cleaned transaction table with fused weather and holiday features
- `02_hourly_start_count.csv` — Station-level hourly departure count aggregation
- `03_hourly_net_flow.csv` — Station-level hourly tidal net flow (returns − departures)

### Classification Labels
- `station_labels.csv` — K-Means 4-class basic station functional labels
- `station_labels_pandemic.csv` — K-Means 4-class pandemic resilience labels
- `station_labels_netflow.csv` — Hierarchical 3-class tidal net flow labels

### Visualization Outputs
- Static PNG charts: station frequency, tidal patterns, spatial distributions, weather impacts, trend analysis, confusion matrices, ROC curves, radar plots, F1 heatmaps, regression performance dashboards
- Interactive HTML maps: tidal classification map, station function map, pandemic resilience map, full OD flow corridor heatglow map

### Trained Model Checkpoints
- TimesNet and NS-Transformer model weights (`.pth` files) for both regression and classification tasks, stored under `checkpoints/` directories within the respective framework folders

---

## 9. Software Originality Statement

This software system is independently designed and developed. The core pipeline scripts (data preprocessing, feature engineering, clustering analysis, model launcher configuration, evaluation visualization, and geospatial map generation) are original work. The system integrates two open-source deep learning frameworks — **Time-Series-Library** (Tsinghua University, MIT License) and **Nonstationary Transformers** (MIT License) — as model backbone dependencies, with custom data loaders and configurations adapted for the London bike-sharing domain.

---

## 10. License

Copyright (c) 2025. All rights reserved.

This software is submitted for Computer Software Copyright Registration.
