# 🚲 基于深度学习与时空特征的伦敦共享单车潮汐调度预测系统
**(London Shared Bicycle Tidal Dispatch & Demand Forecasting System)**

## 🌟 项目简介 (Project Overview)
本项目旨在解决超大城市共享单车系统中的核心痛点：“早高峰住宅区无车可借，商业区无位可还”的潮汐拥堵问题。
项目从 5GB 的伦敦官方骑行流水底表出发，经过严苛的抗 OOM 数据清洗与多源时空特征融合，构建了完备的数据分析流水线。在算法层面，项目不仅深入探索了数据分布规律（EDA），还引入了目前时序预测领域的顶会大模型（**TimesNet** 与 **Non-stationary Transformer**），完成了从“短时需求预测（回归）”到“动态门派分类（分类）”，再到“终极物理空间验证（POI）”的全业务闭环。

---

## 📂 项目核心架构与文件指南 (Directory Tree)

本项目严格遵循工业级数据科学工程规范，代码、数据、产物完全分离，按照 1 到 5 的流水线阶段顺序编排。

```text
📁 数据挖掘课程设计
│
├── 📁 代码 (核心算法工程区)
│   │
│   ├── 📁 figures/                      # [大模型分类盲测战报区：混淆矩阵与评估结果]
│   │   ├── cm_classification_LondonBikeClass_TimesNet.png       # 基础画像分类：TimesNet 战绩
│   │   ├── cm_classification_LondonBikeClass_Nonstationary...   # 基础画像分类：NS-Transformer 战绩
│   │   ├── cm_classification_LondonBikePandemic_TimesNet.png    # 疫情抗压分类：TimesNet 战绩
│   │   ├── cm_classification_LondonBikeNetFlow_TimesNet.png     # 终极潮汐净流量：TimesNet 战绩
│   │   ├── cm_classification_LondonBikeNetFlow_Nonstationary... # 终极潮汐净流量：NS-Transformer 战绩
│   │   └── confusion_matrix_TimesNet.png                        # 汇总混淆矩阵
│   │
│   ├── 📁 分类数据准备/                 # [考卷A：常规24小时画像标签]
│   │   ├── cluster_profiles.png         # 各门派单车 24小时潮汐波动折线图
│   │   └── station_labels.csv           # 基础聚类标准答案名单
│   │
│   ├── 📁 分类数据准备_疫情演变/         # [考卷B：疫情前后抗压标签]
│   │   ├── pandemic_evolution_profiles.png # “刚需避风港”与“脆弱崩溃型”波动画像图
│   │   └── station_labels_pandemic.csv     # 疫情抗压聚类标准答案名单
│   │
│   ├── 📁 分类数据准备_潮汐净流量/       # [考卷C：终极业务调度指标标签]
│   │   ├── 01_dendrogram.png            # 极其硬核的层次聚类树状图 (揭示站点亲缘关系)
│   │   ├── 02_netflow_profiles.png      # 三大门派净流量(借入-借出)波形对比图
│   │   └── station_labels_netflow.csv   # 终极潮汐三大门派标准答案名单
│   │
│   ├── 📁 Time_Series_Library/          # 清华大学开源的时序大模型底层框架 (内含 TimesNet)
│   ├── 📁 Nonstationary_Transformers/   # NS-Transformer 专属外部依赖组件库
│   │
│   ├── 📌 阶段一：数据清洗与特征工程
│   ├── 1_data_preprocessing.py          # [核心底盘] Chunking 分块处理 5GB 原始流水，防内存溢出，并左连接融合天气/节假日特征
│   │
│   ├── 📌 阶段二：数据探索与可视化 (EDA)
│   ├── 2_data_exploration.py            # [EDA分析] 对清洗后的数据进行探索性统计，生成时空规律分布图
│   │
│   ├── 📌 阶段三：短时需求预测 (回归任务)
│   ├── 3_1_prepare_transformer_data.py  # 数据预处理：将时间序列格式化为大模型监督学习所需要的滑窗张量格式
│   ├── 3_2_run_transformer.py           # 基础 Transformer 模型点火脚本
│   ├── 3_3_check_leakage.py             # [安全检查] 算法工程级脚本，严查数据泄露 (Data Leakage)
│   ├── 3_4_run_timesnet.py              # TimesNet 顶会大模型点火脚本 (预测核心)
│   ├── 3_5_compare_results.py           # 多模型预测成绩横向对比与评估
│   │
│   ├── 📌 阶段四：城市站点二维功能盲测 (分类任务)
│   ├── 4_1_station_clustering.py        # 提取全天24小时借车画像，K-Means划分常规门派
│   ├── 4_2_prepare_ts_dataset.py        # 将聚类标签打包成 TSlib 严格标准的 .ts 考卷格式
│   ├── 4_3_run_timesnet_classification.py # TimesNet 常规站点功能分类大考启动器
│   ├── 4_4_compare_classification.py    # 智能扫描分类成绩，生成混淆矩阵热力图
│   ├── 4_5_pandemic_clustering.py       # [高阶聚类] 提取疫情前后对比画像，挖掘 "刚需避风港" 与 "脆弱崩溃型"
│   ├── 4_6_prepare_pandemic_ts.py       # 疫情抗压标签打包
│   ├── 4_7/4_8_run_...                  # TimesNet 与 NS-Transformer 疫情动态抗压盲测双雄对决
│   │
│   ├── 📌 阶段五：终极调度演进 (潮汐净流量体系 & 空间验证)
│   ├── 5_1_prepare_net_flow.py          # [算力挑战] 从底表中提取终极业务指标 "潮汐净流量" (借入-借出)
│   ├── 5_2_advanced_clustering.py       # 层次聚类 (Hierarchical)，绘制树状图并精确提取 3大潮汐门派
│   ├── 5_3_prepare_netflow_ts.py        # 终极净流量标签打包
│   ├── 5_4/5_5_run_...                  # TimesNet 与 NS-Transformer 终极潮汐大考
│   └── 5_7_poi_map_visualization.py     # [业务闭环王炸] 结合经纬度，将聚类的3大门派投射到真实伦敦地图，进行 POI 空间实锤验证！
│
├── 📁 处理后的数据集 (中间产物区)
│   ├── 01_rentals_cleaned_with_features.csv # 清洗并融合外部特征后的全量底表
│   ├── 02_hourly_start_count.csv        # 小时级借车量聚合表
│   └── 03_hourly_net_flow.csv           # 小时级潮汐净流量聚合表
│
├── 📁 输出结果 (业务可视化产出区)
│   └── 📁 figures/                      # [精美业务图表与模型产物库]
│       ├── 01_station_frequency.png     # 📊 站点高频使用热力分布
│       ├── 02_hourly_tide_pattern.png   # 📊 24小时整体潮汐波形特征图
│       ├── 03_spatial_peak_distribution.png # 📊 早晚高峰空间分布对比图
│       ├── 04_weather_impact.png        # 📊 气温与降水对骑行量的影响相关性分析
│       ├── 05_daily_trend_seasonality.png # 📊 每日骑行趋势与季节性周期规律
│       ├── 06_weekday_distribution.png  # 📊 工作日 vs 周末骑行特征差异对比
│       ├── 07_duration_distribution.png # 📊 骑行时长分布与异常过滤展示
│       ├── 08_prediction_time_series.png# 📈 阶段三：短时需求预测折线拟合度对比图
│       ├── 09_feature_importance.png    # 📈 阶段三：多维时空特征重要性排行
│       ├── 10_cluster_radar.png         # 🎯 阶段五：3大潮汐门派画像雷达图
│       ├── 11_cluster_spatial_map.png   # 🎯 阶段五：潮汐聚类结果静态空间映射图
│       └── London_Ultimate_Tidal_Map.html # 🚀 [王炸产物] 终极交互式上帝视角潮汐全景分布地图 (可在浏览器中自由缩放查看)
│
└── 📁 实验记录 (研究笔记)                 # 存放各阶段研究思路拆解的 Markdown 笔记

🚀 核心技术亮点 (Technical Highlights)
亿级数据防爆工程：面对 5GB 级别的原始 CSV 流水，通过 Pandas chunksize 技术实现低内存占用下的分块清洗与特征融合（气象、节假日特征提取）。

多维高阶聚类挖掘：不仅仅停留在基础聚合，项目设计了“疫情前后抗压对比图谱”与“终极潮汐净流量（Net Flow）波形”，结合 K-Means 与层次聚类，精准勾勒出站点的物理隐性属性（如：通勤商业区、外围住宅区、自平衡区）。

顶会时序大模型实战验证：抛弃传统的 ARIMA 或基础 LSTM，系统性引入并验证了清华大学开源的时序顶会框架（基于 2D 张量变换的 TimesNet 以及针对非平稳数据优化的 Non-stationary Transformer），在大规模盲测中实现了对潮汐类别的精准分类预测。

算法结果的真实世界闭环 (POI 空间验证)：采用 folium 库生成交互式 HTML 城市地图。将纯粹基于“时间序列算法”聚类出的无监督标签，强行与真实的“GPS 物理空间”进行碰撞。完美证实了算法识别出的“清晨汇聚点”高度重合于伦敦金融城等核心 CBD，为物理世界的车辆调度货车路线提供了无可辩驳的数学支撑。