# 🚲 基于深度学习与时空特征的伦敦共享单车潮汐调度预测系统
**(London Shared Bicycle Tidal Dispatch & Demand Forecasting System)**

## 🌟 项目简介 (Project Overview)
本项目旨在解决超大城市共享单车系统中的核心痛点："早高峰住宅区无车可借，商业区无位可还"的潮汐拥堵问题。
项目从 5GB 的伦敦官方骑行流水底表出发，经过严苛的抗 OOM 数据清洗与多源时空特征融合，构建了完备的数据分析流水线。在算法层面，项目不仅深入探索了数据分布规律（EDA），还引入了目前时序预测领域的顶会大模型（**TimesNet** 与 **Non-stationary Transformer**），完成了从"短时需求预测（回归）"到"动态门派分类（分类）"，再到"终极物理空间验证（POI）"的全业务闭环。

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
│   ├── 📁 分类数据准备_疫情演变/         # [考卷B：疫情前后抗压标签]
│   ├── 📁 分类数据准备_潮汐净流量/       # [考卷C：终极业务调度指标标签]
│   │
│   ├── 📁 Time_Series_Library/          # 清华大学开源的时序大模型底层框架 (内含 TimesNet)
│   ├── 📁 Nonstationary_Transformers/   # NS-Transformer 专属外部依赖组件库
│   │
│   ├── 📌 阶段一：数据清洗与特征工程
│   ├── 1_data_preprocessing.py          # [核心底盘] Chunking 分块处理 5GB 原始流水，防内存溢出
│   │
│   ├── 📌 阶段二：数据探索与可视化 (EDA)
│   ├── 2_data_exploration.py            # [EDA分析] 对清洗后的数据进行探索性统计
│   │
│   ├── 📌 阶段三：短时需求预测 (回归任务)
│   ├── 3_1 ~ 3_7                        # TimesNet / NS-Transformer / LightGBM 回归预测
│   │
│   ├── 📌 阶段四：城市站点功能分类 (分类任务)
│   ├── 4_1 ~ 4_9                        # K-Means聚类 + 大模型分类盲测
│   │
│   ├── 📌 阶段五：终极调度演进 (潮汐净流量 & 空间验证)
│   └── 5_1 ~ 5_10                       # 层次聚类 + POI地图验证
│
├── 📁 处理后的数据集 (中间产物区)
├── 📁 输出结果 (业务可视化产出区)
└── 📁 实验记录 (研究笔记)
```

---

🚀 核心技术亮点 (Technical Highlights)
亿级数据防爆工程：面对 5GB 级别的原始 CSV 流水，通过 Pandas chunksize 技术实现低内存占用下的分块清洗与特征融合（气象、节假日特征提取）。

多维高阶聚类挖掘：不仅仅停留在基础聚合，项目设计了"疫情前后抗压对比图谱"与"终极潮汐净流量（Net Flow）波形"，结合 K-Means 与层次聚类，精准勾勒出站点的物理隐性属性（如：通勤商业区、外围住宅区、自平衡区）。

顶会时序大模型实战验证：抛弃传统的 ARIMA 或基础 LSTM，系统性引入并验证了清华大学开源的时序顶会框架（基于 2D 张量变换的 TimesNet 以及针对非平稳数据优化的 Non-stationary Transformer），在大规模盲测中实现了对潮汐类别的精准分类预测。

算法结果的真实世界闭环 (POI 空间验证)：采用 folium 库生成交互式 HTML 城市地图。将纯粹基于"时间序列算法"聚类出的无监督标签，强行与真实的"GPS 物理空间"进行碰撞。完美证实了算法识别出的"清晨汇聚点"高度重合于伦敦金融城等核心 CBD，为物理世界的车辆调度货车路线提供了无可辩驳的数学支撑。
