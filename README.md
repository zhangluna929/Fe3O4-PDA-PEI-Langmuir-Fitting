<!-- README.md -->
# Advanced Analysis Toolkit for Adsorption and Electrochemical Systems
**Author: lunazhang**

本系统是一个为材料科学与电化学研究领域设计的高级数据分析工具包，专注于非线性模型的拟合与验证。它集成了一套复杂的算法，旨在精确解析吸附等温线和电化学阻抗谱（EIS）数据，为研究人员提供一个从原始数据到参数化模型的自动化、可复现的分析流程。此工具包的核心在于其对多种经典物理化学模型的支持，以及一个能够根据统计学指标自动甄别最优模型的健壮框架。

This system is an advanced data analysis toolkit designed for materials science and electrochemistry research, with a specific focus on the fitting and validation of non-linear models. It integrates a sophisticated suite of algorithms to precisely deconvolve data from adsorption isotherms and Electrochemical Impedance Spectroscopy (EIS), providing researchers with an automated and reproducible pipeline from raw data to a parameterized model. The core of this toolkit lies in its support for multiple classic physicochemical models and a robust framework capable of automatically identifying the optimal model based on statistical metrics.

## 核心功能 (Core Features)

该项目提供了一系列强大的分析功能，旨在将复杂的数据处理流程标准化、自动化。

The project offers a suite of powerful analytical features designed to standardize and automate complex data processing workflows.

#### 多模型吸附等温线分析 (Multi-Model Adsorption Isotherm Analysis)
本工具包能够对吸附过程进行深度分析，支持包括Langmuir、Freundlich和Sips在内的多种关键等温线模型。系统采用非线性最小二乘法对实验数据进行拟合，并通过计算确定系数（R²）来量化每个模型的拟合优度。最终，它会自动筛选出与实验数据最为契合的模型，并生成高质量的可视化图表与格式化的数据报告，为深入理解吸附剂-吸附质相互作用机制提供定量依据。

This toolkit enables in-depth analysis of adsorption processes, supporting a range of critical isotherm models including Langmuir, Freundlich, and Sips. The system employs a non-linear least squares method to fit experimental data and quantifies the goodness-of-fit for each model by calculating the coefficient of determination (R²). Ultimately, it automatically selects the model that best conforms to the experimental data, generating high-quality visualizations and formatted data reports to provide a quantitative basis for understanding adsorbent-adsorbate interaction mechanisms.

#### 电化学阻抗谱解析 (Electrochemical Impedance Spectroscopy Deconvolution)
在电化学分析方面，该工具包实现了对EIS数据的精确解析，特别是针对经典的Randles等效电路模型。通过对奈奎斯特图（Nyquist Plot）进行拟合，系统能够从复杂的阻抗数据中提取出关键的电化学参数，如溶液电阻（R_s）、电荷转移电阻（R_ct）和双电层电容（C_dl）。这些参数对于评估电极材料性能、研究电极/电解质界面动力学以及揭示电化学反应机理至关重要。

In the realm of electrochemical analysis, the toolkit facilitates the precise deconvolution of EIS data, particularly for the classic Randles equivalent circuit model. By fitting the Nyquist plot, the system can extract critical electrochemical parameters from complex impedance data, such as solution resistance (R_s), charge-transfer resistance (R_ct), and double-layer capacitance (C_dl). These parameters are vital for evaluating the performance of electrode materials, studying the kinetics of the electrode/electrolyte interface, and elucidating electrochemical reaction mechanisms.

## 项目架构 (Project Architecture)

项目的架构经过精心设计，遵循模块化和关注点分离的原则，以确保代码的可扩展性、可维护性和易用性。核心逻辑被封装在 `ssb_analyzer` 包中，而顶层脚本则负责协调整个分析流程。

The project architecture is meticulously designed following principles of modularity and separation of concerns to ensure scalability, maintainability, and ease of use. The core logic is encapsulated within the `ssb_analyzer` package, while a top-level script orchestrates the entire analysis pipeline.

```
.
├── ssb_analyzer/
│   ├── __init__.py                 # 包初始化文件 (Package initializer)
│   ├── adsorption_models.py        # 定义吸附等温线模型 (Defines adsorption isotherm models)
│   ├── analysis.py                 # 核心分析功能模块 (Core analysis functions)
│   ├── electrochemical_models.py   # 定义电化学等效电路模型 (Defines electrochemical equivalent circuit models)
│   ├── fitting_algorithms.py       # 实现核心非线性拟合算法 (Implements core non-linear fitting algorithms)
│   └── plotting.py                 # 标准化绘图功能模块 (Module for standardized plotting functions)
├── main_analysis.py                  # 分析流程的主执行脚本 (Main execution script for the analysis pipeline)
├── requirements.txt                  # 项目依赖库 (Project dependencies)
├── adsorption_data.csv               # 示例吸附数据 (Sample adsorption data)
├── adsorption_fit.png                # 示例拟合结果图 (Sample fit result plot)
└── README.md                         # 本文档 (This document)
```

## 使用说明 (Usage Instructions)

请遵循以下步骤来运行完整的分析流程。确保您的环境中已安装Python。

Follow the steps below to run the complete analysis pipeline. Ensure you have Python installed in your environment.

1.  **安装依赖 (Install Dependencies)**

    执行以下命令安装所有必需的Python库：
    Execute the following command to install all required Python libraries:
    ```sh
    pip install -r requirements.txt
    ```

2.  **运行分析 (Execute Analysis)**

    运行主分析脚本。该脚本将自动执行吸附和EIS数据的所有处理、拟合和可视化步骤。
    Run the main analysis script. This script will automatically perform all processing, fitting, and visualization steps for both adsorption and EIS data.
    ```sh
    python main_analysis.py
    ```

3.  **查看结果 (Review Results)**

    所有生成的图表（.png格式）和数据文件（.csv格式）将被保存在 `results/` 目录下，供您进一步分析和报告。
    All generated plots (.png format) and data files (.csv format) will be saved in the `results/` directory for your further analysis and reporting.