# main_analysis.py

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# 从我们的工具包中导入所有需要的模块
from ssb_analyzer import adsorption_models
from ssb_analyzer import electrochemical_models
from ssb_analyzer.fitting_algorithms import fit_isotherm, calculate_r2, fit_eis_data
from ssb_analyzer.plotting import plot_adsorption_isotherm

def run_adsorption_analysis():
    """
    运行吸附等温线分析的全流程：
    1. 模拟或加载数据
    2. 使用多个模型进行拟合
    3. 比较模型并输出结果
    4. 可视化最佳模型
    """
    # 1. 模拟数据
    # 为了更好地测试不同模型，我们使用Sips模型作为真实模型来生成数据
    C = np.linspace(0.1, 20, 50)
    TRUE_QMAX, TRUE_KS, TRUE_BETA = 60, 0.15, 0.85
    np.random.seed(42)
    Q_true = adsorption_models.sips(C, TRUE_QMAX, TRUE_KS, TRUE_BETA)
    noise = np.random.normal(0, 1.5, size=C.size)
    Q_obs = Q_true + noise

    # 2. 定义要测试的模型、初始猜测值 (p0) 和参数边界 (bounds)
    models_to_test = {
        "Langmuir": {
            "func": adsorption_models.langmuir,
            "p0": [50, 0.1],
            "bounds": ([0, 0], [100, 1])
        },
        "Freundlich": {
            "func": adsorption_models.freundlich,
            "p0": [5, 1.5],
            "bounds": ([0, 0.1], [50, 10])
        },
        "Sips": {
            "func": adsorption_models.sips,
            "p0": [50, 0.1, 0.8],
            "bounds": ([0, 0, 0.1], [100, 1, 1])
        }
    }

    results = {}
    print("===== 开始对多个吸附模型进行拟合 =====")

    # 3. 循环拟合每个模型
    for name, model in models_to_test.items():
        try:
            popt, _ = fit_isotherm(model["func"], C, Q_obs, p0=model["p0"], bounds=model["bounds"])
            Q_fit = model["func"](C, *popt)
            r2 = calculate_r2(Q_obs, Q_fit)
            results[name] = {"popt": popt, "r2": r2, "Q_fit": Q_fit}
            print(f"- 模型: {name}, R² = {r2:.4f}")
        except RuntimeError:
            print(f"- 模型: {name}, 拟合失败")

    # 4. 找出最佳模型 (基于 R²)
    best_model_name = max(results, key=lambda name: results[name]["r2"])
    best_result = results[best_model_name]
    
    print("\n===== 拟合结果总结 =====")
    print(f"最佳拟合模型: {best_model_name} (R² = {best_result['r2']:.4f})")
    param_names = models_to_test[best_model_name]["func"].__code__.co_varnames[1:]
    param_values_dict = dict(zip(param_names, best_result['popt']))
    param_values_str = ", ".join([f"{p_name}={val:.3f}" for p_name, val in param_values_dict.items()])
    print(f"拟合参数: {param_values_str}")

    # 5. 结果输出与保存
    os.makedirs("results", exist_ok=True)
    
    # 将最佳拟合结果保存到 CSV
    csv_path = "results/best_fit_adsorption_data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Concentration (mg/L)", "Q_obs (mg/g)", f"Q_fit_{best_model_name} (mg/g)"])
        writer.writerows(zip(C, Q_obs, best_result['Q_fit']))

    # 使用我们标准化的绘图模块生成并保存图像
    fig = plot_adsorption_isotherm(
        C, 
        Q_obs, 
        best_result['Q_fit'], 
        best_model_name, 
        param_values_dict
    )
    fig_path = "results/best_fit_adsorption_fit.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    print(f"\n数据已保存至: {csv_path}")
    print(f"最佳拟合图像已保存至: {fig_path}")

def run_eis_analysis():
    """
    运行电化学阻抗谱 (EIS) 分析流程 (使用自建的稳定拟合算法)
    """
    print("\n===== 开始运行 EIS 分析流程 =====")
    
    # 1. 定义真实参数并生成模拟数据
    TRUE_R_S, TRUE_R_CT, TRUE_C_DL = 20, 150, 1e-5
    f_range = np.logspace(5, -2, num=71)
    
    Z_true = electrochemical_models.randles_model(f_range, TRUE_R_S, TRUE_R_CT, TRUE_C_DL)
    
    np.random.seed(42)
    noise_real = np.random.normal(0, 0.5, Z_true.size)
    noise_imag = np.random.normal(0, 0.5, Z_true.size)
    Z_obs = Z_true + noise_real + 1j*noise_imag

    # 2. 使用我们自建的算法进行拟合
    initial_guess = [10, 100, 1e-6]
    bounds = ([0, 0, 0], [100, 500, 1e-3])
    
    popt, _ = fit_eis_data(electrochemical_models.randles_model, f_range, Z_obs, p0=initial_guess, bounds=bounds)
    Z_fit = electrochemical_models.randles_model(f_range, *popt)
    
    # 3. 打印拟合结果
    print("EIS 拟合结果:")
    param_names = ['R_s', 'R_ct', 'C_dl']
    fitted_params = dict(zip(param_names, popt))
    print(fitted_params)

    # 4. 绘图与保存
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(np.real(Z_obs), -np.imag(Z_obs), 'o', markersize=5, label='Observed Data')
    ax.plot(np.real(Z_fit), -np.imag(Z_fit), '-', linewidth=2, label='Fit')
    ax.set_xlabel("Z' (Ω)", fontsize=12)
    ax.set_ylabel("-Z'' (Ω)", fontsize=12)
    ax.set_title("Nyquist Plot - Custom Fit", fontsize=14)
    ax.legend()
    ax.set_aspect('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    fig_path = "results/eis_nyquist_fit.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"EIS 拟合图像已保存至: {fig_path}")

if __name__ == "__main__":
    run_adsorption_analysis()
    run_eis_analysis()
