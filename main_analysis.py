# main_analysis.py

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from ssb_analyzer import adsorption_models
from ssb_analyzer.fitting_algorithms import fit_isotherm, calculate_r2
from ssb_analyzer.plotting import plot_adsorption_isotherm

def run_adsorption_analysis():
    """
    运行吸附等温线分析：模拟数据、多模型拟合、结果比较和可视化
    """
    # 模拟数据 - 细菌吸附 (CFU/mg)
    C = np.linspace(0.1, 20, 50)
    TRUE_QMAX, TRUE_KS, TRUE_BETA = 1.5e8, 0.15, 0.85  # 1.5×10⁸ CFU/mg
    np.random.seed(42)
    Q_true = adsorption_models.sips(C, TRUE_QMAX, TRUE_KS, TRUE_BETA)
    noise = np.random.normal(0, 1.5e6, size=C.size)  # σ=1.5×10⁶
    Q_obs = Q_true + noise
    Q_obs = np.maximum(Q_obs, 0.1e6)

    # 定义要测试的模型和参数边界
    models_to_test = {
        "Langmuir": {
            "func": adsorption_models.langmuir,
            "p0": [1.0e8, 0.1],
            "bounds": ([1e6, 0.01], [5e8, 2.0])
        },
        "Freundlich": {
            "func": adsorption_models.freundlich,
            "p0": [1e7, 1.5],
            "bounds": ([1e6, 0.5], [1e9, 5.0])
        },
        "Sips": {
            "func": adsorption_models.sips,
            "p0": [1.2e8, 0.12, 0.8],
            "bounds": ([1e6, 0.01, 0.1], [5e8, 2.0, 1.0])
        }
    }

    results = {}
    print("===== 开始对多个吸附模型进行拟合 =====")

    # 循环拟合每个模型
    for name, model in models_to_test.items():
        try:
            popt, _ = fit_isotherm(model["func"], C, Q_obs, p0=model["p0"], bounds=model["bounds"])
            Q_fit = model["func"](C, *popt)
            r2 = calculate_r2(Q_obs, Q_fit)
            results[name] = {"popt": popt, "r2": r2, "Q_fit": Q_fit}
            print(f"- 模型: {name}, R² = {r2:.4f}")
        except RuntimeError:
            print(f"- 模型: {name}, 拟合失败")

    # 找出最佳模型 (基于 R²)
    best_model_name = max(results, key=lambda name: results[name]["r2"])
    best_result = results[best_model_name]
    
    print("\n===== 拟合结果总结 =====")
    print(f"最佳拟合模型: {best_model_name} (R² = {best_result['r2']:.4f})")
    param_names = models_to_test[best_model_name]["func"].__code__.co_varnames[1:]
    param_values_dict = dict(zip(param_names, best_result['popt']))
    
    param_strs = []
    for p_name, val in param_values_dict.items():
        if p_name.lower() == 'qmax':
            param_strs.append(f"{p_name}={val:.2e} CFU/mg")
        else:
            param_strs.append(f"{p_name}={val:.3f}")
    param_values_str = ", ".join(param_strs)
    print(f"拟合参数: {param_values_str}")

    # 结果输出与保存
    os.makedirs("results", exist_ok=True)
    
    csv_path = "results/best_fit_adsorption_data.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Concentration (mg/L)", "Q_obs (CFU/mg)", f"Q_fit_{best_model_name} (CFU/mg)"])
        writer.writerows(zip(C, Q_obs, best_result['Q_fit']))

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



if __name__ == "__main__":
    run_adsorption_analysis()
