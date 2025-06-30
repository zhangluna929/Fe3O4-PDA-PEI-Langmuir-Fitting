# adsorption_fit_demo.py
# 模拟 Fe₃O₄@PDA@PEI 对污染物的 Langmuir 等温线吸附数据并完成拟合。

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def langmuir(C, Qmax, K):
    """
    Langmuir isotherm: Q = (Qmax*K*C)/(1+K*C)
    """
    return (Qmax * K * C) / (1 + K * C)

C = np.linspace(0.1, 10, 50)
TRUE_QMAX, TRUE_K = 50, 0.3
np.random.seed(42)
Q_true = langmuir(C, TRUE_QMAX, TRUE_K)
noise = np.random.normal(0, 2, size=C.size)
Q_obs = Q_true + noise

popt, _ = curve_fit(langmuir, C, Q_obs, p0=[40, 0.2])
FITTED_QMAX, FITTED_K = popt
Q_fit = langmuir(C, *popt)

assert abs(FITTED_QMAX - TRUE_QMAX) < 10, "Qmax 拟合偏差过大"
assert abs(FITTED_K - TRUE_K) < 0.2, "K 拟合偏差过大"

os.makedirs("results", exist_ok=True)
with open("results/adsorption_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Concentration (mg/L)", "Q_obs (mg/g)", "Q_fit (mg/g)"])
    writer.writerows(zip(C, Q_obs, Q_fit))

plt.figure(figsize=(6, 4))
plt.scatter(C, Q_obs, label="Observed (Simulated)", color="tab:blue", s=25)
plt.plot(C, Q_fit, label=f"Langmuir Fit\nQmax={FITTED_QMAX:.2f}, K={FITTED_K:.2f}", color="tab:red")
plt.xlabel("Concentration (mg/L)")
plt.ylabel("Adsorption Capacity (mg/g)")
plt.title("Fe₃O₄@PDA@PEI Langmuir Adsorption Isotherm (Simulation)")
plt.legend()
plt.tight_layout()
plt.savefig("results/adsorption_fit.png", dpi=300)
plt.close()

print("===== Langmuir 拟合结果 =====")
print(f"真实值      : Qmax = {TRUE_QMAX:.2f} mg/g, K = {TRUE_K:.2f} L/mg")
print(f"拟合结果    : Qmax = {FITTED_QMAX:.2f} mg/g, K = {FITTED_K:.2f} L/mg")
R2 = 1 - np.sum((Q_obs - Q_fit)**2) / np.sum((Q_obs - Q_obs.mean())**2)
print(f"R² (拟合优度): {R2:.4f}")
print("数据与图像已保存至 results/ 目录。")
