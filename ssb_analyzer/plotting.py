# ssb_analyzer/plotting.py

import matplotlib.pyplot as plt

def plot_adsorption_isotherm(C, Q_obs, Q_fit, model_name, popt_dict):
    """
    Plots the adsorption isotherm data and the fitted model dynamically.

    Parameters:
    - C (array-like): Concentration data.
    - Q_obs (array-like): Observed adsorption capacity data.
    - Q_fit (array-like): Fitted adsorption capacity data.
    - model_name (str): Name of the model for the plot title and legend.
    - popt_dict (dict): Dictionary of optimized parameter names and values.
    """
    param_strs = []
    for key, val in popt_dict.items():
        if key.lower() == 'qmax':
            param_strs.append(f"{key}={val:.2e} CFU/mg")
        elif isinstance(val, (int, float)) and val > 1e6:
            param_strs.append(f"{key}={val:.2e}")
        else:
            param_strs.append(f"{key}={val:.2f}")
    param_str = ", ".join(param_strs)
    
    plt.figure(figsize=(8, 6))
    
    if max(Q_obs) > 1e6:
        plt.scatter(C, Q_obs/1e6, label="Observed Data", color="tab:blue", s=30, zorder=5)
        plt.plot(C, Q_fit/1e6, 
                 label=f"Fit: {model_name}\n({param_str})", 
                 color="tab:red", linewidth=2)
        plt.ylabel("Adsorption Capacity (×10⁶ CFU/mg)", fontsize=12)
    else:
        plt.scatter(C, Q_obs, label="Observed Data", color="tab:blue", s=30, zorder=5)
        plt.plot(C, Q_fit, 
                 label=f"Fit: {model_name}\n({param_str})", 
                 color="tab:red", linewidth=2)
        plt.ylabel("Adsorption Capacity (CFU/mg)", fontsize=12)
    
    plt.xlabel("Equilibrium Concentration (mg/L)", fontsize=12)
    plt.title(f"{model_name} Adsorption Isotherm", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    return plt.gcf()
