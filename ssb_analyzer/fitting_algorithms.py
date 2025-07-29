# ssb_analyzer/fitting_algorithms.py

from scipy.optimize import curve_fit
import numpy as np

def fit_isotherm(model_func, C_data, Q_data, p0=None, bounds=None, maxfev=5000):
    """
    Fits an isotherm model to the given data.

    Parameters:
    - model_func (callable): The isotherm model function to fit (e.g., langmuir, freundlich).
    - C_data (array-like): Concentration data.
    - Q_data (array-like): Adsorption capacity data.
    - p0 (list, optional): Initial guess for the parameters.
    - bounds (tuple, optional): Bounds for the parameters.
    - maxfev (int, optional): Maximum number of function evaluations. Default 5000.

    Returns:
    - popt (array): Optimal parameters found by the fit.
    - pcov (2D array): The estimated covariance of popt.
    """
    if bounds is None:
        popt, pcov = curve_fit(model_func, C_data, Q_data, p0=p0, maxfev=maxfev)
    else:
        popt, pcov = curve_fit(model_func, C_data, Q_data, p0=p0, bounds=bounds, maxfev=maxfev)
    
    return popt, pcov

def calculate_r2(Q_obs, Q_fit):
    """
    Calculates the coefficient of determination (R²).

    Parameters:
    - Q_obs (array-like): Observed data.
    - Q_fit (array-like): Fitted data from the model.

    Returns:
    - r2 (float): The R² value.
    """
    ss_res = np.sum((Q_obs - Q_fit)**2)
    ss_tot = np.sum((Q_obs - np.mean(Q_obs))**2)
    if ss_tot == 0:
        return 1.0
    r2 = 1 - (ss_res / ss_tot)
    return r2


