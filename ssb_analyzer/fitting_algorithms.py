# ssb_analyzer/fitting_algorithms.py

from scipy.optimize import curve_fit
import numpy as np

def fit_isotherm(model_func, C_data, Q_data, p0=None, bounds=None):
    """
    Fits an isotherm model to the given data.

    Parameters:
    - model_func (callable): The isotherm model function to fit (e.g., langmuir, freundlich).
    - C_data (array-like): Concentration data.
    - Q_data (array-like): Adsorption capacity data.
    - p0 (list, optional): Initial guess for the parameters.
    - bounds (tuple, optional): Bounds for the parameters.

    Returns:
    - popt (array): Optimal parameters found by the fit.
    - pcov (2D array): The estimated covariance of popt.
    """
    if bounds is None:
        popt, pcov = curve_fit(model_func, C_data, Q_data, p0=p0, maxfev=5000)
    else:
        popt, pcov = curve_fit(model_func, C_data, Q_data, p0=p0, bounds=bounds, maxfev=5000)
    
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
        return 1.0 # Perfect fit if all points are the same
    r2 = 1 - (ss_res / ss_tot)
    return r2

def fit_eis_data(model_func, f_data, Z_data, p0=None, bounds=None):
    """
    Fits a complex impedance model to data using curve_fit.

    This function works by fitting the real and imaginary parts of the
    impedance data simultaneously.

    Parameters:
    - model_func (callable): The complex impedance model function (e.g., randles_model).
    - f_data (array-like): Frequency data.
    - Z_data (array-like): Complex impedance data.
    - p0 (list, optional): Initial guess for the parameters.
    - bounds (tuple, optional): Bounds for the parameters.

    Returns:
    - popt (array): Optimal parameters found by the fit.
    - pcov (2D array): The estimated covariance of popt.
    """
    # Define a wrapper function for curve_fit that returns real and imaginary parts
    def model_wrapper(f, *params):
        Z_model = model_func(f, *params)
        return np.concatenate((Z_model.real, Z_model.imag))

    # Prepare the y-data for fitting (concatenate real and imaginary parts)
    y_data_fit = np.concatenate((Z_data.real, Z_data.imag))

    # Perform the fit
    if bounds is None:
        popt, pcov = curve_fit(model_wrapper, f_data, y_data_fit, p0=p0, maxfev=10000)
    else:
        popt, pcov = curve_fit(model_wrapper, f_data, y_data_fit, p0=p0, bounds=bounds, maxfev=10000)
    
    return popt, pcov
