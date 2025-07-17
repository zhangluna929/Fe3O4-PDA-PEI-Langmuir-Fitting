# ssb_analyzer/electrochemical_models.py
import numpy as np

def randles_model(f, R_s, R_ct, C_dl):
    """
    Calculates the impedance of a simple Randles circuit.

    Z = R_s + R_ct / (1 + jωR_ctC_dl)
    where ω = 2πf

    Parameters:
    - f (array-like): Frequency array.
    - R_s (float): Series Resistance (Ohmic resistance).
    - R_ct (float): Charge Transfer Resistance.
    - C_dl (float): Double Layer Capacitance.

    Returns:
    - Z (array-like): Complex impedance.
    """
    omega = 2 * np.pi * np.asarray(f)
    Z = R_s + R_ct / (1 + 1j * omega * R_ct * C_dl)
    return Z
