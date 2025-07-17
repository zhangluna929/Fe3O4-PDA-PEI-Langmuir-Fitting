# ssb_analyzer/adsorption_models.py

import numpy as np

def langmuir(C, Qmax, K):
    """
    Langmuir isotherm model.

    Parameters:
    - C (array-like): Equilibrium concentration of the adsorbate.
    - Qmax (float): Maximum adsorption capacity.
    - K (float): Langmuir constant related to the energy of adsorption.

    Returns:
    - Q (array-like): Amount of adsorbate adsorbed per unit mass of adsorbent.
    """
    return (Qmax * K * C) / (1 + K * C)

def freundlich(C, Kf, n):
    """
    Freundlich isotherm model.

    Q = Kf * C^(1/n)

    Parameters:
    - C (array-like): Equilibrium concentration.
    - Kf (float): Freundlich capacity factor.
    - n (float): Freundlich intensity parameter.

    Returns:
    - Q (array-like): Adsorbed amount.
    """
    return Kf * (C**(1/n))

def sips(C, Qmax, Ks, beta):
    """
    Sips (Langmuir-Freundlich) isotherm model.

    Q = (Qmax * (Ks*C)^beta) / (1 + (Ks*C)^beta)

    Parameters:
    - C (array-like): Equilibrium concentration.
    - Qmax (float): Maximum adsorption capacity from Sips model.
    - Ks (float): Sips model constant.
    - beta (float): Sips model exponent (heterogeneity factor).

    Returns:
    - Q (array-like): Adsorbed amount.
    """
    return (Qmax * (Ks * C)**beta) / (1 + (Ks * C)**beta)
