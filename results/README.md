# Fe₃O₄@PDA@PEI Langmuir Adsorption Isotherm Fitting (Simulation)

A **simulated demonstration** of Langmuir isotherm fitting for the adsorption capacity of Fe₃O₄@PDA@PEI nanomaterials.  
This project showcases data simulation, curve fitting, result visualization, and reproducible Python workflow, ideal for PhD applications or research portfolio.

---

## 📘 Project Overview

- **Material:** Fe₃O₄ nanoparticles sequentially modified with polydopamine (PDA) and polyethyleneimine (PEI)
- **Purpose:** Simulate and visualize the Langmuir adsorption isotherm, demonstrating ability to process, fit, and interpret materials adsorption data.
- **Skills:** Python, NumPy, SciPy, Matplotlib, scientific computing, data visualization

---

## 🧪 Simulation Workflow

1. **Model:**  
   The Langmuir isotherm equation:
   \[
   Q = \frac{Q_{max} \cdot K \cdot C}{1 + K \cdot C}
   \]
   where  
   - \( Q \): Adsorbed amount (mg/g)  
   - \( C \): Concentration (mg/L)  
   - \( Q_{max} \): Maximum adsorption capacity (mg/g)  
   - \( K \): Langmuir constant (L/mg)

2. **Data Generation:**  
   - Generate simulated concentration values (`C`)
   - Calculate theoretical adsorption (`Q_true`)
   - Add Gaussian noise to mimic experimental variation (`Q_obs`)

3. **Curve Fitting:**  
   - Use `scipy.optimize.curve_fit` to fit the Langmuir model to noisy data
   - Compare fitted parameters to true values

4. **Visualization & Output:**  
   - Plot both simulated observations and fitted isotherm
   - Save figure and data for validation and presentation

---

## 📊 Example Output

| 真实参数 | 拟合参数 | R² (拟合优度) |
|----------|---------|---------------|
| Qmax = 50.00 mg/g, K = 0.30 L/mg | Qmax ≈ 48.56 mg/g, K ≈ 0.31 L/mg | 0.96+ |

<p align="center">
  <img src="results/adsorption_fit.png" alt="Langmuir Isotherm Fitting Example" width="500"/>
</p>

---

## 🚀 Usage

### 1. Clone this repository

```bash
git clone https://github.com/<your-username>/Fe3O4-PDA-PEI-Langmuir-Fitting.git
cd Fe3O4-PDA-PEI-Langmuir-Fitting
