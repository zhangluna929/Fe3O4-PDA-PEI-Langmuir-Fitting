import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

from . import adsorption_models
from .fitting_algorithms import fit_isotherm, calculate_r2
from .plotting import plot_adsorption_isotherm


@dataclass
class ModelResult:
    """存储单个模型拟合结果的数据类"""
    name: str
    parameters: Dict[str, float]
    r2: float
    rmse: float
    aic: float
    fitted_values: np.ndarray
    parameter_errors: Optional[Dict[str, float]] = None


@dataclass
class AnalysisReport:
    """存储完整分析报告的数据类"""
    best_model: ModelResult
    all_models: Dict[str, ModelResult]
    data_summary: Dict[str, Any]
    model_ranking: List[Tuple[str, float]]


class IsothermAnalyzer:
    """
    吸附等温线综合分析器
    
    功能:
    - 多模型自动拟合和比较
    - 统计分析和模型选择
    - 参数不确定性评估
    - 结果可视化和报告生成
    """
    
    def __init__(self):
        self.models = {
            "Langmuir": {
                "func": adsorption_models.langmuir,
                "param_names": ["Qmax", "K"],
                "default_p0": [1.0e8, 0.1],
                "default_bounds": ([1e6, 0.01], [5e8, 2.0])
            },
            "Freundlich": {
                "func": adsorption_models.freundlich,
                "param_names": ["Kf", "n"],
                "default_p0": [1e7, 1.5],
                "default_bounds": ([1e6, 0.5], [1e9, 5.0])
            },
            "Sips": {
                "func": adsorption_models.sips,
                "param_names": ["Qmax", "Ks", "beta"],
                "default_p0": [1.2e8, 0.12, 0.8],
                "default_bounds": ([1e6, 0.01, 0.1], [5e8, 2.0, 1.0])
            }
        }
        
        self.C_data = None
        self.Q_data = None
        self.results = {}
    
    def load_data(self, concentration: np.ndarray, adsorption: np.ndarray) -> None:
        """
        加载实验数据
        
        Parameters:
        - concentration: 平衡浓度数组 (mg/L)
        - adsorption: 吸附量数组 (CFU/mg)
        """
        self.C_data = np.asarray(concentration)
        self.Q_data = np.asarray(adsorption)
        
        if len(self.C_data) != len(self.Q_data):
            raise ValueError("浓度和吸附量数据长度不匹配")
        
        if np.any(self.C_data < 0) or np.any(self.Q_data < 0):
            warnings.warn("检测到负值数据，可能影响拟合结果")
    
    def generate_synthetic_data(self, model_name: str = "Sips", 
                              C_range: Tuple[float, float] = (0.1, 20),
                              n_points: int = 50,
                              noise_level: float = 1.5e6,
                              random_seed: int = 42,
                              **true_params) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成合成数据用于模型测试
        
        Parameters:
        - model_name: 用于生成数据的模型名称
        - C_range: 浓度范围 (min, max)
        - n_points: 数据点数量
        - noise_level: 噪声标准差
        - random_seed: 随机种子
        - **true_params: 真实模型参数
        """
        np.random.seed(random_seed)
        C = np.linspace(C_range[0], C_range[1], n_points)
        
        if model_name not in self.models:
            raise ValueError(f"未知模型: {model_name}")
        
        model_func = self.models[model_name]["func"]
        
        # 使用默认参数或用户提供的参数
        if not true_params:
            if model_name == "Langmuir":
                true_params = {"Qmax": 1.5e8, "K": 0.15}
            elif model_name == "Freundlich":
                true_params = {"Kf": 1e7, "n": 2.0}
            elif model_name == "Sips":
                true_params = {"Qmax": 1.5e8, "Ks": 0.15, "beta": 0.85}
        
        Q_true = model_func(C, **true_params)
        noise = np.random.normal(0, noise_level, size=C.size)
        Q_obs = Q_true + noise
        Q_obs = np.maximum(Q_obs, 0.1e6)  # 确保为正值
        
        self.load_data(C, Q_obs)
        return C, Q_obs
    
    def fit_single_model(self, model_name: str, 
                        p0: Optional[List[float]] = None,
                        bounds: Optional[Tuple] = None,
                        maxfev: int = 5000) -> ModelResult:
        """
        拟合单个等温线模型
        
        Parameters:
        - model_name: 模型名称
        - p0: 初始参数猜测
        - bounds: 参数边界
        - maxfev: 最大迭代次数
        """
        if self.C_data is None or self.Q_data is None:
            raise ValueError("请先加载数据")
        
        if model_name not in self.models:
            raise ValueError(f"未知模型: {model_name}")
        
        model_info = self.models[model_name]
        model_func = model_info["func"]
        
        if p0 is None:
            p0 = model_info["default_p0"]
        if bounds is None:
            bounds = model_info["default_bounds"]
        
        try:
            popt, pcov = fit_isotherm(model_func, self.C_data, self.Q_data, 
                                    p0=p0, bounds=bounds, maxfev=maxfev)
            
            Q_fit = model_func(self.C_data, *popt)
            r2 = calculate_r2(self.Q_data, Q_fit)
            rmse = np.sqrt(np.mean((self.Q_data - Q_fit)**2))
            
            # 计算AIC (赤池信息准则)
            n = len(self.Q_data)
            k = len(popt)
            rss = np.sum((self.Q_data - Q_fit)**2)
            aic = n * np.log(rss/n) + 2*k
            
            # 参数字典
            param_dict = dict(zip(model_info["param_names"], popt))
            
            # 参数标准误差
            param_errors = None
            try:
                param_std_errors = np.sqrt(np.diag(pcov))
                param_errors = dict(zip(model_info["param_names"], param_std_errors))
            except:
                warnings.warn(f"无法计算{model_name}模型的参数标准误差")
            
            return ModelResult(
                name=model_name,
                parameters=param_dict,
                r2=r2,
                rmse=rmse,
                aic=aic,
                fitted_values=Q_fit,
                parameter_errors=param_errors
            )
            
        except Exception as e:
            raise RuntimeError(f"{model_name}模型拟合失败: {str(e)}")
    
    def fit_all_models(self, maxfev: int = 5000) -> Dict[str, ModelResult]:
        """
        拟合所有可用模型并返回结果
        """
        self.results = {}
        failed_models = []
        
        for model_name in self.models.keys():
            try:
                result = self.fit_single_model(model_name, maxfev=maxfev)
                self.results[model_name] = result
            except RuntimeError as e:
                failed_models.append(model_name)
                warnings.warn(str(e))
        
        if not self.results:
            raise RuntimeError("所有模型拟合均失败")
        
        if failed_models:
            print(f"警告: 以下模型拟合失败: {failed_models}")
        
        return self.results
    
    def select_best_model(self, criterion: str = "r2") -> ModelResult:
        """
        根据指定准则选择最佳模型
        
        Parameters:
        - criterion: 选择准则 ("r2", "aic", "rmse")
        """
        if not self.results:
            raise ValueError("请先进行模型拟合")
        
        if criterion == "r2":
            best_name = max(self.results, key=lambda name: self.results[name].r2)
        elif criterion == "aic":
            best_name = min(self.results, key=lambda name: self.results[name].aic)
        elif criterion == "rmse":
            best_name = min(self.results, key=lambda name: self.results[name].rmse)
        else:
            raise ValueError("无效的选择准则，请使用 'r2', 'aic', 或 'rmse'")
        
        return self.results[best_name]
    
    def rank_models(self, criterion: str = "r2") -> List[Tuple[str, float]]:
        """
        根据指定准则对模型进行排序
        """
        if not self.results:
            raise ValueError("请先进行模型拟合")
        
        if criterion == "r2":
            ranking = sorted(self.results.items(), 
                           key=lambda x: x[1].r2, reverse=True)
        elif criterion == "aic":
            ranking = sorted(self.results.items(), 
                           key=lambda x: x[1].aic)
        elif criterion == "rmse":
            ranking = sorted(self.results.items(), 
                           key=lambda x: x[1].rmse)
        else:
            raise ValueError("无效的选择准则")
        
        return [(name, getattr(result, criterion)) for name, result in ranking]
    
    def generate_report(self, criterion: str = "r2") -> AnalysisReport:
        """
        生成完整的分析报告
        """
        if not self.results:
            raise ValueError("请先进行模型拟合")
        
        best_model = self.select_best_model(criterion)
        ranking = self.rank_models(criterion)
        
        data_summary = {
            "n_points": len(self.C_data),
            "concentration_range": (self.C_data.min(), self.C_data.max()),
            "adsorption_range": (self.Q_data.min(), self.Q_data.max()),
            "mean_adsorption": self.Q_data.mean(),
            "std_adsorption": self.Q_data.std()
        }
        
        return AnalysisReport(
            best_model=best_model,
            all_models=self.results,
            data_summary=data_summary,
            model_ranking=ranking
        )
    
    def calculate_prediction_intervals(self, model_name: str, 
                                     confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算模型预测的置信区间（需要实现bootstrap方法）
        """
        # 简化实现，实际应用中可以使用bootstrap重采样
        if model_name not in self.results:
            raise ValueError(f"模型 {model_name} 未拟合")
        
        result = self.results[model_name]
        residuals = self.Q_data - result.fitted_values
        std_residual = np.std(residuals)
        
        # 使用简单的±标准差作为近似置信区间
        alpha = 1 - confidence_level
        z_score = 1.96  # 95%置信水平的z值
        
        lower_bound = result.fitted_values - z_score * std_residual
        upper_bound = result.fitted_values + z_score * std_residual
        
        return lower_bound, upper_bound
    
    def export_results(self, filename: str = "isotherm_analysis_results.csv") -> None:
        """
        导出拟合结果到CSV文件
        """
        if not self.results:
            raise ValueError("请先进行模型拟合")
        
        data_dict = {
            "Concentration": self.C_data,
            "Observed_Adsorption": self.Q_data
        }
        
        for name, result in self.results.items():
            data_dict[f"{name}_Fitted"] = result.fitted_values
            data_dict[f"{name}_Residuals"] = self.Q_data - result.fitted_values
        
        df = pd.DataFrame(data_dict)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"结果已导出到: {filename}")
    
    def plot_comparison(self, show_intervals: bool = False) -> None:
        """
        绘制所有模型的比较图
        """
        if not self.results:
            raise ValueError("请先进行模型拟合")
        
        best_model = self.select_best_model()
        fig = plot_adsorption_isotherm(
            self.C_data,
            self.Q_data,
            best_model.fitted_values,
            best_model.name,
            best_model.parameters
        )
        
        return fig
    
    def print_summary(self) -> None:
        """
        打印分析结果摘要
        """
        if not self.results:
            print("尚未进行模型拟合")
            return
        
        report = self.generate_report()
        
        print("=" * 50)
        print("吸附等温线分析结果摘要")
        print("=" * 50)
        
        print(f"数据点数: {report.data_summary['n_points']}")
        print(f"浓度范围: {report.data_summary['concentration_range'][0]:.2f} - "
              f"{report.data_summary['concentration_range'][1]:.2f} mg/L")
        
        print(f"\n最佳模型: {report.best_model.name}")
        print(f"R² = {report.best_model.r2:.4f} ({report.best_model.r2*100:.1f}%)")
        print(f"RMSE = {report.best_model.rmse:.2e}")
        print(f"AIC = {report.best_model.aic:.2f}")
        
        print(f"\n拟合参数:")
        for param, value in report.best_model.parameters.items():
            if param.lower() == 'qmax':
                print(f"  {param}: {value:.2e} CFU/mg")
            else:
                print(f"  {param}: {value:.4f}")
        
        print(f"\n模型排序 (按R²):")
        for i, (name, r2) in enumerate(report.model_ranking, 1):
            print(f"  {i}. {name}: R² = {r2:.4f}")
        
        print("=" * 50)
