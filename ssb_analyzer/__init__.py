"""
SSB Analyzer - 吸附等温线分析工具包

用于Fe₃O₄@PDA@PEI纳米复合材料细菌吸附行为建模和分析的综合工具包
"""

from .analysis import IsothermAnalyzer, ModelResult, AnalysisReport
from . import adsorption_models
from . import fitting_algorithms
from . import plotting

__version__ = "1.0.0"
__author__ = "lunazhang"

__all__ = [
    "IsothermAnalyzer",
    "ModelResult", 
    "AnalysisReport",
    "adsorption_models",
    "fitting_algorithms", 
    "plotting"
]
