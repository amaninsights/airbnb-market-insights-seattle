"""
Airbnb Market Insights - Seattle
================================

A production-grade analytics pipeline for Seattle Airbnb market analysis.

This package provides:
- Data ingestion from Kaggle datasets
- ETL transformations and cleaning
- Price prediction and market analysis
- Neighborhood performance metrics
- Automated visualization generation

Author: Aman Saroha
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "Aman Saroha"

from src.analysis import market_analysis
from src.data import loader, transformer
from src.visualization import charts

__all__ = ["loader", "transformer", "market_analysis", "charts"]
