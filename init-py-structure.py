"""
Peso Module Initialization Files

This shows the structure and contents of each __init__.py file in the Peso project's modules.
All modules follow the same pattern for consistency and proper module imports.
"""

# database/__init__.py
"""
Peso Database Module
Provides core database functionality for the Peso data warehouse.
"""

__version__ = "0.5.0"

from database.database_module import PesoDatabase

__all__ = ['PesoDatabase']


# query/__init__.py
"""
Peso Query Module
Provides advanced querying capabilities for the Peso data warehouse.
"""

__version__ = "0.5.0"

from query.query_module import QueryEngine

__all__ = ['QueryEngine']


# insights/__init__.py
"""
Peso Insights Module
Provides data analysis and insights generation for the Peso data warehouse.
"""

__version__ = "0.5.0"

from insights.insights_module import InsightsEngine

__all__ = ['InsightsEngine']


# collection/__init__.py
"""
Peso Collection Module
Provides data collection and processing for the Peso data warehouse.
"""

__version__ = "0.5.0"

from collection.data_collection import DataCollector

__all__ = ['DataCollector']


# integration/__init__.py
"""
Peso Integration Module
Provides AI tool integration for the Peso data warehouse.
"""

__version__ = "0.5.0"

from integration.ai_integration import AIToolIntegration

__all__ = ['AIToolIntegration']


# ml/__init__.py
"""
Peso Machine Learning Module
Provides ML prediction capabilities for the Peso data warehouse.
"""

__version__ = "0.5.0"

from ml.ml_prediction import MLPredictionEngine

__all__ = ['MLPredictionEngine']
