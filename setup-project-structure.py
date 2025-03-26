#!/usr/bin/env python3
"""
Peso Project Setup Script
Creates the project directory structure and initializes modules
"""

import os
import sys
import shutil
import argparse

# Module definitions
MODULES = {
    'database': {
        'class': 'PesoDatabase',
        'file': 'database_module.py',
        'desc': 'Provides core database functionality for the Peso data warehouse.'
    },
    'query': {
        'class': 'QueryEngine',
        'file': 'query_module.py',
        'desc': 'Provides advanced querying capabilities for the Peso data warehouse.'
    },
    'insights': {
        'class': 'InsightsEngine',
        'file': 'insights_module.py',
        'desc': 'Provides data analysis and insights generation for the Peso data warehouse.'
    },
    'collection': {
        'class': 'DataCollector',
        'file': 'data_collection.py',
        'desc': 'Provides data collection and processing for the Peso data warehouse.'
    },
    'integration': {
        'class': 'AIToolIntegration',
        'file': 'ai_integration.py',
        'desc': 'Provides AI tool integration for the Peso data warehouse.'
    },
    'ml': {
        'class': 'MLPredictionEngine',
        'file': 'ml_prediction.py',
        'desc': 'Provides ML prediction capabilities for the Peso data warehouse.'
    }
}

# Additional directories to create
EXTRA_DIRS = [
    'models',
    'logs',
    'backups',
    'insights_reports'
]

def create_init_file(module_name, module_data, base_dir):
    """Create an __init__.py file for a module"""
    init_path = os.path.join(base_dir, module_name, '__init__.py')
    
    with open(init_path, 'w') as f:
        f.write(f'''"""
Peso {module_name.title()} Module
{module_data['desc']}
"""

__version__ = "0.5.0"

from {module_name}.{module_data['file'].replace('.py', '')} import {module_data['class']}

__all__ = ['{module_data['class']}']
''')
    
    print(f"Created {init_path}")

def setup_project_structure(base_dir, force=False):
    """Set up the project directory structure"""
    # Check if directory exists
    if os.path.exists(base_dir) and not force:
        print(f"Directory {base_dir} already exists. Use --force to overwrite.")
        return False
    
    # Create base directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created project directory: {base_dir}")
    
    # Create README.md if it doesn't exist
    readme_path = os.path.join(base_dir, 'README.md')
    if not os.path.exists(readme_path) or force:
        with open(readme_path, 'w') as f:
            f.write('''# Peso

A modular, extensible data warehouse for marketing datasets with a focus on non-human marketing data.

## Overview

Peso is a comprehensive data warehouse solution designed specifically for marketing datasets. It provides a flexible, scalable infrastructure for collecting, storing, analyzing, and deriving insights from marketing data, with special capabilities for non-human marketing targets.

The system is built with a modular architecture that enables continuous evolution and self-improvement through AI integration.

## Project Structure

```
peso-project/
├── README.md                      # This file
├── project_progress.md            # Ongoing project updates
├── config.json                    # Configuration file
├── main.py                        # Main entry point script
│
├── database/                      # Database module
│   └── database_module.py         # Core database functionality
│
├── collection/                    # Data collection module
│   └── data_collection.py         # Data collection tools
│
├── integration/                   # AI integration module
│   └── ai_integration.py          # AI tool connectors
│
├── query/                         # Query module
│   └── query_module.py            # Advanced query capabilities
│
├── insights/                      # Insights generation module
│   └── insights_module.py         # Data analysis and insights
│
├── ml/                            # Machine learning module
│   └── ml_prediction.py           # ML model training and prediction
│
└── models/                        # Directory for saved ML models
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/peso.git
   cd peso
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the application:
   ```
   cp config.example.json config.json
   # Edit config.json with your settings
   ```
''')
        print(f"Created {readme_path}")
    
    # Create project_progress.md if it doesn't exist
    progress_path = os.path.join(base_dir, 'project_progress.md')
    if not os.path.exists(progress_path) or force:
        with open(progress_path, 'w') as f:
            f.write('''# Peso Project Progress

## Current Status

- **Version**: 0.5.0
- **Last Updated**: April 2025
- **Status**: In Development

## Completed Modules

- Database Module
- Query Module
- Data Collection Module
- AI Integration Module
- Insights Generation Module
- ML Prediction Module

## Next Steps

1. API Layer Development
2. Visualization Dashboard
3. Self-Improvement Mechanisms
''')
        print(f"Created {progress_path}")
    
    # Create requirements.txt if it doesn't exist
    requirements_path = os.path.join(base_dir, 'requirements.txt')
    if not os.path.exists(requirements_path) or force:
        with open(requirements_path, 'w') as f:
            f.write('''# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0

# Database
sqlalchemy>=1.4.0

# HTTP and API
requests>=2.26.0

# AI integrations
openai>=1.0.0
anthropic>=0.5.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Utilities
python-dotenv>=0.19.0
''')
        print(f"Created {requirements_path}")
    
    # Create module directories and __init__.py files
    for module_name, module_data in MODULES.items():
        module_dir = os.path.join(base_dir, module_name)
        if not os.path.exists(module_dir):
            os.makedirs(module_dir)
            print(f"Created module directory: {module_dir}")
        
        # Create __init__.py file
        create_init_file(module_name, module_data, base_dir)
    
    # Create extra directories
    for extra_dir in EXTRA_DIRS:
        dir_path = os.path.join(base_dir, extra_dir)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
    
    # Create example config file
    config_example_path = os.path.join(base_dir, 'config.example.json')
    if not os.path.exists(config_example_path) or force:
        with open(config_example_path, 'w') as f:
            f.write('''{
  "database": {
    "path": "peso_marketing.db",
    "backup_dir": "backups",
    "max_versions": 10
  },
  "ai_tools": {
    "openai": "YOUR_OPENAI_API_KEY",
    "anthropic": "YOUR_ANTHROPIC_API_KEY"
  },
  "ml": {
    "models_dir": "models",
    "default_test_size": 0.2,
    "default_classifier": "random_forest",
    "default_regressor": "gradient_boosting"
  },
  "data_sources": {
    "public_apis": [
      "https://api.example.com/marketing-data"
    ],
    "refresh_interval_hours": 24
  },
  "logging": {
    "level": "INFO",
    "file": "logs/peso.log",
    "max_size_mb": 10,
    "backup_count": 5
  }
}''')
        print(f"Created {config_example_path}")
    
    # Create main.py if it doesn't exist
    main_path = os.path.join(base_dir, 'main.py')
    if not os.path.exists(main_path) or force:
        with open(main_path, 'w') as f:
            f.write('''#!/usr/bin/env python3
"""
Peso: Marketing Data Warehouse
Main module for demonstrating functionality
"""

import logging
import json
from database import PesoDatabase
from query import QueryEngine
from insights import InsightsEngine
from collection import DataCollector
from integration import AIToolIntegration
from ml import MLPredictionEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_components():
    """Initialize all project components"""
    logger.info("Initializing Peso project components...")
    
    # Initialize database
    db = PesoDatabase()
    
    # Initialize other components
    query_engine = QueryEngine(db)
    insights_engine = InsightsEngine(db, query_engine)
    data_collector = DataCollector(db)
    ai_integration = AIToolIntegration()
    ml_engine = MLPredictionEngine(db)
    
    return {
        'database': db,
        'query': query_engine,
        'insights': insights_engine,
        'collector': data_collector,
        'ai': ai_integration,
        'ml': ml_engine
    }

def demo_data_collection():
    """Demonstrate data collection functionality"""
    logger.info("Demonstrating data collection...")
    
    # Initialize components
    components = initialize_components()
    db = components['database']
    collector = components['collector']
    
    # Generate synthetic non-human marketing data
    synthetic_data = collector.generate_synthetic_data('non-human', 50)
    logger.info(f"Generated {len(synthetic_data)} synthetic non-human marketing records")
    
    # Store the dataset
    dataset_id = collector.process_and_store_data(
        synthetic_data,
        "Demo Non-Human Marketing Dataset",
        "non-human"
    )
    
    logger.info(f"Stored dataset with ID: {dataset_id}")
    return dataset_id

def demo_insights_generation(dataset_id):
    """Demonstrate insights generation functionality"""
    logger.info("Demonstrating insights generation...")
    
    # Initialize components
    components = initialize_components()
    insights = components['insights']
    
    # Generate dataset summary
    summary = insights.generate_dataset_summary(dataset_id)
    logger.info("Generated dataset summary")
    
    # Generate marketing insights
    marketing_insights = insights.generate_marketing_insights(dataset_id)
    logger.info("Generated marketing insights")
    
    # Detect anomalies
    anomalies = insights.detect_anomalies(dataset_id)
    logger.info("Completed anomaly detection")
    
    return {
        'summary': summary,
        'marketing_insights': marketing_insights,
        'anomalies': anomalies
    }

def demo_query_execution():
    """Demonstrate query execution functionality"""
    logger.info("Demonstrating query execution...")
    
    # Initialize components
    components = initialize_components()
    query = components['query']
    
    # Query non-human datasets
    datasets = query.query_datasets({'type': 'non-human'})
    logger.info(f"Found {len(datasets)} non-human datasets")
    
    if datasets:
        # Advanced query on the first dataset
        dataset_id = datasets[0]['id']
        results = query.advanced_query({
            'dataset_id': dataset_id,
            'content_search': {'country': 'USA'},
            'group_by': 'industry'
        })
        
        logger.info(f"Advanced query found {results.get('metrics', {}).get('total_records', 0)} records")
        return results
    else:
        logger.warning("No datasets available for query demonstration")
        return None

def demo_ml_prediction(dataset_id):
    """Demonstrate machine learning prediction functionality"""
    logger.info("Demonstrating ML prediction...")
    
    # Initialize components
    components = initialize_components()
    ml = components['ml']
    
    # Train a classifier
    model_info = ml.train_classifier(
        dataset_id=dataset_id,
        target_column="industry",
        features=["country"]
    )
    
    logger.info(f"Trained classifier model: {model_info.get('model_name')}")
    
    # Generate test data for prediction
    test_data = [
        {"country": "USA"},
        {"country": "Germany"},
        {"country": "Japan"}
    ]
    
    # Make predictions
    if 'error' not in model_info:
        predictions = ml.predict(model_info['model_name'], test_data)
        logger.info(f"Generated predictions: {predictions}")
        return predictions
    else:
        logger.error(f"Model training failed: {model_info.get('error')}")
        return None

def main():
    """Main execution function"""
    logger.info("Starting Peso Project demonstration")
    
    try:
        # Demonstrate data collection
        dataset_id = demo_data_collection()
        
        # Demonstrate insights generation
        insights_results = demo_insights_generation(dataset_id)
        
        # Demonstrate query execution
        query_results = demo_query_execution()
        
        # Demonstrate ML prediction
        ml_results = demo_ml_prediction(dataset_id)
        
        logger.info("Demonstration completed successfully")
        
        # Return comprehensive results
        return {
            'dataset_id': dataset_id,
            'insights': insights_results,
            'query': query_results,
            'ml_predictions': ml_results
        }
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        return {'error': str(e)}

if __name__ == "__main__":
    results = main()
    print(json.dumps(results, indent=2, default=str))
''')
        print(f"Created {main_path}")
        
    print("\nProject structure setup complete!")
    return True

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Set up Peso project structure")
    parser.add_argument("--dir", default="peso-project", help="Base directory for the project")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    success = setup_project_structure(args.dir, args.force)
    sys.exit(0 if success else 1)
