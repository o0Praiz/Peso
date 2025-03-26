# Peso

A modular, extensible data warehouse for marketing datasets with a focus on non-human marketing data.

## Overview

Peso is a comprehensive data warehouse solution designed specifically for marketing datasets. It provides a flexible, scalable infrastructure for collecting, storing, analyzing, and deriving insights from marketing data, with special capabilities for non-human marketing targets.

The system is built with a modular architecture that enables continuous evolution and self-improvement through AI integration.

## Key Features

- **Modular Database Architecture**: Flexible SQLite-based database that supports versioning and metadata tracking
- **Versatile Data Collection**: Tools for synthetic data generation and public data aggregation
- **AI-Powered Enrichment**: Integration with leading AI platforms to enhance marketing datasets
- **Advanced Querying**: Sophisticated query capabilities with filtering, grouping, and aggregation
- **Insights Generation**: Automated statistical analysis and marketing-specific insights
- **Machine Learning**: Built-in ML prediction capabilities for classification and regression tasks
- **Extensible Framework**: Designed for continuous evolution with easy addition of new capabilities

## Project Structure

```
peso-project/
├── README.md                      # This file
├── project_progress.md            # Ongoing project updates
├── config.json                    # Configuration file
├── main.py                        # Main entry point script
│
├── database/                      # Database module
│   └── peso_database_module.py    # Core database functionality
│
├── data_collection/               # Data collection module
│   └── peso_data_collection.py    # Data collection tools
│
├── ai_integration/                # AI integration module
│   └── peso_ai_integration.py     # AI tool connectors
│
├── query/                         # Query module
│   └── peso_query_module.py       # Advanced query capabilities
│
├── insights/                      # Insights generation module
│   └── peso_insights_generation.py # Data analysis and insights
│
├── ml/                            # Machine learning module
│   └── peso_ml_prediction.py      # ML model training and prediction
│
├── integration/                   # Integration layer
│   └── peso_integration_module.py # Cross-module orchestration
│
└── models/                        # Directory for saved ML models
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/o0Praiz/Peso.git
   cd Peso
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

## Usage

### Basic Commands

```bash
# Create a synthetic non-human marketing dataset
python main.py dataset create --name "Tech Product Marketing" --type "non-human" --count 500

# List all datasets
python main.py dataset list

# Analyze a dataset
python main.py dataset analyze --id 1 --output analysis.json

# Enrich a dataset with AI
python main.py dataset enrich --id 1 --tool openai

# Train a machine learning model
python main.py ml train --dataset 1 --target "industry" --type classifier

# Make predictions with a trained model
python main.py ml predict --model classifier_1_20250326_120000 --data new_data.json

# Query data
python main.py query --dataset 1 --filter filters.json --output results.json
```

### Configuration

The application is configured through the `config.json` file:

```json
{
  "database": {
    "path": "peso_marketing.db"
  },
  "ai_tools": {
    "openai": "YOUR_OPENAI_API_KEY",
    "anthropic": "YOUR_ANTHROPIC_API_KEY"
  },
  "ml": {
    "models_dir": "models"
  },
  "data_sources": {
    "public_apis": [
      "https://api.example.com/marketing-data"
    ]
  }
}
```

## Development Roadmap

- **Current Version**: v0.5
- **Next Milestones**:
  1. API layer for external access (v0.6)
  2. Visualization dashboard (v0.7)
  3. Self-improving capabilities (v0.8)
  4. Automated data source discovery (v0.9)
  5. Comprehensive marketing recommendation engine (v1.0)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is proprietary and confidential.

## Contact

For more information, please contact [project owner].
