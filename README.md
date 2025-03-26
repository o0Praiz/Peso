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
- **API Layer**: RESTful API for external access to Peso functionality (v0.6 in progress)

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
├── api/                           # API module (in development)
│   ├── api_module.py              # RESTful API endpoints 
│   └── api_server.py              # API server implementation
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

### API Layer (Beta)

The API layer is currently in development (v0.6) and provides RESTful access to Peso functionality:

```bash
# Start the API server
python api/api_server.py --host 127.0.0.1 --port 8000

# Example API request with curl
curl -X GET http://localhost:8000/datasets -H "X-API-Key: your_api_key"
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

- **Current Version**: v0.5 (Core functionality complete)
- **In Development**: v0.6 (API Layer)
- **Next Milestones**:
  1. **v0.7**: Visualization dashboard with interactive data exploration
  2. **v0.8**: Self-improving capabilities with automated model retraining
  3. **v0.9**: Automated data source discovery and schema detection
  4. **v1.0**: Comprehensive marketing recommendation engine

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is proprietary and confidential.

## Contact

For more information, please contact the project owner.
