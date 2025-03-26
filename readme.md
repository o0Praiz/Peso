# Peso

## Marketing Data Warehouse Project

Peso is a specialized data warehouse designed for collecting, storing, analyzing, and visualizing marketing datasets with a focus on non-human marketing data. The system is built with modularity and self-evolution in mind.

## Key Features

- **Modular Database Architecture**: Flexible SQLite-based storage with versioning
- **Advanced Querying**: Complex search capabilities with aggregation and filtering
- **AI-Powered Insights**: Integration with multiple AI tools for data enrichment
- **ML Prediction**: Classification and regression models for marketing predictions
- **Data Collection Framework**: Tools for synthetic and public data acquisition

## Project Structure

```
peso-project/
│
├── database/
│   └── database_module.py     # Core database functionality
│
├── query/
│   └── query_module.py        # Advanced querying engine
│
├── insights/
│   └── insights_module.py     # Data analysis and insights generation
│
├── ml/
│   └── ml_prediction.py       # Machine learning prediction module
│
├── collection/
│   └── data_collection.py     # Data collection framework
│
├── integration/
│   └── ai_integration.py      # AI tool integration
│
├── docs/
│   └── project_progress.md    # Ongoing updates document
│
└── README.md                  # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- SQLite3
- Required Python packages listed in `requirements.txt`

### Installation

1. Clone this repository
   ```
   git clone https://github.com/o0Praiz/Peso.git
   cd Peso
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Initialize the database
   ```python
   from database.database_module import PesoDatabase
   
   # Initialize the database
   db = PesoDatabase()
   ```

## Usage Examples

### Adding a Dataset

```python
from database.database_module import PesoDatabase

db = PesoDatabase()

# Add a new dataset
dataset_id = db.add_dataset(
    name="Global Marketing Contacts",
    data_type="non-human",
    metadata={"description": "Non-human marketing targets"}
)

# Add data to the dataset
sample_data = [
    {"name": "Entity A", "country": "USA", "industry": "Tech"},
    {"name": "Entity B", "country": "Germany", "industry": "Manufacturing"}
]
db.add_dataset_version(dataset_id, sample_data)
```

### Generating Insights

```python
from database.database_module import PesoDatabase
from insights.insights_module import InsightsEngine

db = PesoDatabase()
insights = InsightsEngine(db)

# Generate summary for a dataset
dataset_id = 1
summary = insights.generate_dataset_summary(dataset_id)
print(summary)
```

## Roadmap

See the [Project Progress Document](docs/project_progress.md) for detailed development status and roadmap.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Contributors to the project
- Open-source libraries utilized
