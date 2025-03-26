# Peso API Module

This module provides RESTful API endpoints for the Peso project functionality.

## Getting Started

1. Install dependencies:
```bash
pip install flask
```

2. Run the API server:
```bash
python run_api.py
```

The API will be available at `http://localhost:5000`.

## API Endpoints

### General

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info and available endpoints |

### Datasets

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/datasets` | GET | List all datasets (with optional filters) |
| `/datasets` | POST | Create a new dataset |
| `/datasets/<id>` | GET | Retrieve dataset details |
| `/datasets/<id>` | PUT | Update dataset (add new version) |

### Insights

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/insights/<dataset_id>?type=summary` | GET | Generate dataset summary |
| `/insights/<dataset_id>?type=marketing` | GET | Generate marketing insights |
| `/insights/<dataset_id>?type=anomalies` | GET | Detect anomalies |

### Queries

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Execute advanced query |

### Data Collection

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/collect` | POST | Generate synthetic data |

### Machine Learning

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ml/train` | POST | Train a model |
| `/ml/predict` | POST | Make predictions |
| `/ml/models` | GET | List all trained models |

## Examples

### Create a Dataset

```bash
curl -X POST http://localhost:5000/datasets \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test Dataset",
    "data_type": "non-human",
    "data": [
      {"name": "Entity A", "country": "USA", "industry": "Tech"},
      {"name": "Entity B", "country": "Germany", "industry": "Manufacturing"}
    ]
  }'
```

### Get Insights

```bash
curl http://localhost:5000/insights/1?type=marketing
```

### Train a Model

```bash
curl -X POST http://localhost:5000/ml/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": 1,
    "target_column": "industry",
    "model_type": "classifier",
    "features": ["country"]
  }'
```

### Make Predictions

```bash
curl -X POST http://localhost:5000/ml/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "classifier_1_20230101_120000",
    "data": [{"country": "USA"}, {"country": "UK"}]
  }'
```
