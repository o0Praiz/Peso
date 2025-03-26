"""
Peso API Module - RESTful API endpoints for Peso functionality
"""

from flask import Flask, request, jsonify
import logging
import json
from database.database_module import PesoDatabase
from query.query_module import QueryEngine
from insights.insights_module import InsightsEngine
from collection.data_collection import DataCollector
from ml.ml_prediction import MLPredictionEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize components
db = PesoDatabase()
query_engine = QueryEngine(db)
insights_engine = InsightsEngine(db, query_engine)
data_collector = DataCollector(db)
ml_engine = MLPredictionEngine(db)

@app.route("/")
def index():
    """API root endpoint"""
    return jsonify({
        "name": "Peso API",
        "version": "0.1.0",
        "endpoints": [
            "/datasets",
            "/datasets/<id>",
            "/insights/<dataset_id>",
            "/query",
            "/collect",
            "/ml/train",
            "/ml/predict"
        ]
    })

@app.route("/datasets", methods=["GET", "POST"])
def datasets():
    """
    GET: List all datasets
    POST: Create a new dataset
    """
    if request.method == "GET":
        # Get filter parameters
        filters = {}
        if request.args.get("type"):
            filters["type"] = request.args.get("type")
        if request.args.get("name_contains"):
            filters["name_contains"] = request.args.get("name_contains")
        
        # Query datasets
        datasets_list = query_engine.query_datasets(filters)
        return jsonify({"datasets": datasets_list})
    
    elif request.method == "POST":
        try:
            # Parse request data
            data = request.json
            required_fields = ["name", "data_type", "data"]
            
            for field in required_fields:
                if field not in data:
                    return jsonify({"error": f"Missing required field: {field}"}), 400
            
            # Create dataset
            dataset_id = db.add_dataset(
                name=data["name"],
                data_type=data["data_type"],
                source=data.get("source"),
                metadata=data.get("metadata", {})
            )
            
            # Add data
            version = db.add_dataset_version(dataset_id, data["data"])
            
            return jsonify({
                "success": True,
                "dataset_id": dataset_id,
                "version": version
            }), 201
            
        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            return jsonify({"error": str(e)}), 500

@app.route("/datasets/<int:dataset_id>", methods=["GET", "PUT", "DELETE"])
def dataset_operations(dataset_id):
    """
    GET: Retrieve dataset
    PUT: Update dataset (add new version)
    DELETE: Delete dataset
    """
    if request.method == "GET":
        # Get version parameter
        version = request.args.get("version")
        if version:
            try:
                version = int(version)
            except ValueError:
                return jsonify({"error": "Version must be an integer"}), 400
        
        # Retrieve dataset
        dataset = db.get_dataset(dataset_id, version)
        if dataset:
            return jsonify(dataset)
        else:
            return jsonify({"error": "Dataset not found"}), 404
    
    elif request.method == "PUT":
        try:
            # Parse request data
            data = request.json
            if "data" not in data:
                return jsonify({"error": "Missing required field: data"}), 400
            
            # Add new version
            version = db.add_dataset_version(dataset_id, data["data"])
            
            return jsonify({
                "success": True,
                "dataset_id": dataset_id,
                "version": version
            })
            
        except Exception as e:
            logger.error(f"Error updating dataset: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    elif request.method == "DELETE":
        # Note: Deletion not implemented in base database module
        return jsonify({"error": "Dataset deletion not supported"}), 501

@app.route("/insights/<int:dataset_id>", methods=["GET"])
def get_insights(dataset_id):
    """Get insights for a dataset"""
    insight_type = request.args.get("type", "summary")
    version = request.args.get("version")
    
    if version:
        try:
            version = int(version)
        except ValueError:
            return jsonify({"error": "Version must be an integer"}), 400
    
    if insight_type == "summary":
        result = insights_engine.generate_dataset_summary(dataset_id, version)
    elif insight_type == "marketing":
        result = insights_engine.generate_marketing_insights(dataset_id, version)
    elif insight_type == "anomalies":
        result = insights_engine.detect_anomalies(dataset_id, version=version)
    else:
        return jsonify({"error": f"Unsupported insight type: {insight_type}"}), 400
    
    if "error" in result:
        return jsonify(result), 404
    
    return jsonify(result)

@app.route("/query", methods=["POST"])
def advanced_query():
    """Execute advanced query"""
    try:
        query_spec = request.json
        if not query_spec:
            return jsonify({"error": "Query specification required"}), 400
        
        results = query_engine.advanced_query(query_spec)
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Query execution error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/collect", methods=["POST"])
def collect_data():
    """Generate synthetic data and store in database"""
    try:
        params = request.json
        data_type = params.get("data_type", "non-human")
        count = params.get("count", 50)
        dataset_name = params.get("name", f"Synthetic {data_type.capitalize()} Data")
        
        # Generate synthetic data
        synthetic_data = data_collector.generate_synthetic_data(data_type, count)
        
        # Store the dataset
        dataset_id = data_collector.process_and_store_data(
            synthetic_data,
            dataset_name,
            data_type
        )
        
        return jsonify({
            "success": True,
            "dataset_id": dataset_id,
            "data_count": len(synthetic_data)
        })
    
    except Exception as e:
        logger.error(f"Data collection error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/ml/train", methods=["POST"])
def train_model():
    """Train a machine learning model"""
    try:
        params = request.json
        
        required_fields = ["dataset_id", "target_column", "model_type"]
        for field in required_fields:
            if field not in params:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        dataset_id = params["dataset_id"]
        target_column = params["target_column"]
        model_type = params["model_type"]
        features = params.get("features")
        model_name = params.get("model_name")
        version = params.get("version")
        
        if model_type == "classifier":
            model_info = ml_engine.train_classifier(
                dataset_id=dataset_id,
                target_column=target_column,
                features=features,
                model_name=model_name,
                version=version
            )
        elif model_type == "regressor":
            model_info = ml_engine.train_regressor(
                dataset_id=dataset_id,
                target_column=target_column,
                features=features,
                model_name=model_name,
                version=version
            )
        else:
            return jsonify({"error": f"Unsupported model type: {model_type}"}), 400
        
        if "error" in model_info:
            return jsonify(model_info), 400
        
        return jsonify(model_info)
    
    except Exception as e:
        logger.error(f"Model training error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/ml/predict", methods=["POST"])
def predict():
    """Make predictions using trained model"""
    try:
        params = request.json
        
        required_fields = ["model_name", "data"]
        for field in required_fields:
            if field not in params:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        model_name = params["model_name"]
        data = params["data"]
        
        predictions = ml_engine.predict(model_name, data)
        
        return jsonify({
            "predictions": predictions,
            "count": len(predictions)
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/ml/models", methods=["GET"])
def list_models():
    """List all trained models"""
    try:
        models = ml_engine.list_models()
        return jsonify({"models": models})
    
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
