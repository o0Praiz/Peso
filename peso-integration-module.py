import logging
import json
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Import all required modules
from peso_database_module import PesoDatabase
from peso_data_collection import DataCollector
from peso_ai_integration import AIToolIntegration
from peso_query_module import QueryEngine
from peso_insights_generation import InsightsEngine
from peso_ml_prediction import MLPredictionEngine

class PesoIntegration:
    """
    Main integration class that brings together all Peso modules
    """
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the Peso integration with all components
        
        Args:
            config_path: Path to configuration file
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("peso.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Peso Integration")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize database
        self.db = PesoDatabase(self.config.get('database', {}).get('path', 'peso_marketing.db'))
        
        # Initialize query engine
        self.query_engine = QueryEngine(self.db)
        
        # Initialize data collector
        self.data_collector = DataCollector(self.db)
        
        # Initialize AI integration
        self.ai_integration = AIToolIntegration(self.config.get('ai_tools', {}))
        
        # Initialize insights engine
        self.insights_engine = InsightsEngine(self.db, self.query_engine)
        
        # Initialize ML engine
        models_dir = self.config.get('ml', {}).get('models_dir', 'models')
        self.ml_engine = MLPredictionEngine(self.db, models_dir)
        
        self.logger.info("Peso Integration successfully initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary with configuration
        """
        default_config = {
            "database": {
                "path": "peso_marketing.db"
            },
            "ai_tools": {
                "openai": None,
                "anthropic": None
            },
            "ml": {
                "models_dir": "models"
            },
            "data_sources": {
                "public_apis": []
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"Loaded configuration from {config_path}")
                return config
            else:
                self.logger.warning(f"Configuration file {config_path} not found, using defaults")
                return default_config
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return default_config
    
    def collect_and_process_data(self, name: str, data_type: str, synthetic_count: int = 100) -> int:
        """
        Collect and process data, then store in database
        
        Args:
            name: Name for the dataset
            data_type: Type of data (human/non-human)
            synthetic_count: Number of synthetic records to generate
            
        Returns:
            Dataset ID
        """
        try:
            self.logger.info(f"Collecting {data_type} data: {name}")
            
            # Generate synthetic data
            data = self.data_collector.generate_synthetic_data(data_type, synthetic_count)
            
            # Process and store data
            dataset_id = self.data_collector.process_and_store_data(data, name, data_type)
            
            self.logger.info(f"Successfully created dataset {dataset_id} with {len(data)} records")
            
            return dataset_id
        except Exception as e:
            self.logger.error(f"Error collecting data: {e}")
            return -1
    
    def enrich_dataset_with_ai(self, dataset_id: int, ai_tool: str = 'openai') -> bool:
        """
        Enrich dataset using AI tools
        
        Args:
            dataset_id: ID of the dataset to enrich
            ai_tool: AI tool to use
            
        Returns:
            Success flag
        """
        try:
            # Get the dataset
            dataset = self.db.get_dataset(dataset_id)
            
            if not dataset or not dataset['data']:
                self.logger.error(f"Dataset {dataset_id} not found or empty")
                return False
            
            # Enrich the dataset
            enriched_data = self.ai_integration.enrich_dataset(dataset['data'], ai_tool)
            
            # Store the enriched version
            version = self.db.add_dataset_version(dataset_id, enriched_data)
            
            self.logger.info(f"Enriched dataset {dataset_id}, created version {version}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error enriching dataset: {e}")
            return False
    
    def analyze_dataset(self, dataset_id: int) -> Dict[str, Any]:
        """
        Comprehensive analysis of a dataset
        
        Args:
            dataset_id: ID of the dataset
            
        Returns:
            Dictionary with analysis results
        """
        try:
            self.logger.info(f"Analyzing dataset {dataset_id}")
            
            # Get dataset summary
            summary = self.insights_engine.generate_dataset_summary(dataset_id)
            
            # Detect anomalies
            anomalies = self.insights_engine.detect_anomalies(dataset_id)
            
            # Generate marketing insights
            marketing_insights = self.insights_engine.generate_marketing_insights(dataset_id)
            
            # Combine all analyses
            analysis = {
                'dataset_id': dataset_id,
                'timestamp': datetime.now().isoformat(),
                'summary': summary,
                'anomalies': anomalies,
                'marketing_insights': marketing_insights
            }
            
            self.logger.info(f"Completed analysis for dataset {dataset_id}")
            
            return analysis
        except Exception as e:
            self.logger.error(f"Error analyzing dataset: {e}")
            return {'error': str(e)}
    
    def train_ml_model(self, dataset_id: int, target_column: str, 
                       model_type: str = 'classifier') -> Dict[str, Any]:
        """
        Train a machine learning model on a dataset
        
        Args:
            dataset_id: ID of the dataset
            target_column: Column to predict
            model_type: Type of model (classifier/regressor)
            
        Returns:
            Dictionary with model info
        """
        try:
            self.logger.info(f"Training {model_type} on dataset {dataset_id} for {target_column}")
            
            if model_type == 'classifier':
                model_info = self.ml_engine.train_classifier(dataset_id, target_column)
            elif model_type == 'regressor':
                model_info = self.ml_engine.train_regressor(dataset_id, target_column)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            self.logger.info(f"Successfully trained {model_type} {model_info.get('model_name')}")
            
            return model_info
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return {'error': str(e)}
    
    def make_predictions(self, model_name: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make predictions using a trained model
        
        Args:
            model_name: Name of the trained model
            data: Data to make predictions on
            
        Returns:
            Dictionary with predictions
        """
        try:
            self.logger.info(f"Making predictions with model {model_name}")
            
            predictions = self.ml_engine.predict(model_name, data)
            
            return {
                'model_name': model_name,
                'record_count': len(data),
                'predictions': predictions
            }
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            return {'error': str(e)}
    
    def save_analysis_to_file(self, analysis: Dict[str, Any], filename: str) -> bool:
        """
        Save analysis results to a JSON file
        
        Args:
            analysis: Analysis results
            filename: Filename to save to
            
        Returns:
            Success flag
        """
        try:
            with open(filename, 'w') as f:
                json.dump(analysis, f, indent=2)
            
            self.logger.info(f"Saved analysis to {filename}")
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving analysis: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize integration
    peso = PesoIntegration()
    
    # Collect and process data
    dataset_id = peso.collect_and_process_data(
        name="Non-Human Marketing Dataset",
        data_type="non-human",
        synthetic_count=500
    )
    
    # Analyze dataset
    analysis = peso.analyze_dataset(dataset_id)
    
    # Save analysis
    peso.save_analysis_to_file(analysis, "analysis_results.json")
    
    # Train ML model
    model_info = peso.train_ml_model(dataset_id, "industry", "classifier")
    
    print(f"Dataset ID: {dataset_id}")
    print(f"Model Name: {model_info.get('model_name')}")
