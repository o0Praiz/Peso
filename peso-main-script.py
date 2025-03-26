#!/usr/bin/env python3
"""
Peso: Non-Human Marketing Data Warehouse
Main script to run the Peso application
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

# Import the integration module
from peso_integration_module import PesoIntegration

def setup_logging():
    """Configure logging for the application"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"peso_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("peso-main")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Peso: Non-Human Marketing Data Warehouse")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Dataset commands
    dataset_parser = subparsers.add_parser("dataset", help="Dataset operations")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command")
    
    # Create dataset
    create_parser = dataset_subparsers.add_parser("create", help="Create a new dataset")
    create_parser.add_argument("--name", required=True, help="Dataset name")
    create_parser.add_argument("--type", required=True, choices=["human", "non-human", "mixed"], 
                              help="Dataset type")
    create_parser.add_argument("--count", type=int, default=100, help="Number of synthetic records")
    
    # Analyze dataset
    analyze_parser = dataset_subparsers.add_parser("analyze", help="Analyze a dataset")
    analyze_parser.add_argument("--id", type=int, required=True, help="Dataset ID")
    analyze_parser.add_argument("--output", help="Output file for analysis results")
    
    # List datasets
    list_parser = dataset_subparsers.add_parser("list", help="List datasets")
    
    # Enrich dataset
    enrich_parser = dataset_subparsers.add_parser("enrich", help="Enrich a dataset using AI")
    enrich_parser.add_argument("--id", type=int, required=True, help="Dataset ID")
    enrich_parser.add_argument("--tool", default="openai", choices=["openai", "anthropic"],
                             help="AI tool to use for enrichment")
    
    # ML commands
    ml_parser = subparsers.add_parser("ml", help="Machine learning operations")
    ml_subparsers = ml_parser.add_subparsers(dest="ml_command")
    
    # Train model
    train_parser = ml_subparsers.add_parser("train", help="Train a machine learning model")
    train_parser.add_argument("--dataset", type=int, required=True, help="Dataset ID")
    train_parser.add_argument("--target", required=True, help="Target column to predict")
    train_parser.add_argument("--type", default="classifier", choices=["classifier", "regressor"],
                            help="Type of model to train")
    
    # Predict
    predict_parser = ml_subparsers.add_parser("predict", help="Make predictions with a model")
    predict_parser.add_argument("--model", required=True, help="Model name")
    predict_parser.add_argument("--data", required=True, help="JSON file with data for prediction")
    
    # Query commands
    query_parser = subparsers.add_parser("query", help="Query operations")
    query_parser.add_argument("--dataset", type=int, help="Dataset ID")
    query_parser.add_argument("--filter", help="JSON file with filter criteria")
    query_parser.add_argument("--output", help="Output file for query results")
    
    # Version commands
    version_parser = subparsers.add_parser("version", help="Version information")
    
    return parser.parse_args()

def main():
    """Main entry point for the application"""
    logger = setup_logging()
    args = parse_arguments()
    
    try:
        # Initialize the integration module
        peso = PesoIntegration()
        
        if args.command == "dataset":
            handle_dataset_commands(peso, args, logger)
        elif args.command == "ml":
            handle_ml_commands(peso, args, logger)
        elif args.command == "query":
            handle_query_commands(peso, args, logger)
        elif args.command == "version":
            print_version_info()
        else:
            logger.error(f"Unknown command: {args.command}")
            return 1
        
        return 0
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return 1

def handle_dataset_commands(peso, args, logger):
    """Handle dataset-related commands"""
    if args.dataset_command == "create":
        dataset_id = peso.collect_and_process_data(
            name=args.name,
            data_type=args.type,
            synthetic_count=args.count
        )
        if dataset_id > 0:
            print(f"Created dataset with ID: {dataset_id}")
        else:
            print("Failed to create dataset")
    
    elif args.dataset_command == "analyze":
        analysis = peso.analyze_dataset(args.id)
        
        if "error" in analysis:
            print(f"Error: {analysis['error']}")
        else:
            if args.output:
                peso.save_analysis_to_file(analysis, args.output)
                print(f"Analysis saved to {args.output}")
            else:
                print(json.dumps(analysis, indent=2))
    
    elif args.dataset_command == "list":
        datasets = peso.query_engine.query_datasets()
        print(f"Found {len(datasets)} datasets:")
        for ds in datasets:
            print(f"ID: {ds['id']}, Name: {ds['name']}, Type: {ds['type']}, Records: ?")
    
    elif args.dataset_command == "enrich":
        success = peso.enrich_dataset_with_ai(args.id, args.tool)
        if success:
            print(f"Successfully enriched dataset {args.id}")
        else:
            print(f"Failed to enrich dataset {args.id}")
    
    else:
        logger.error(f"Unknown dataset command: {args.dataset_command}")

def handle_ml_commands(peso, args, logger):
    """Handle machine learning commands"""
    if args.ml_command == "train":
        model_info = peso.train_ml_model(args.dataset, args.target, args.type)
        
        if "error" in model_info:
            print(f"Error: {model_info['error']}")
        else:
            print(f"Trained model: {model_info['model_name']}")
            print(f"Metrics: {json.dumps(model_info['metrics'], indent=2)}")
    
    elif args.ml_command == "predict":
        try:
            with open(args.data, 'r') as f:
                data = json.load(f)
            
            predictions = peso.make_predictions(args.model, data)
            
            if "error" in predictions:
                print(f"Error: {predictions['error']}")
            else:
                print(f"Made {len(predictions['predictions'])} predictions:")
                for i, pred in enumerate(predictions['predictions']):
                    print(f"  {i+1}: {pred}")
        except Exception as e:
            logger.error(f"Error loading prediction data: {e}")
    
    else:
        logger.error(f"Unknown ML command: {args.ml_command}")

def handle_query_commands(peso, args, logger):
    """Handle query commands"""
    filters = {}
    
    if args.filter:
        try:
            with open(args.filter, 'r') as f:
                filters = json.load(f)
        except Exception as e:
            logger.error(f"Error loading filter criteria: {e}")
            return
    
    if args.dataset:
        # Query within a specific dataset
        if filters:
            results = peso.query_engine.search_dataset_contents(args.dataset, filters)
        else:
            dataset = peso.db.get_dataset(args.dataset)
            results = dataset['data'] if dataset and 'data' in dataset else []
        
        print(f"Found {len(results)} records in dataset {args.dataset}")
    else:
        # Query datasets
        results = peso.query_engine.query_datasets(filters)
        print(f"Found {len(results)} datasets")
    
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    else:
        print(json.dumps(results[:5], indent=2))
        if len(results) > 5:
            print(f"... and {len(results) - 5} more results (use --output to save all)")

def print_version_info():
    """Print version information"""
    print("Peso: Non-Human Marketing Data Warehouse")
    print("Version: 0.5.0")
    print("Copyright 2025")

if __name__ == "__main__":
    sys.exit(main())