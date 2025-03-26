#!/usr/bin/env python3
"""
Peso API Server Launcher
Simple script to run the Peso API server
"""

import os
import argparse
import logging
from peso_api_module import run_api_server

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run the Peso API server")
    
    parser.add_argument(
        "--host", 
        default=os.environ.get("PESO_API_HOST", "127.0.0.1"),
        help="Host to bind the server to"
    )
    
    parser.add_argument(
        "--port", 
        type=int,
        default=int(os.environ.get("PESO_API_PORT", "8000")),
        help="Port to bind the server to"
    )
    
    parser.add_argument(
        "--config",
        default=os.environ.get("PESO_CONFIG", "config.json"),
        help="Path to the configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=os.environ.get("PESO_LOG_LEVEL", "INFO"),
        help="Logging level"
    )
    
    return parser.parse_args()

def setup_logging(log_level):
    """Configure logging"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Set environment variables
    os.environ["PESO_CONFIG"] = args.config
    
    # Print startup info
    print(f"Starting Peso API server at http://{args.host}:{args.port}")
    print(f"Using configuration file: {args.config}")
    print(f"Log level: {args.log_level}")
    print("Press Ctrl+C to stop the server")
    
    # Run the server
    run_api_server(args.host, args.port)

if __name__ == "__main__":
    main()
