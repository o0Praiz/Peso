#!/usr/bin/env python3
"""
Peso Project Initialization Script
Creates the necessary directory structure and initial files
"""

import os
import json
import shutil
import argparse
from datetime import datetime


def create_directory_structure(base_dir="."):
    """Create the project directory structure"""
    directories = [
        "database",
        "data_collection",
        "ai_integration",
        "query",
        "insights",
        "ml",
        "integration",
        "models",
        "logs",
        "backups",
        "insights_reports"
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"  Created: {dir_path}")
        else:
            print(f"  Already exists: {dir_path}")


def create_init_files(base_dir="."):
    """Create __init__.py files in Python module directories"""
    module_dirs = [
        "database",
        "data_collection",
        "ai_integration",
        "query",
        "insights",
        "ml",
        "integration"
    ]
    
    print("Creating Python module __init__.py files...")
    for directory in module_dirs:
        init_file = os.path.join(base_dir, directory, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f"""\"\"\"
Peso {directory.replace('_', ' ').title()} Module
\"\"\"

__version__ = "0.5.0"
""")
            print(f"  Created: {init_file}")
        else:
            print(f"  Already exists: {init_file}")


def copy_config_example(base_dir="."):
    """Copy the example config file if config.json doesn't exist"""
    example_path = os.path.join(base_dir, "config.example.json")
    config_path = os.path.join(base_dir, "config.json")
    
    if os.path.exists(example_path) and not os.path.exists(config_path):
        shutil.copy(example_path, config_path)
        print(f"Created default config.json from example")
    elif not os.path.exists(config_path):
        # Create a minimal default config if the example doesn't exist
        default_config = {
            "database": {
                "path": "peso_marketing.db"
            },
            "ml": {
                "models_dir": "models"
            },
            "logging": {
                "level": "INFO",
                "file": "logs/peso.log"
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"Created minimal default config.json")
    else:
        print("config.json already exists")


def create_initial_database(base_dir="."):
    """Create an empty initial database file"""
    try:
        import sqlite3
        
        db_path = os.path.join(base_dir, "peso_marketing.db")
        
        if not os.path.exists(db_path):
            # Create basic database structure
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create base tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS marketing_datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    source TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            # Create a table for tracking dataset versions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    dataset_id INTEGER,
                    version INTEGER,
                    data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(dataset_id) REFERENCES marketing_datasets(id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            print(f"Created initial empty database at {db_path}")
        else:
            print(f"Database already exists at {db_path}")
            
    except ImportError:
        print("SQLite3 module not available. Skipping database creation.")
    except Exception as e:
        print(f"Error creating database: {e}")


def main():
    parser = argparse.ArgumentParser(description="Initialize the Peso project structure")
    parser.add_argument("--dir", default=".", help="Base directory for the project")
    args = parser.parse_args()
    
    print("=== Peso Project Initialization ===")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Base directory: {os.path.abspath(args.dir)}")
    print()
    
    create_directory_structure(args.dir)
    print()
    
    create_init_files(args.dir)
    print()
    
    copy_config_example(args.dir)
    print()
    
    create_initial_database(args.dir)
    print()
    
    print("Initialization complete!")
    print("Next steps:")
    print("1. Review and update config.json with your settings")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run the application: python main.py")


if __name__ == "__main__":
    main()
