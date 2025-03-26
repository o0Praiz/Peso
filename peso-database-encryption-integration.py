#!/usr/bin/env python3
"""
Integration example for the Peso Database Encryption Module
Shows how to integrate encryption with the existing database module
"""

import json
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import the database modules
from peso_database_module import PesoDatabase
from peso_database_encryption import DatabaseEncryption, EncryptedDatabaseWrapper

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_encrypted_database(config_path: str = "config.json") -> EncryptedDatabaseWrapper:
    """
    Set up an encrypted database instance
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Encrypted database wrapper
    """
    # Load configuration
    config = load_config(config_path)
    
    # Create database instance
    db_path = config.get('database', {}).get('path', 'peso_marketing.db')
    db = PesoDatabase(db_path)
    
    # Initialize encryption
    encryption_enabled = config.get('security', {}).get('enable_data_encryption', False)
    
    if not encryption_enabled:
        logger.info("Data encryption is disabled. Using standard database.")
        return db
    
    # Get encryption key file path
    key_file = config.get('security', {}).get('encryption_key_file')
    
    # Get list of sensitive fields to encrypt
    sensitive_fields = config.get('security', {}).get('sensitive_fields', [
        'email', 'phone', 'address', 'credit_card',
        'ssn', 'password', 'api_key', 'access_token'
    ])
    
    # Initialize encryption with configuration
    encryption = DatabaseEncryption(config=config)
    
    # Create encrypted wrapper
    logger.info("Data encryption is enabled. Using encrypted database wrapper.")
    return EncryptedDatabaseWrapper(db, encryption, sensitive_fields)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    default_config = {
        "database": {
            "path": "peso_marketing.db"
        },
        "security": {
            "enable_data_encryption": False,
            "encryption_key_file": "security/encryption.key",
            "sensitive_fields": [
                "email", "phone", "address", "credit_card", 
                "ssn", "password", "api_key", "access_token"
            ]
        }
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        else:
            logger.warning(f"Configuration file {config_path} not found, using defaults")
            return default_config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return default_config

def update_config_for_encryption(config_path: str, enable: bool = True) -> bool:
    """
    Update configuration to enable/disable encryption
    
    Args:
        config_path: Path to configuration file
        enable: Whether to enable or disable encryption
    
    Returns:
        Success flag
    """
    try:
        # Load existing config
        config = load_config(config_path)
        
        # Update encryption setting
        if 'security' not in config:
            config['security'] = {}
        
        config['security']['enable_data_encryption'] = enable
        
        # Ensure encryption key file is set
        if enable and 'encryption_key_file' not in config['security']:
            config['security']['encryption_key_file'] = "security/encryption.key"
        
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Updated configuration in {config_path}")
        return True
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return False

def migrate_to_encrypted_database(config_path: str) -> bool:
    """
    Migrate an existing database to use encryption
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Success flag
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Get database path
        db_path = config.get('database', {}).get('path', 'peso_marketing.db')
        
        # Create standard database instance
        db = PesoDatabase(db_path)
        
        # Get list of sensitive fields to encrypt
        sensitive_fields = config.get('security', {}).get('sensitive_fields', [
            'email', 'phone', 'address', 'credit_card',
            'ssn', 'password', 'api_key', 'access_token'
        ])
        
        # Initialize encryption
        encryption = DatabaseEncryption(config=config)
        
        # Get all datasets
        dataset_ids = get_all_dataset_ids(db)
        
        # Process each dataset
        for dataset_id in dataset_ids:
            # Get dataset
            dataset = db.get_dataset(dataset_id)
            
            if not dataset:
                continue
            
            # Encrypt metadata if present
            if dataset.get('metadata') and isinstance(dataset['metadata'], dict):
                encrypted_metadata = encryption.encrypt_dict(dataset['metadata'], sensitive_fields)
                # Update metadata in database
                db.update_dataset_metadata(dataset_id, encrypted_metadata)
            
            # Get all versions
            versions = get_dataset_versions(db, dataset_id)
            
            # Process each version
            for version in versions:
                # Get version data
                version_data = db.get_dataset(dataset_id, version)
                
                if not version_data or not version_data.get('data'):
                    continue
                
                # Encrypt data
                encrypted_data = []
                for record in version_data['data']:
                    encrypted_record = encryption.encrypt_dict(record, sensitive_fields)
                    encrypted_data.append(encrypted_record)
                
                # Update version data in database
                db.update_dataset_version(dataset_id, version, encrypted_data)
        
        # Enable encryption in config
        update_config_for_encryption(config_path, True)
        
        logger.info("Database migration to encrypted format complete")
        return True
    except Exception as e:
        logger.error(f"Error migrating database: {e}")
        return False

def get_all_dataset_ids(db: PesoDatabase) -> List[int]:
    """
    Get all dataset IDs from the database
    
    Args:
        db: Database instance
    
    Returns:
        List of dataset IDs
    """
    # This is a simplified implementation
    # In a real database, you would query all dataset IDs
    # For this example, we'll return a placeholder list
    return [1, 2, 3]

def get_dataset_versions(db: PesoDatabase, dataset_id: int) -> List[int]:
    """
    Get all versions for a dataset
    
    Args:
        db: Database instance
        dataset_id: Dataset ID
    
    Returns:
        List of version numbers
    """
    # This is a simplified implementation
    # In a real database, you would query all versions for a dataset
    # For this example, we'll return a placeholder list
    return [1]

# Example usage
if __name__ == "__main__":
    # Check if encryption is enabled
    config = load_config("config.json")
    encryption_enabled = config.get('security', {}).get('enable_data_encryption', False)
    
    if not encryption_enabled:
        # Prompt user to enable encryption
        print("Database encryption is not enabled. Would you like to enable it? (y/n)")
        choice = input().lower()
        
        if choice == 'y' or choice == 'yes':
            print("Migrating database to encrypted format...")
            success = migrate_to_encrypted_database("config.json")
            
            if success:
                print("Database encryption enabled successfully!")
            else:
                print("Failed to enable database encryption.")
                exit(1)
    
    # Set up encrypted database
    db = setup_encrypted_database("config.json")
    
    # Example usage of encrypted database
    print("\nAdding test dataset with sensitive information...")
    
    # Sample data with sensitive information
    sample_data = [
        {
            'name': 'John Doe',
            'email': 'john@example.com',
            'phone': '555-123-4567',
            'country': 'USA'
        },
        {
            'name': 'Jane Smith',
            'email': 'jane@example.com',
            'phone': '555-987-6543',
            'credit_card': '4111-1111-1111-1111',
            'country': 'UK'
        }
    ]
    
    # Add dataset with encryption
    dataset_id = db.add_dataset(
        name="Encrypted Test Dataset",
        data_type="test",
        metadata={'description': 'Contains sensitive information'}
    )
    
    # Add version with encrypted data
    db.add_dataset_version(dataset_id, sample_data)
    
    print(f"Added dataset with ID: {dataset_id}")
    
    # Retrieve and display the dataset
    print("\nRetrieving dataset (sensitive fields are automatically decrypted):")
    dataset = db.get_dataset(dataset_id)
    
    # Print dataset excluding actual data for brevity
    dataset_info = {
        'id': dataset['metadata']['id'],
        'name': dataset['metadata']['name'],
        'type': dataset['metadata']['type'],
        'record_count': len(dataset['data'])
    }
    
    print(json.dumps(dataset_info, indent=2))
    
    # Print first record as example
    if dataset['data']:
        print("\nFirst record example:")
        print(json.dumps(dataset['data'][0], indent=2))
