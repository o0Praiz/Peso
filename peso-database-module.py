import sqlite3
import json
from typing import Dict, Any, List
import os

class PesoDatabase:
    def __init__(self, db_path: str = 'peso_marketing.db'):
        """
        Initialize the Peso database with a SQLite backend
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._initialize_database()

    def _initialize_database(self):
        """
        Create initial database structure if it doesn't exist
        """
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Create base tables
            self.cursor.execute('''
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
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS dataset_versions (
                    dataset_id INTEGER,
                    version INTEGER,
                    data TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(dataset_id) REFERENCES marketing_datasets(id)
                )
            ''')
            
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")
        finally:
            if self.conn:
                self.conn.close()
                self.conn = None
                self.cursor = None

    def connect(self):
        """
        Establish a database connection
        """
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    def close(self):
        """
        Close the database connection
        """
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def add_dataset(self, name: str, data_type: str, source: str = None, metadata: Dict[str, Any] = None) -> int:
        """
        Add a new marketing dataset to the database
        
        Args:
            name (str): Name of the dataset
            data_type (str): Type of dataset (human/non-human/mixed)
            source (str, optional): Source of the dataset
            metadata (dict, optional): Additional metadata about the dataset
        
        Returns:
            int: ID of the newly created dataset
        """
        try:
            self.connect()
            
            # Convert metadata to JSON string
            metadata_json = json.dumps(metadata) if metadata else None
            
            self.cursor.execute('''
                INSERT INTO marketing_datasets 
                (name, type, source, metadata) 
                VALUES (?, ?, ?, ?)
            ''', (name, data_type, source, metadata_json))
            
            dataset_id = self.cursor.lastrowid
            self.conn.commit()
            return dataset_id
        except sqlite3.Error as e:
            print(f"Error adding dataset: {e}")
            return -1
        finally:
            self.close()

    def add_dataset_version(self, dataset_id: int, data: List[Dict[str, Any]]) -> int:
        """
        Add a new version of a dataset
        
        Args:
            dataset_id (int): ID of the dataset
            data (list): List of data entries
        
        Returns:
            int: Version number of the added dataset
        """
        try:
            self.connect()
            
            # Convert data to JSON string
            data_json = json.dumps(data)
            
            # Get the latest version
            self.cursor.execute('''
                SELECT COALESCE(MAX(version), 0) + 1 
                FROM dataset_versions 
                WHERE dataset_id = ?
            ''', (dataset_id,))
            version = self.cursor.fetchone()[0]
            
            self.cursor.execute('''
                INSERT INTO dataset_versions 
                (dataset_id, version, data) 
                VALUES (?, ?, ?)
            ''', (dataset_id, version, data_json))
            
            # Update last_updated in marketing_datasets
            self.cursor.execute('''
                UPDATE marketing_datasets 
                SET last_updated = CURRENT_TIMESTAMP 
                WHERE id = ?
            ''', (dataset_id,))
            
            self.conn.commit()
            return version
        except sqlite3.Error as e:
            print(f"Error adding dataset version: {e}")
            return -1
        finally:
            self.close()

    def get_dataset(self, dataset_id: int, version: int = None) -> Dict[str, Any]:
        """
        Retrieve a specific dataset or its specific version
        
        Args:
            dataset_id (int): ID of the dataset
            version (int, optional): Specific version to retrieve
        
        Returns:
            dict: Dataset information
        """
        try:
            self.connect()
            
            # Fetch dataset metadata
            self.cursor.execute('''
                SELECT * FROM marketing_datasets 
                WHERE id = ?
            ''', (dataset_id,))
            dataset_metadata = self.cursor.fetchone()
            
            if not dataset_metadata:
                return None
            
            # Fetch dataset version
            if version:
                self.cursor.execute('''
                    SELECT * FROM dataset_versions 
                    WHERE dataset_id = ? AND version = ?
                ''', (dataset_id, version))
            else:
                # Get the latest version
                self.cursor.execute('''
                    SELECT * FROM dataset_versions 
                    WHERE dataset_id = ? 
                    ORDER BY version DESC 
                    LIMIT 1
                ''', (dataset_id,))
            
            dataset_version = self.cursor.fetchone()
            
            return {
                'metadata': {
                    'id': dataset_metadata[0],
                    'name': dataset_metadata[1],
                    'type': dataset_metadata[2],
                    'source': dataset_metadata[3],
                    'created_at': dataset_metadata[4],
                    'last_updated': dataset_metadata[5],
                    'additional_metadata': json.loads(dataset_metadata[6]) if dataset_metadata[6] else None
                },
                'version': dataset_version[1] if dataset_version else None,
                'data': json.loads(dataset_version[2]) if dataset_version else None
            }
        except sqlite3.Error as e:
            print(f"Error retrieving dataset: {e}")
            return None
        finally:
            self.close()

# Example usage
if __name__ == "__main__":
    # Initialize the database
    db = PesoDatabase()
    
    # Add a sample dataset
    dataset_id = db.add_dataset(
        name="Global Marketing Contacts",
        data_type="mixed",
        source="Public Sources",
        metadata={
            "description": "Initial global marketing contact dataset",
            "collection_method": "Web Scraping"
        }
    )
    
    # Add a version to the dataset
    sample_data = [
        {"name": "John Doe", "country": "USA", "industry": "Tech"},
        {"name": "Jane Smith", "country": "UK", "industry": "Finance"}
    ]
    db.add_dataset_version(dataset_id, sample_data)
    
    # Retrieve the dataset
    retrieved_dataset = db.get_dataset(dataset_id)
    print(json.dumps(retrieved_dataset, indent=2))
