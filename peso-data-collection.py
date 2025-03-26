import requests
import json
from typing import List, Dict, Any
import logging
from datetime import datetime
import random

class DataCollector:
    def __init__(self, base_db):
        """
        Initialize data collector with database connection
        
        Args:
            base_db: PesoDatabase instance
        """
        self.db = base_db
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def collect_public_data(self, sources: List[str]) -> List[Dict[str, Any]]:
        """
        Collect data from public sources
        
        Args:
            sources (list): List of public data source URLs
        
        Returns:
            list: Collected data entries
        """
        collected_data = []
        
        for source in sources:
            try:
                response = requests.get(source, timeout=10)
                response.raise_for_status()
                
                # Basic data extraction (mock implementation)
                data = response.json() if response.headers.get('content-type') == 'application/json' else []
                
                self.logger.info(f"Collected {len(data)} entries from {source}")
                collected_data.extend(data)
            except Exception as e:
                self.logger.error(f"Error collecting data from {source}: {e}")
        
        return collected_data

    def generate_synthetic_data(self, data_type: str, count: int = 100) -> List[Dict[str, Any]]:
        """
        Generate synthetic marketing data
        
        Args:
            data_type (str): Type of data to generate
            count (int): Number of data entries to generate
        
        Returns:
            list: Synthetic data entries
        """
        synthetic_data = []
        
        industries = ['Tech', 'Finance', 'Healthcare', 'Retail', 'Education']
        countries = ['USA', 'UK', 'Germany', 'Japan', 'Canada', 'Australia']
        
        for _ in range(count):
            entry = {
                'id': random.randint(1000, 9999),
                'name': f"{random.choice(['John', 'Jane', 'Alex'])} {random.choice(['Smith', 'Doe', 'Johnson'])}",
                'industry': random.choice(industries),
                'country': random.choice(countries),
                'email': f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=5))}@example.com",
                'data_type': data_type,
                'timestamp': datetime.now().isoformat()
            }
            synthetic_data.append(entry)
        
        return synthetic_data

    def process_and_store_data(self, data: List[Dict[str, Any]], dataset_name: str, data_type: str):
        """
        Process collected data and store in database
        
        Args:
            data (list): Collected data entries
            dataset_name (str): Name for the dataset
            data_type (str): Type of dataset
        
        Returns:
            int: Dataset ID
        """
        # Clean and validate data
        cleaned_data = [
            {k: v for k, v in entry.items() if v is not None}
            for entry in data
        ]
        
        # Add dataset to database
        dataset_id = self.db.add_dataset(
            name=dataset_name,
            data_type=data_type,
            metadata={
                'collection_timestamp': datetime.now().isoformat(),
                'entry_count': len(cleaned_data)
            }
        )
        
        # Store dataset version
        self.db.add_dataset_version(dataset_id, cleaned_data)
        
        self.logger.info(f"Stored {len(cleaned_data)} entries for dataset: {dataset_name}")
        
        return dataset_id

# Example usage
if __name__ == "__main__":
    # Initialize database and data collector
    from base import PesoDatabase
    
    db = PesoDatabase()
    collector = DataCollector(db)
    
    # Collect synthetic data
    human_data = collector.generate_synthetic_data('human', 50)
    non_human_data = collector.generate_synthetic_data('non-human', 50)
    
    # Process and store datasets
    human_dataset_id = collector.process_and_store_data(
        human_data, 
        'Synthetic Human Marketing Contacts', 
        'human'
    )
    
    non_human_dataset_id = collector.process_and_store_data(
        non_human_data, 
        'Synthetic Non-Human Marketing Targets', 
        'non-human'
    )
