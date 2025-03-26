import sqlite3
import json
from typing import Dict, Any, List, Optional, Union
import logging

class QueryEngine:
    def __init__(self, db_instance):
        """
        Initialize QueryEngine with a database instance
        
        Args:
            db_instance: PesoDatabase instance
        """
        self.db = db_instance
        self.logger = logging.getLogger(__name__)
        
    def query_datasets(self, filters: Dict[str, Any] = None) -> List[Dict]:
        """
        Query datasets based on filters
        
        Args:
            filters: Dictionary of filter conditions
            
        Returns:
            List of datasets matching the criteria
        """
        try:
            self.db.connect()
            
            # Start with base query
            query = "SELECT * FROM marketing_datasets WHERE 1=1"
            params = []
            
            # Apply filters if provided
            if filters:
                if 'type' in filters:
                    query += " AND type = ?"
                    params.append(filters['type'])
                    
                if 'name_contains' in filters:
                    query += " AND name LIKE ?"
                    params.append(f"%{filters['name_contains']}%")
                    
                if 'created_after' in filters:
                    query += " AND created_at > ?"
                    params.append(filters['created_after'])
                    
                if 'created_before' in filters:
                    query += " AND created_at < ?"
                    params.append(filters['created_before'])
            
            # Execute query
            self.db.cursor.execute(query, params)
            rows = self.db.cursor.fetchall()
            
            # Convert to list of dictionaries
            results = []
            for row in rows:
                metadata = json.loads(row[6]) if row[6] else {}
                results.append({
                    'id': row[0],
                    'name': row[1],
                    'type': row[2],
                    'source': row[3],
                    'created_at': row[4],
                    'last_updated': row[5],
                    'metadata': metadata
                })
                
            return results
            
        except sqlite3.Error as e:
            self.logger.error(f"Database query error: {e}")
            return []
        finally:
            self.db.close()
    
    def search_dataset_contents(self, dataset_id: int, search_terms: Dict[str, Any], 
                                version: int = None) -> List[Dict]:
        """
        Search within a dataset's contents for specific terms
        
        Args:
            dataset_id: ID of the dataset to search
            search_terms: Key-value pairs to search for
            version: Optional specific version to search
            
        Returns:
            List of matching records
        """
        try:
            # Get the dataset
            dataset = self.db.get_dataset(dataset_id, version)
            
            if not dataset or not dataset['data']:
                return []
                
            # Search through the data
            results = []
            for record in dataset['data']:
                match = True
                for key, value in search_terms.items():
                    if key not in record or not self._matches(record[key], value):
                        match = False
                        break
                
                if match:
                    results.append(record)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Content search error: {e}")
            return []
    
    def _matches(self, field_value: Any, search_value: Any) -> bool:
        """
        Check if a field matches a search value
        
        Args:
            field_value: Value from the record
            search_value: Value to search for
            
        Returns:
            Boolean indicating if there's a match
        """
        # Handle string contains
        if isinstance(field_value, str) and isinstance(search_value, str):
            return search_value.lower() in field_value.lower()
        
        # Handle list contains
        if isinstance(field_value, list):
            return search_value in field_value
        
        # Handle direct equality
        return field_value == search_value
    
    def advanced_query(self, query_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an advanced query with complex conditions
        
        Args:
            query_spec: Dictionary with advanced query specifications
            
        Returns:
            Query results and metadata
        """
        results = {
            'datasets': [],
            'records': [],
            'metrics': {
                'total_datasets': 0,
                'total_records': 0,
                'query_time_ms': 0
            }
        }
        
        try:
            # Dataset filtering logic
            if 'dataset_filters' in query_spec:
                results['datasets'] = self.query_datasets(query_spec['dataset_filters'])
                results['metrics']['total_datasets'] = len(results['datasets'])
            
            # Record filtering logic
            if 'content_search' in query_spec and query_spec.get('dataset_id'):
                results['records'] = self.search_dataset_contents(
                    query_spec['dataset_id'],
                    query_spec['content_search'],
                    query_spec.get('version')
                )
                results['metrics']['total_records'] = len(results['records'])
            
            # Apply grouping if specified
            if 'group_by' in query_spec and results['records']:
                results['grouped_results'] = self._group_results(
                    results['records'], 
                    query_spec['group_by']
                )
            
            # Apply aggregations if specified
            if 'aggregations' in query_spec and results['records']:
                results['aggregations'] = self._calculate_aggregations(
                    results['records'],
                    query_spec['aggregations']
                )
                
            return results
            
        except Exception as e:
            self.logger.error(f"Advanced query error: {e}")
            return {
                'error': str(e),
                'datasets': [],
                'records': [],
                'metrics': {
                    'total_datasets': 0,
                    'total_records': 0
                }
            }
    
    def _group_results(self, records: List[Dict], group_by: str) -> Dict[str, List]:
        """
        Group results by a specific field
        
        Args:
            records: List of record dictionaries
            group_by: Field to group by
            
        Returns:
            Dictionary with groups
        """
        grouped = {}
        
        for record in records:
            if group_by in record:
                key = record[group_by]
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(record)
        
        return grouped
    
    def _calculate_aggregations(self, records: List[Dict], 
                               aggregations: List[Dict]) -> Dict[str, Any]:
        """
        Calculate aggregations on the result set
        
        Args:
            records: List of record dictionaries
            aggregations: List of aggregation specs
            
        Returns:
            Dictionary with aggregation results
        """
        results = {}
        
        for agg in aggregations:
            field = agg.get('field')
            func = agg.get('function')
            
            if not field or not func:
                continue
                
            values = [record.get(field) for record in records 
                      if field in record and isinstance(record[field], (int, float))]
            
            if func == 'avg' and values:
                results[f'avg_{field}'] = sum(values) / len(values)
            elif func == 'sum' and values:
                results[f'sum_{field}'] = sum(values)
            elif func == 'min' and values:
                results[f'min_{field}'] = min(values)
            elif func == 'max' and values:
                results[f'max_{field}'] = max(values)
            elif func == 'count':
                results[f'count_{field}'] = len(values)
        
        return results

# Example usage
if __name__ == "__main__":
    from peso-database-module import PesoDatabase
    
    # Initialize database
    db = PesoDatabase()
    
    # Initialize query engine
    query_engine = QueryEngine(db)
    
    # Simple dataset query
    non_human_datasets = query_engine.query_datasets({'type': 'non-human'})
    print(f"Found {len(non_human_datasets)} non-human datasets")
    
    # Advanced query example
    results = query_engine.advanced_query({
        'dataset_filters': {'type': 'non-human'},
        'dataset_id': 1,  # Assuming dataset with ID 1 exists
        'content_search': {'country': 'USA'},
        'group_by': 'industry',
        'aggregations': [
            {'field': 'id', 'function': 'count'},
            {'field': 'age', 'function': 'avg'}
        ]
    })
    
    print(json.dumps(results, indent=2))
