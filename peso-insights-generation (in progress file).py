import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from scipy import stats

class InsightsEngine:
    def __init__(self, db_instance, query_engine=None):
        """
        Initialize insights engine
        
        Args:
            db_instance: PesoDatabase instance
            query_engine: QueryEngine instance (optional)
        """
        self.db = db_instance
        self.query_engine = query_engine
        self.logger = logging.getLogger(__name__)
        
    def generate_dataset_summary(self, dataset_id: int, version: int = None) -> Dict[str, Any]:
        """
        Generate summary statistics and insights for a dataset
        
        Args:
            dataset_id: ID of the dataset
            version: Specific version to analyze
            
        Returns:
            Dictionary with summary insights
        """
        try:
            # Get the dataset
            dataset = self.db.get_dataset(dataset_id, version)
            
            if not dataset or not dataset['data']:
                return {'error': f"Dataset with ID {dataset_id} not found or empty"}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(dataset['data'])
            
            # Generate basic statistics
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Summary insights
            summary = {
                'dataset_info': {
                    'id': dataset_id,
                    'name': dataset['metadata']['name'],
                    'type': dataset['metadata']['type'],
                    'version': dataset['version'],
                    'record_count': len(df)
                },
                'structure': {
                    'columns': list(df.columns),
                    'numeric_columns': numeric_columns,
                    'categorical_columns': categorical_columns,
                    'missing_values': df.isnull().sum().to_dict()
                },
                'numeric_statistics': {},
                'categorical_statistics': {},
                'insights': []
            }
            
            # Calculate numeric statistics
            for col in numeric_columns:
                summary['numeric_statistics'][col] = {
                    'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    'median': float(df[col].median()) if not pd.isna(df[col].median()) else None,
                    'std': float(df[col].std()) if not pd.isna(df[col].std()) else None
                }
            
            # Calculate categorical statistics
            for col in categorical_columns:
                value_counts = df[col].value_counts().to_dict()
                top_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                
                summary['categorical_statistics'][col] = {
                    'unique_values': df[col].nunique(),
                    'top_values': dict(top_values)
                }
            
            # Generate insights
            insights = []
            
            # Check for data completeness
            missing_percentages = (df.isnull().sum() / len(df) * 100).to_dict()
            columns_high_missing = [col for col, pct in missing_percentages.items() if pct > 20]
            
            if columns_high_missing:
                insights.append({
                    'type': 'data_quality',
                    'severity': 'high',
                    'message': f"High percentage of missing values in columns: {', '.join(columns_high_missing)}",
                    'recommendation': "Consider data imputation or excluding these columns from analysis"
                })
            
            # Check for distribution skewness in numeric columns
            for col in numeric_columns:
                if df[col].count() > 10:  # Only check if we have enough data
                    skewness = stats.skew(df[col].dropna())
                    if abs(skewness) > 1:
                        direction = "right" if skewness > 0 else "left"
                        insights.append({
                            'type': 'distribution',
                            'severity': 'medium',
                            'message': f"Column '{col}' has a {direction}-skewed distribution (skewness: {skewness:.2f})",
                            'recommendation': "Consider data transformation before machine learning"
                        })
            
            # Check for potential category imbalance
            for col in categorical_columns:
                if df[col].nunique() > 1:  # Only check if we have multiple categories
                    value_counts = df[col].value_counts()
                    max_count = value_counts.max()
                    min_count = value_counts.min()
                    
                    if max_count > min_count * 10:  # 10:1 ratio or more
                        insights.append({
                            'type': 'class_imbalance',
                            'severity': 'medium',
                            'message': f"Severe imbalance detected in column '{col}'",
                            'recommendation': "Consider balancing techniques for machine learning"
                        })
            
            # Add insights to summary
            summary['insights'] = insights
            
            return summary
        
        except Exception as e:
            self.logger.error(f"Error generating dataset summary: {e}")
            return {'error': str(e)}
    
    def detect_anomalies(self, dataset_id: int, columns: List[str] = None, 
                         threshold: float = 3.0, version: int = None) -> Dict[str, Any]:
        """
        Detect anomalies in dataset columns using z-score
        
        Args:
            dataset_id: ID of the dataset
            columns: Specific columns to check (if None, checks all numeric)
            threshold: Z-score threshold for anomaly detection
            version: Specific version to analyze
            
        Returns:
            Dictionary with anomaly insights
        """
        try:
            # Get the dataset
            dataset = self.db.get_dataset(dataset_id, version)
            
            if not dataset or not dataset['data']:
                return {'error': f"Dataset with ID {dataset_id} not found or empty"}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(dataset['data'])
            
            # Determine columns to analyze
            if columns is None:
                columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            else:
                # Filter to only include columns that exist and are numeric
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                columns = [col for col in columns if col in numeric_columns]
            
            if not columns:
                return {'error': "No numeric columns available for anomaly detection"}
            
            # Detect anomalies for each column
            anomalies = {}
            for col in columns:
                column_data = df[col].dropna()
                
                if len(column_data) < 10:
                    continue  # Skip columns with too little data
                
                # Calculate z-scores
                z_scores = np.abs(stats.zscore(column_data))
                
                # Find anomalies
                anomaly_indices = np.where(z_scores > threshold)[0]
                anomaly_records = []
                
                for idx in anomaly_indices:
                    record_idx = column_data.index[idx]
                    record = df.loc[record_idx].to_dict()
                    record['z_score'] = float(z_scores[idx])
                    anomaly_records.append(record)
                
                if anomaly_records:
                    anomalies[col] = {
                        'count': len(