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
                        'count': len(anomaly_records),
                        'threshold': threshold,
                        'records': anomaly_records
                    }
            
            return {
                'dataset_id': dataset_id,
                'version': dataset['version'],
                'anomaly_count': sum(len(details['records']) for details in anomalies.values()),
                'columns_analyzed': columns,
                'anomalies': anomalies
            }
        
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return {'error': str(e)}
    
    def generate_marketing_insights(self, dataset_id: int, version: int = None) -> Dict[str, Any]:
        """
        Generate specific marketing-related insights from a dataset
        
        Args:
            dataset_id: ID of the dataset
            version: Specific version to analyze
            
        Returns:
            Dictionary with marketing insights
        """
        try:
            # Get the dataset
            dataset = self.db.get_dataset(dataset_id, version)
            
            if not dataset or not dataset['data']:
                return {'error': f"Dataset with ID {dataset_id} not found or empty"}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(dataset['data'])
            
            insights = {
                'dataset_id': dataset_id,
                'version': dataset['version'],
                'timestamp': datetime.now().isoformat(),
                'segments': [],
                'trends': [],
                'recommendations': []
            }
            
            # Analyze data type (human vs non-human)
            data_type = dataset['metadata'].get('type', 'unknown')
            
            # Segment analysis
            if 'country' in df.columns:
                country_distribution = df['country'].value_counts().to_dict()
                top_countries = sorted(country_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
                
                insights['segments'].append({
                    'type': 'geographic',
                    'distribution': dict(top_countries),
                    'insight': f"Top geographic segments: {', '.join([c[0] for c in top_countries])}"
                })
            
            if 'industry' in df.columns:
                industry_distribution = df['industry'].value_counts().to_dict()
                top_industries = sorted(industry_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
                
                insights['segments'].append({
                    'type': 'industry',
                    'distribution': dict(top_industries),
                    'insight': f"Top industry segments: {', '.join([i[0] for i in top_industries])}"
                })
            
            # Find potential cross-correlations
            if 'country' in df.columns and 'industry' in df.columns:
                # Get cross-tabulation
                cross_tab = pd.crosstab(df['country'], df['industry'])
                
                # Find top country-industry combinations
                # Flatten the cross-tab into a series
                flat_crosstab = cross_tab.unstack()
                top_combinations = flat_crosstab.nlargest(3)
                
                combination_insights = []
                for (country, industry), count in top_combinations.items():
                    combination_insights.append(f"{country}-{industry} ({count} records)")
                
                insights['segments'].append({
                    'type': 'cross_segment',
                    'insight': f"Top country-industry combinations: {', '.join(combination_insights)}"
                })
            
            # Trend analysis
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['date'] = df['timestamp'].dt.date
                    
                    # Time-based distribution
                    date_counts = df.groupby('date').size()
                    
                    # Check for growth trend
                    if len(date_counts) > 2:
                        growth_rate = (date_counts.iloc[-1] - date_counts.iloc[0]) / date_counts.iloc[0] * 100
                        
                        trend_direction = "upward" if growth_rate > 10 else "downward" if growth_rate < -10 else "stable"
                        
                        insights['trends'].append({
                            'type': 'growth',
                            'rate': float(growth_rate),
                            'insight': f"{trend_direction.capitalize()} trend detected ({growth_rate:.1f}% change)"
                        })
                except Exception as te:
                    self.logger.warning(f"Error in trend analysis: {te}")
            
            # Generate recommendations based on data type
            if data_type == 'non-human':
                insights['recommendations'].append({
                    'type': 'targeting',
                    'recommendation': "Focus on industry-specific marketing approaches based on segment distribution"
                })
                
                # Check for geographic concentration
                if 'country' in df.columns:
                    top_country_pct = top_countries[0][1] / len(df) * 100 if top_countries else 0
                    
                    if top_country_pct > 50:
                        insights['recommendations'].append({
                            'type': 'geographic_focus',
                            'recommendation': f"High concentration ({top_country_pct:.1f}%) in {top_countries[0][0]}; consider specialized regional approach"
                        })
                    else:
                        insights['recommendations'].append({
                            'type': 'geographic_diversity',
                            'recommendation': "Geographic diversity detected; develop multi-regional marketing strategy"
                        })
            
            return insights
        
        except Exception as e:
            self.logger.error(f"Error generating marketing insights: {e}")
            return {'error': str(e)}
    
    def generate_trend_report(self, dataset_id: int, time_column: str, 
                             value_columns: List[str], version: int = None) -> Dict[str, Any]:
        """
        Generate trend analysis report for time-series data
        
        Args:
            dataset_id: ID of the dataset
            time_column: Column containing timestamp data
            value_columns: Columns to analyze for trends
            version: Specific version to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        try:
            # Get the dataset
            dataset = self.db.get_dataset(dataset_id, version)
            
            if not dataset or not dataset['data']:
                return {'error': f"Dataset with ID {dataset_id} not found or empty"}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(dataset['data'])
            
            if time_column not in df.columns:
                return {'error': f"Time column '{time_column}' not found in dataset"}
            
            # Check that all value columns exist
            missing_columns = [col for col in value_columns if col not in df.columns]
            if missing_columns:
                return {'error': f"Columns not found in dataset: {missing_columns}"}
            
            # Convert time column to datetime
            try:
                df[time_column] = pd.to_datetime(df[time_column])
                df = df.sort_values(by=time_column)
            except Exception as e:
                return {'error': f"Error converting time column: {e}"}
            
            # Analyze trends for each value column
            trends = {}
            for column in value_columns:
                # Basic stats
                stats = {
                    'mean': float(df[column].mean()),
                    'min': float(df[column].min()),
                    'max': float(df[column].max()),
                    'start_value': float(df[column].iloc[0]),
                    'end_value': float(df[column].iloc[-1]),
                    'change_pct': float((df[column].iloc[-1] - df[column].iloc[0]) / df[column].iloc[0] * 100) if df[column].iloc[0] != 0 else None
                }
                
                # Resample to regular time intervals if needed
                if len(df) > 5:
                    # Determine appropriate time frequency (day, week, month)
                    time_range = (df[time_column].max() - df[time_column].min()).days
                    
                    if time_range > 365:
                        freq = 'M'  # Monthly
                        freq_name = 'monthly'
                    elif time_range > 60:
                        freq = 'W'  # Weekly
                        freq_name = 'weekly' 
                    else:
                        freq = 'D'  # Daily
                        freq_name = 'daily'
                    
                    # Set time column as index for resampling
                    df_resampled = df.set_index(time_column)
                    
                    # Resample and get mean values
                    resampled = df_resampled[column].resample(freq).mean()
                    
                    # Calculate linear regression for trend direction
                    x = np.arange(len(resampled))
                    y = resampled.values
                    
                    # Handle potential NaN values
                    mask = ~np.isnan(y)
                    if np.sum(mask) > 1:  # Need at least 2 points for regression
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])
                        
                        # Determine trend direction and strength
                        trend_direction = "increasing" if slope > 0 else "decreasing"
                        trend_strength = abs(r_value)
                        
                        # Classify trend strength
                        if trend_strength > 0.7:
                            strength_desc = "strong"
                        elif trend_strength > 0.4:
                            strength_desc = "moderate"
                        else:
                            strength_desc = "weak"
                        
                        trends[column] = {
                            'statistics': stats,
                            'trend': {
                                'direction': trend_direction,
                                'strength': strength_desc,
                                'r_squared': float(r_value ** 2),
                                'slope': float(slope),
                                'p_value': float(p_value),
                                'frequency': freq_name
                            }
                        }
                    else:
                        trends[column] = {
                            'statistics': stats,
                            'trend': {
                                'direction': "insufficient data",
                                'strength': "unknown"
                            }
                        }
                else:
                    # Not enough data for trend analysis
                    trends[column] = {
                        'statistics': stats,
                        'trend': {
                            'direction': "insufficient data",
                            'strength': "unknown"
                        }
                    }
            
            # Generate overall summary and insights
            report = {
                'dataset_id': dataset_id,
                'version': dataset['version'],
                'time_range': {
                    'start': df[time_column].min().isoformat(),
                    'end': df[time_column].max().isoformat(),
                    'duration_days': (df[time_column].max() - df[time_column].min()).days
                },
                'trends': trends,
                'insights': []
            }
            
            # Add key insights based on trend analysis
            for column, analysis in trends.items():
                if 'trend' in analysis and analysis['trend']['direction'] in ['increasing', 'decreasing']:
                    # Strong trends get highlighted
                    if analysis['trend']['strength'] == 'strong':
                        report['insights'].append({
                            'type': 'strong_trend',
                            'column': column,
                            'insight': f"Strong {analysis['trend']['direction']} trend detected in {column} (r²: {analysis['trend']['r_squared']:.2f})"
                        })
                    
                    # Large percentage changes get highlighted
                    if analysis['statistics']['change_pct'] and abs(analysis['statistics']['change_pct']) > 50:
                        report['insights'].append({
                            'type': 'significant_change',
                            'column': column,
                            'insight': f"Significant change in {column}: {analysis['statistics']['change_pct']:.1f}% {analysis['trend']['direction']}"
                        })
            
            return report
        
        except Exception as e:
            self.logger.error(f"Error generating trend report: {e}")
            return {'error': str(e)}
    
    def compare_datasets(self, dataset_id1: int, dataset_id2: int, 
                        version1: int = None, version2: int = None) -> Dict[str, Any]:
        """
        Compare two datasets and identify key differences
        
        Args:
            dataset_id1: ID of the first dataset
            dataset_id2: ID of the second dataset
            version1: Version of the first dataset (optional)
            version2: Version of the second dataset (optional)
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Get both datasets
            dataset1 = self.db.get_dataset(dataset_id1, version1)
            dataset2 = self.db.get_dataset(dataset_id2, version2)
            
            if not dataset1 or not dataset2 or not dataset1['data'] or not dataset2['data']:
                return {'error': "One or both datasets not found or empty"}
            
            # Convert to DataFrames
            df1 = pd.DataFrame(dataset1['data'])
            df2 = pd.DataFrame(dataset2['data'])
            
            # Prepare comparison results
            comparison = {
                'dataset1': {
                    'id': dataset_id1,
                    'name': dataset1['metadata']['name'],
                    'version': dataset1['version'],
                    'record_count': len(df1)
                },
                'dataset2': {
                    'id': dataset_id2,
                    'name': dataset2['metadata']['name'],
                    'version': dataset2['version'],
                    'record_count': len(df2)
                },
                'schema_comparison': {
                    'common_columns': [],
                    'unique_to_dataset1': [],
                    'unique_to_dataset2': []
                },
                'data_comparison': {},
                'insights': []
            }
            
            # Compare schemas
            cols1 = set(df1.columns)
            cols2 = set(df2.columns)
            
            comparison['schema_comparison']['common_columns'] = list(cols1.intersection(cols2))
            comparison['schema_comparison']['unique_to_dataset1'] = list(cols1 - cols2)
            comparison['schema_comparison']['unique_to_dataset2'] = list(cols2 - cols1)
            
            # Compare data distributions for common columns
            for col in comparison['schema_comparison']['common_columns']:
                # Skip if either column is empty
                if df1[col].count() == 0 or df2[col].count() == 0:
                    continue
                
                # Handle different column types
                if pd.api.types.is_numeric_dtype(df1[col]) and pd.api.types.is_numeric_dtype(df2[col]):
                    # Numeric column comparison
                    stats1 = {
                        'mean': float(df1[col].mean()),
                        'median': float(df1[col].median()),
                        'std': float(df1[col].std())
                    }
                    
                    stats2 = {
                        'mean': float(df2[col].mean()),
                        'median': float(df2[col].median()),
                        'std': float(df2[col].std())
                    }
                    
                    # Calculate percentage differences
                    pct_diff_mean = ((stats2['mean'] - stats1['mean']) / stats1['mean'] * 100) if stats1['mean'] != 0 else None
                    
                    comparison['data_comparison'][col] = {
                        'type': 'numeric',
                        'dataset1_stats': stats1,
                        'dataset2_stats': stats2,
                        'differences': {
                            'mean_difference': float(stats2['mean'] - stats1['mean']),
                            'mean_pct_change': float(pct_diff_mean) if pct_diff_mean is not None else None
                        }
                    }
                    
                    # Add insight for significant differences
                    if pct_diff_mean is not None and abs(pct_diff_mean) > 20:
                        comparison['insights'].append({
                            'type': 'significant_difference',
                            'column': col,
                            'insight': f"Significant difference in {col} mean: {pct_diff_mean:.1f}% {'increase' if pct_diff_mean > 0 else 'decrease'}"
                        })
                
                elif pd.api.types.is_categorical_dtype(df1[col]) or pd.api.types.is_object_dtype(df1[col]):
                    # Categorical column comparison
                    counts1 = df1[col].value_counts(normalize=True).to_dict()
                    counts2 = df2[col].value_counts(normalize=True).to_dict()
                    
                    # Find all unique categories
                    all_categories = set(counts1.keys()).union(set(counts2.keys()))
                    
                    # Compare distributions
                    distribution_diff = {}
                    for category in all_categories:
                        pct1 = counts1.get(category, 0) * 100
                        pct2 = counts2.get(category, 0) * 100
                        distribution_diff[category] = pct2 - pct1
                    
                    comparison['data_comparison'][col] = {
                        'type': 'categorical',
                        'dataset1_distribution': {k: v * 100 for k, v in counts1.items()},
                        'dataset2_distribution': {k: v * 100 for k, v in counts2.items()},
                        'distribution_difference_pct': distribution_diff
                    }
                    
                    # Add insights for major category shifts
                    significant_shifts = {k: v for k, v in distribution_diff.items() if abs(v) > 10}
                    if significant_shifts:
                        top_shifts = sorted(significant_shifts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                        for category, shift in top_shifts:
                            comparison['insights'].append({
                                'type': 'category_shift',
                                'column': col,
                                'category': category,
                                'insight': f"Significant shift in '{category}' category: {abs(shift):.1f}% {'increase' if shift > 0 else 'decrease'}"
                            })
            
            # Overall dataset size comparison
            size_diff_pct = ((len(df2) - len(df1)) / len(df1) * 100) if len(df1) > 0 else None
            
            if size_diff_pct is not None:
                comparison['insights'].append({
                    'type': 'dataset_size',
                    'insight': f"Dataset size {'increased' if size_diff_pct > 0 else 'decreased'} by {abs(size_diff_pct):.1f}%"
                })
            
            return comparison
        
        except Exception as e:
            self.logger.error(f"Error comparing datasets: {e}")
            return {'error': str(e)}

# Example usage
if __name__ == "__main__":
    from database_module import PesoDatabase
    from query_module import QueryEngine
    
    # Initialize database and insight engine
    db = PesoDatabase()
    query_engine = QueryEngine(db)
    insights = InsightsEngine(db, query_engine)
    
    # Generate summary for a dataset
    dataset_id = 1  # Assuming dataset with ID 1 exists
    summary = insights.generate_dataset_summary(dataset_id)
    
    print(json.dumps(summary, indent=2))