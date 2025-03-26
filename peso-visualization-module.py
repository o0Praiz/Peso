#!/usr/bin/env python3
"""
Peso Visualization Module
Provides visualization capabilities for marketing datasets
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import base64

class VisualizationEngine:
    """Engine for generating visualizations from marketing datasets"""
    
    def __init__(self, db_instance, output_dir: str = "visualizations"):
        """
        Initialize visualization engine
        
        Args:
            db_instance: PesoDatabase instance
            output_dir: Directory to store visualization outputs
        """
        self.db = db_instance
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set up visualization styles
        sns.set_theme(style="whitegrid")
        
        # Define color palettes
        self.palettes = {
            "default": sns.color_palette("muted"),
            "categorical": sns.color_palette("husl", 8),
            "sequential": sns.color_palette("Blues"),
            "diverging": sns.color_palette("vlag"),
            "marketing": sns.color_palette(["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"])
        }
    
    def generate_basic_stats_chart(self, dataset_id: int, columns: List[str], 
                                 version: int = None) -> Dict[str, Any]:
        """
        Generate basic statistics charts for specified columns
        
        Args:
            dataset_id: ID of the dataset
            columns: Columns to include in visualization
            version: Specific dataset version to use
            
        Returns:
            Dictionary with chart metadata and file paths
        """
        try:
            # Get the dataset
            dataset = self.db.get_dataset(dataset_id, version)
            
            if not dataset or not dataset['data']:
                return {'error': f"Dataset with ID {dataset_id} not found or empty"}
            
            # Convert to DataFrame
            df = pd.DataFrame(dataset['data'])
            
            # Filter to include only specified columns
            available_columns = [col for col in columns if col in df.columns]
            if not available_columns:
                return {'error': "None of the specified columns found in dataset"}
            
            # Determine column types
            numeric_columns = df[available_columns].select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = df[available_columns].select_dtypes(include=['object', 'category']).columns.tolist()
            
            chart_files = []
            
            # Generate charts for each numeric column
            for col in numeric_columns:
                chart_id = f"stats_{dataset_id}_{col}_{uuid.uuid4().hex[:8]}"
                chart_path = os.path.join(self.output_dir, f"{chart_id}.png")
                
                # Create figure with multiple subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
                # Histogram
                sns.histplot(df[col].dropna(), kde=True, ax=ax1, color=self.palettes['marketing'][0])
                ax1.set_title(f"Distribution of {col}")
                
                # Box plot
                sns.boxplot(y=df[col].dropna(), ax=ax2, color=self.palettes['marketing'][1])
                ax2.set_title(f"Box Plot of {col}")
                
                # Set overall title
                fig.suptitle(f"Statistical Analysis of {col}", fontsize=16)
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                
                # Save figure
                plt.savefig(chart_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                # Add to chart files
                chart_files.append({
                    'column': col,
                    'chart_type': 'basic_stats',
                    'chart_id': chart_id,
                    'file_path': chart_path,
                    'url': f"/api/visualizations/{chart_id}"
                })
            
            # Generate charts for each categorical column
            for col in categorical_columns:
                chart_id = f"categorical_{dataset_id}_{col}_{uuid.uuid4().hex[:8]}"
                chart_path = os.path.join(self.output_dir, f"{chart_id}.png")
                
                # Create figure
                plt.figure(figsize=(10, 6))
                
                # Get value counts and sort
                value_counts = df[col].value_counts().sort_values(ascending=False)
                
                # Limit to top categories if there are too many
                if len(value_counts) > 15:
                    other_count = value_counts[15:].sum()
                    value_counts = value_counts[:15]
                    value_counts['Other'] = other_count
                
                # Create bar chart
                sns.barplot(x=value_counts.index, y=value_counts.values, palette=self.palettes['categorical'])
                plt.title(f"Distribution of {col}", fontsize=14)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Save figure
                plt.savefig(chart_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                # Add to chart files
                chart_files.append({
                    'column': col,
                    'chart_type': 'categorical',
                    'chart_id': chart_id,
                    'file_path': chart_path,
                    'url': f"/api/visualizations/{chart_id}"
                })
            
            return {
                'dataset_id': dataset_id,
                'version': dataset.get('version'),
                'chart_count': len(chart_files),
                'charts': chart_files
            }
            
        except Exception as e:
            self.logger.error(f"Error generating basic stats charts: {e}")
            return {'error': str(e)}
            
    def generate_dashboard(self, dataset_id: int, version: int = None) -> Dict[str, Any]:
        """
        Generate a comprehensive dashboard for a dataset
        
        Args:
            dataset_id: ID of the dataset
            version: Specific dataset version to use
            
        Returns:
            Dictionary with dashboard metadata and file paths
        """
        try:
            # Get the dataset
            dataset = self.db.get_dataset(dataset_id, version)
            
            if not dataset or not dataset['data']:
                return {'error': f"Dataset with ID {dataset_id} not found or empty"}
            
            # Convert to DataFrame
            df = pd.DataFrame(dataset['data'])
            
            # Generate unique dashboard ID
            dashboard_id = f"dashboard_{dataset_id}_{uuid.uuid4().hex[:8]}"
            dashboard_dir = os.path.join(self.output_dir, dashboard_id)
            
            # Create dashboard directory
            if not os.path.exists(dashboard_dir):
                os.makedirs(dashboard_dir)
            
            # Track all generated charts
            charts = []
            
            # Determine column types
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            date_columns = []
            
            # Try to identify date columns
            for col in df.columns:
                try:
                    if pd.to_datetime(df[col], errors='coerce').notna().mean() > 0.7:  # If >70% can be converted
                        date_columns.append(col)
                except:
                    pass  # Not a date column
            
            # 1. Dataset Summary Statistics
            summary_path = os.path.join(dashboard_dir, "summary_stats.png")
            plt.figure(figsize=(12, 8))
            
            # Create summary table
            summary_data = []
            
            # Basic dataset info
            summary_data.append(["Dataset Name", dataset['metadata']['name']])
            summary_data.append(["Dataset Type", dataset['metadata']['type']])
            summary_data.append(["Record Count", len(df)])
            summary_data.append(["Column Count", len(df.columns)])
            summary_data.append(["Numeric Columns", len(numeric_columns)])
            summary_data.append(["Categorical Columns", len(categorical_columns)])
            
            # Missing values summary
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                cols_with_missing = missing_values[missing_values > 0]
                summary_data.append(["Columns with Missing Values", len(cols_with_missing)])
                summary_data.append(["Total Missing Values", missing_values.sum()])
            else:
                summary_data.append(["Missing Values", "None"])
            
            # Create table
            plt.table(cellText=summary_data, colWidths=[0.3, 0.5], 
                     loc='center', cellLoc='left')
            plt.axis('off')
            plt.title("Dataset Summary", fontsize=16)
            plt.tight_layout()
            plt.savefig(summary_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            charts.append({
                'title': 'Dataset Summary',
                'chart_type': 'summary_table',
                'file_path': summary_path,
                'url': f"/api/visualizations/{dashboard_id}/summary_stats.png"
            })
            
            # 2. Distribution of numeric columns (top 5)
            if numeric_columns:
                top_numeric = numeric_columns[:min(5, len(numeric_columns))]
                dist_path = os.path.join(dashboard_dir, "distributions.png")
                
                fig, axes = plt.subplots(len(top_numeric), 1, figsize=(12, 3*len(top_numeric)))
                if len(top_numeric) == 1:
                    axes = [axes]  # Make it iterable for single column
                    
                for i, col in enumerate(top_numeric):
                    sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color=self.palettes['marketing'][i % len(self.palettes['marketing'])])
                    axes[i].set_title(f"Distribution of {col}")
                
                plt.tight_layout()
                plt.savefig(dist_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                charts.append({
                    'title': 'Numeric Distributions',
                    'chart_type': 'distributions',
                    'columns': top_numeric,
                    'file_path': dist_path,
                    'url': f"/api/visualizations/{dashboard_id}/distributions.png"
                })
            
            # 3. Correlation matrix (if multiple numeric columns)
            if len(numeric_columns) > 1:
                corr_path = os.path.join(dashboard_dir, "correlation.png")
                
                plt.figure(figsize=(10, 8))
                corr_matrix = df[numeric_columns].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                           square=True, linewidths=0.5)
                plt.title("Correlation Matrix", fontsize=16)
                plt.tight_layout()
                plt.savefig(corr_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                charts.append({
                    'title': 'Correlation Matrix',
                    'chart_type': 'correlation_heatmap',
                    'columns': numeric_columns,
                    'file_path': corr_path,
                    'url': f"/api/visualizations/{dashboard_id}/correlation.png"
                })
            
            # 4. Category distributions (top 3)
            if categorical_columns:
                top_categorical = categorical_columns[:min(3, len(categorical_columns))]
                
                for i, col in enumerate(top_categorical):
                    cat_path = os.path.join(dashboard_dir, f"category_{i}.png")
                    
                    plt.figure(figsize=(10, 6))
                    
                    # Get value counts and sort
                    value_counts = df[col].value_counts().sort_values(ascending=False)
                    
                    # Limit to top categories if there are too many
                    if len(value_counts) > 10:
                        other_count = value_counts[10:].sum()
                        value_counts = value_counts[:10]
                        value_counts['Other'] = other_count
                    
                    # Create bar chart
                    sns.barplot(x=value_counts.index, y=value_counts.values, 
                               palette=self.palettes['categorical'])
                    plt.title(f"Distribution of {col}", fontsize=14)
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(cat_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    
                    charts.append({
                        'title': f'Distribution of {col}',
                        'chart_type': 'category_distribution',
                        'column': col,
                        'file_path': cat_path,
                        'url': f"/api/visualizations/{dashboard_id}/category_{i}.png"
                    })
            
            # 5. Time series (if date column exists)
            if date_columns and numeric_columns:
                time_path = os.path.join(dashboard_dir, "time_series.png")
                
                # Use the first date column and first numeric column
                date_col = date_columns[0]
                value_col = numeric_columns[0]
                
                # Convert date column to datetime
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                # Drop rows with invalid dates
                df_time = df.dropna(subset=[date_col])
                
                # Sort by date
                df_time = df_time.sort_values(by=date_col)
                
                # Set date as index
                df_time = df_time.set_index(date_col)
                
                # Determine appropriate resampling frequency
                time_range = (df_time.index.max() - df_time.index.min()).days
                
                if time_range > 365*2:
                    freq = 'M'  # Monthly
                elif time_range > 90:
                    freq = 'W'  # Weekly
                else:
                    freq = 'D'  # Daily
                
                # Resample with mean aggregation
                resampled = df_time[value_col].resample(freq).mean()
                
                plt.figure(figsize=(12, 6))
                plt.plot(resampled.index, resampled.values, 
                        marker='o' if len(resampled) < 50 else None,
                        color=self.palettes['marketing'][0])
                
                plt.title(f"Time Series of {value_col}", fontsize=16)
                plt.xlabel("Date")
                plt.ylabel(value_col)
                plt.grid(True, alpha=0.3)
                plt.gcf().autofmt_xdate()
                plt.tight_layout()
                plt.savefig(time_path, dpi=100, bbox_inches='tight')
                plt.close()
                
                charts.append({
                    'title': f'Time Series of {value_col}',
                    'chart_type': 'time_series',
                    'date_column': date_col,
                    'value_column': value_col,
                    'file_path': time_path,
                    'url': f"/api/visualizations/{dashboard_id}/time_series.png"
                })
            
            # Create dashboard metadata
            dashboard_metadata = {
                'dashboard_id': dashboard_id,
                'dataset_id': dataset_id,
                'dataset_name': dataset['metadata']['name'],
                'version': dataset.get('version'),
                'created_at': datetime.now().isoformat(),
                'chart_count': len(charts),
                'charts': charts
            }
            
            # Save dashboard metadata
            metadata_path = os.path.join(dashboard_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(dashboard_metadata, f, indent=2)
            
            return dashboard_metadata
            
        except Exception as e:
            self.logger.error(f"Error generating dashboard: {e}")
            return {'error': str(e)}
    
    def get_visualization(self, chart_id: str) -> Optional[str]:
        """
        Get the file path for a visualization by ID
        
        Args:
            chart_id: ID of the chart
            
        Returns:
            Path to the chart file or None if not found
        """
        # Look for the chart file
        chart_path = os.path.join(self.output_dir, f"{chart_id}.png")
        
        if os.path.exists(chart_path):
            return chart_path
        
        # Check if it's part of a dashboard
        dashboard_id = chart_id.split('/')[0] if '/' in chart_id else None
        if dashboard_id:
            file_name = chart_id.split('/')[-1]
            dashboard_path = os.path.join(self.output_dir, dashboard_id, file_name)
            
            if os.path.exists(dashboard_path):
                return dashboard_path
        
        return None
    
    def export_charts_as_html(self, chart_files: List[Dict[str, Any]], 
                            title: str = "Peso Visualizations") -> str:
        """
        Export charts as HTML document
        
        Args:
            chart_files: List of chart file information
            title: Title for the HTML document
            
        Returns:
            HTML string
        """
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .chart-container {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
        h1, h2 {{ color: #333; }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
"""

        for chart in chart_files:
            # Convert the image to base64
            with open(chart['file_path'], 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Add chart to HTML
            html += f"""
        <div class="chart-container">
            <h2>{chart.get('title', '')}</h2>
            <img src="data:image/png;base64,{img_data}" alt="{chart.get('title', '')}">
        </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        return html
        
# Example usage
if __name__ == "__main__":
    from peso_database_module import PesoDatabase
    
    # Initialize database
    db = PesoDatabase()
    
    # Initialize visualization engine
    viz_engine = VisualizationEngine(db)
    
    # Generate basic stats chart
    result = viz_engine.generate_basic_stats_chart(1, ["country", "industry"])
    
    print(json.dumps(result, indent=2))
    
    def generate_time_series_chart(self, dataset_id: int, date_column: str, 
                                value_columns: List[str], group_by: Optional[str] = None,
                                interval: str = 'D', version: int = None) -> Dict[str, Any]:
        """
        Generate time series chart
        
        Args:
            dataset_id: ID of the dataset
            date_column: Column with date/time data
            value_columns: Numeric columns to plot
            group_by: Optional column to group by
            interval: Resampling interval (D=daily, W=weekly, M=monthly, Q=quarterly, Y=yearly)
            version: Specific dataset version to use
            
        Returns:
            Dictionary with chart metadata and file path
        """
        try:
            # Get the dataset
            dataset = self.db.get_dataset(dataset_id, version)
            
            if not dataset or not dataset['data']:
                return {'error': f"Dataset with ID {dataset_id} not found or empty"}
            
            # Convert to DataFrame
            df = pd.DataFrame(dataset['data'])
            
            # Check if columns exist
            if date_column not in df.columns:
                return {'error': f"Date column '{date_column}' not found in dataset"}
            
            available_value_columns = [col for col in value_columns if col in df.columns]
            if not available_value_columns:
                return {'error': f"None of the specified value columns found in dataset"}
            
            # Check if group_by column exists
            if group_by and group_by not in df.columns:
                return {'error': f"Group by column '{group_by}' not found in dataset"}
            
            # Convert date column to datetime
            try:
                df[date_column] = pd.to_datetime(df[date_column])
            except:
                return {'error': f"Could not convert '{date_column}' to datetime"}
            
            # Sort by date
            df = df.sort_values(by=date_column)
            
            # Generate chart ID and path
            chart_id = f"timeseries_{dataset_id}_{uuid.uuid4().hex[:8]}"
            chart_path = os.path.join(self.output_dir, f"{chart_id}.png")
            
            # Create figure
            plt.figure(figsize=(14, 8))
            
            if group_by:
                # Group by the specified column
                groups = df[group_by].unique()
                
                # Limit to top 8 groups if there are too many
                if len(groups) > 8:
                    # Get the groups with the most data points
                    group_counts = df[group_by].value_counts().head(8).index.tolist()
                    groups = group_counts
                
                for i, group_value in enumerate(groups):
                    group_data = df[df[group_by] == group_value]
                    
                    # Resample time series data
                    group_data = group_data.set_index(date_column)
                    
                    for j, col in enumerate(available_value_columns):
                        # Resample with mean aggregation
                        resampled = group_data[col].resample(interval).mean()
                        
                        # Plot the time series
                        label = f"{group_value} - {col}" if len(available_value_columns) > 1 else str(group_value)
                        plt.plot(resampled.index, resampled.values, 
                                marker='o' if len(resampled) < 50 else None,
                                label=label,
                                color=self.palettes['categorical'][i % len(self.palettes['categorical'])])
            else:
                # No grouping, plot each value column directly
                df = df.set_index(date_column)
                
                for i, col in enumerate(available_value_columns):
                    # Resample with mean aggregation
                    resampled = df[col].resample(interval).mean()
                    
                    # Plot the time series
                    plt.plot(resampled.index, resampled.values, 
                            marker='o' if len(resampled) < 50 else None,
                            label=col,
                            color=self.palettes['marketing'][i % len(self.palettes['marketing'])])
            
            # Add labels and title
            plt.title(f"Time Series Analysis" + (f" by {group_by}" if group_by else ""), fontsize=16)
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return {
                'dataset_id': dataset_id,
                'version': dataset.get('version'),
                'chart_type': 'time_series',
                'chart_id': chart_id,
                'file_path': chart_path,
                'url': f"/api/visualizations/{chart_id}",
                'date_column': date_column,
                'value_columns': available_value_columns,
                'group_by': group_by,
                'interval': interval
            }
            
        except Exception as e:
            self.logger.error(f"Error generating time series chart: {e}")
            return {'error': str(e)}
    
    def generate_category_comparison(self, dataset_id: int, category_column: str, 
                                  value_column: str, version: int = None) -> Dict[str, Any]:
        """
        Generate category comparison chart
        
        Args:
            dataset_id: ID of the dataset
            category_column: Column with categories
            value_column: Numeric column to compare
            version: Specific dataset version to use
            
        Returns:
            Dictionary with chart metadata and file path
        """
        try:
            # Get the dataset
            dataset = self.db.get_dataset(dataset_id, version)
            
            if not dataset or not dataset['data']:
                return {'error': f"Dataset with ID {dataset_id} not found or empty"}
            
            # Convert to DataFrame
            df = pd.DataFrame(dataset['data'])
            
            # Check if columns exist
            if category_column not in df.columns:
                return {'error': f"Category column '{category_column}' not found in dataset"}
            
            if value_column not in df.columns:
                return {'error': f"Value column '{value_column}' not found in dataset"}
            
            # Check if value column is numeric
            if not pd.api.types.is_numeric_dtype(df[value_column]):
                return {'error': f"Value column '{value_column}' must be numeric"}
            
            # Generate chart ID and path
            chart_id = f"category_comp_{dataset_id}_{uuid.uuid4().hex[:8]}"
            chart_path = os.path.join(self.output_dir, f"{chart_id}.png")
            
            # Aggregate data by category
            agg_data = df.groupby(category_column)[value_column].agg(['mean', 'count']).reset_index()
            
            # Sort by mean value
            agg_data = agg_data.sort_values('mean', ascending=False)
            
            # Limit to top 15 categories if there are too many
            if len(agg_data) > 15:
                agg_data = agg_data.head(15)
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Bar chart
            ax = sns.barplot(x=category_column, y='mean', data=agg_data, palette=self.palettes['marketing'])
            
            # Add value labels
            for i, v in enumerate(agg_data['mean']):
                ax.text(i, v + 0.1, f"{v:.2f}", ha='center')
            
            # Add count as text
            for i, (_, row) in enumerate(agg_data.iterrows()):
                ax.text(i, row['mean'] / 2, f"n={row['count']}", ha='center', color='white')
            
            plt.title(f"Comparison of {value_column} by {category_column}", fontsize=16)
            plt.xlabel(category_column)
            plt.ylabel(f"Average {value_column}")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            return {
                'dataset_id': dataset_id,
                'version': dataset.get('version'),
                'chart_type': 'category_comparison',
                'chart_id': chart_id,
                'file_path': chart_path,
                'url': f"/api/visualizations/{chart_id}",
                'category_column': category_column,
                'value_column': value_column
            }
            
        except Exception as e:
            self.logger.error(f"Error generating category comparison chart: {e}")
            return {'error': str(e)}
            self.logger.error(f"Error generating scatter matrix: {e}")
            return {'error': str(e)}
            self.logger.error(f"Error generating correlation heatmap: {e}")
            return {'error': str(e)}