import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
import yaml
import os

class PipelineVisualizer:
    def __init__(self, config_path: str, output_dir: str = "visualization_outputs"):
        """Initialize the visualizer with config path and output directory for saving plots."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set base paths - adjust to be relative to CultureBank root
        self.project_base = Path('../../')  # Go up to CultureBank root
        self.result_base = self.project_base / 'data_process_pipeline/results'

    def save_plot(self, fig, filename: str):
        """Save a plotly figure to HTML and PNG formats."""
        fig.write_html(self.output_dir / f"{filename}.html")
        fig.write_image(self.output_dir / f"{filename}.png")

    def read_pipeline_output(self, stage_name: str) -> pd.DataFrame:
        """Read the output CSV file for a specific pipeline stage."""
        try:
            stage_config = self.config[stage_name]
            # Remove the 'data_process_pipeline/' prefix if it exists
            output_file = stage_config['output_file'].replace('data_process_pipeline/', '')
            
            # Construct path relative to project base
            file_path = self.project_base / 'data_process_pipeline' / output_file
            
            print(f"Attempting to read: {file_path}")
            if file_path.exists():
                return pd.read_csv(file_path)
            else:
                raise FileNotFoundError(f"File not found at: {file_path}")
            
        except Exception as e:
            print(f"Error reading output for stage {stage_name}: {str(e)}")
            print(f"Config for stage: {self.config.get(stage_name, 'Stage not found in config')}")
            raise

    def plot_culture_relevance_distribution(self) -> go.Figure:
        """Create a pie chart showing the distribution of culture-relevant vs non-relevant content."""
        df = self.read_pipeline_output('0_culture_relevance_classifier')
        
        # Count the distribution of culture relevance
        relevance_counts = df['is_culture_relevant'].value_counts()
        
        fig = px.pie(values=relevance_counts.values, 
                    names=relevance_counts.index,
                    title='Distribution of Culture-Relevant Content',
                    color_discrete_map={True: '#2ecc71', False: '#e74c3c'})
        
        # Add percentage labels
        fig.update_traces(textinfo='percent+label')
        return fig

    def plot_knowledge_extraction_insights(self) -> go.Figure:
        """Create visualizations for knowledge extraction results."""
        df = self.read_pipeline_output('1_knowledge_extractor')
        
        # Create a figure with subplots
        fig = go.Figure()
        
        # Plot 1: Distribution of extracted insights
        insight_counts = df['insight_type'].value_counts()
        
        fig.add_trace(go.Bar(
            x=insight_counts.index,
            y=insight_counts.values,
            name='Insight Types',
            marker_color='#3498db'
        ))
        
        fig.update_layout(
            title='Distribution of Extracted Cultural Insights',
            xaxis_title='Insight Type',
            yaxis_title='Count',
            showlegend=False
        )
        
        return fig

    def plot_clustering_analysis(self) -> go.Figure:
        """Create comprehensive visualizations for clustering results."""
        df = self.read_pipeline_output('3_clustering_component')
        
        # Create a figure with subplots
        fig = go.Figure()
        
        # Plot 1: Cluster size distribution
        cluster_sizes = df.groupby('cluster_id').size()
        
        fig.add_trace(go.Bar(
            x=cluster_sizes.index,
            y=cluster_sizes.values,
            name='Cluster Sizes',
            marker_color='#9b59b6'
        ))
        
        fig.update_layout(
            title='Cluster Size Distribution',
            xaxis_title='Cluster ID',
            yaxis_title='Number of Statements',
            showlegend=False
        )
        
        return fig

    def plot_topic_normalization(self) -> go.Figure:
        """Create visualizations for topic normalization results."""
        df = self.read_pipeline_output('5_topic_normalizer')
        
        # Create topic-cluster heatmap
        topic_cluster_matrix = pd.crosstab(df['cluster_id'], df['topic'])
        
        fig = px.imshow(topic_cluster_matrix,
                       title='Topic Distribution Across Clusters',
                       labels={'x': 'Topic', 'y': 'Cluster ID', 'color': 'Count'},
                       color_continuous_scale='Viridis')
        
        return fig

    def plot_agreement_analysis(self) -> go.Figure:
        """Create visualizations for agreement analysis."""
        df = self.read_pipeline_output('6_agreement_calculator')
        
        # Create a figure with subplots
        fig = go.Figure()
        
        # Plot 1: Agreement score distribution
        fig.add_trace(go.Box(
            y=df['agreement_score'],
            name='Agreement Scores',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8,
            marker_color='#e67e22'
        ))
        
        fig.update_layout(
            title='Distribution of Agreement Scores',
            yaxis_title='Agreement Score',
            showlegend=False
        )
        
        return fig

    def plot_moderation_analysis(self) -> go.Figure:
        """Create comprehensive visualizations for content moderation results."""
        df = self.read_pipeline_output('7_content_moderation')
        
        # Create a figure with subplots
        fig = go.Figure()
        
        # Plot 1: Controversial prediction scores
        fig.add_trace(go.Histogram(
            x=df['pred_score'],
            name='Prediction Scores',
            nbinsx=50,
            marker_color='#e74c3c'
        ))
        
        fig.update_layout(
            title='Distribution of Controversial Prediction Scores',
            xaxis_title='Prediction Score',
            yaxis_title='Count',
            showlegend=False
        )
        
        return fig

    def create_pipeline_summary_dashboard(self) -> go.Figure:
        """Create a comprehensive dashboard showing key metrics across pipeline stages."""
        # Read data from different stages
        relevance_df = self.read_pipeline_output('0_culture_relevance_classifier')
        moderation_df = self.read_pipeline_output('7_content_moderation')
        agreement_df = self.read_pipeline_output('6_agreement_calculator')
        clustering_df = self.read_pipeline_output('3_clustering_component')
        
        # Calculate summary metrics
        total_statements = len(relevance_df)
        culture_relevant_percentage = (relevance_df['is_culture_relevant'].sum() / total_statements) * 100
        average_agreement_score = agreement_df['agreement_score'].mean()
        total_clusters = len(clustering_df['cluster_id'].unique())
        controversial_percentage = (moderation_df['pred_label'] == 'controversial').mean() * 100
        
        # Create subplots for different metrics
        fig = go.Figure()
        
        # Add traces for different metrics
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=total_statements,
            title={'text': "Total Statements"},
            domain={'x': [0, 0.5], 'y': [0.5, 1]}
        ))
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=culture_relevant_percentage,
            title={'text': "Culture Relevant %"},
            domain={'x': [0.5, 1], 'y': [0.5, 1]}
        ))
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=average_agreement_score,
            title={'text': "Avg Agreement Score"},
            domain={'x': [0, 0.5], 'y': [0, 0.5]}
        ))
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=controversial_percentage,
            title={'text': "Controversial %"},
            domain={'x': [0.5, 1], 'y': [0, 0.5]}
        ))
        
        fig.update_layout(
            title="Pipeline Summary Dashboard",
            grid={'rows': 2, 'columns': 2, 'pattern': "independent"}
        )
        
        return fig

    def plot_pii_detection(self) -> go.Figure:
        """Create visualizations for PII detection results."""
        df = self.read_pipeline_output('7_content_moderation')
        
        # Extract PII types from keywords_list
        pii_types = []
        for keywords in df['keywords_list']:
            if isinstance(keywords, dict):
                pii_types.extend(keywords.values())
        
        # Count PII types
        pii_counts = pd.Series(pii_types).value_counts()
        
        fig = px.bar(
            x=pii_counts.index,
            y=pii_counts.values,
            title='Distribution of Detected PII Types',
            labels={'x': 'PII Type', 'y': 'Count'},
            color=pii_counts.values,
            color_continuous_scale='Viridis'
        )
        
        return fig

    def generate_all_visualizations(self):
        """Generate all visualizations and save them to the output directory."""
        # Generate and save each visualization
        visualizations = {
            'culture_relevance': self.plot_culture_relevance_distribution(),
            'knowledge_extraction': self.plot_knowledge_extraction_insights(),
            'clustering_analysis': self.plot_clustering_analysis(),
            'topic_normalization': self.plot_topic_normalization(),
            'agreement_analysis': self.plot_agreement_analysis(),
            'moderation_analysis': self.plot_moderation_analysis(),
            'pii_detection': self.plot_pii_detection(),
            'pipeline_summary': self.create_pipeline_summary_dashboard()
        }
        
        # Save all visualizations
        for name, fig in visualizations.items():
            self.save_plot(fig, name)

if __name__ == "__main__":
    # Use paths relative to current directory (pipeline/)
    config_path = "../configs/config_dummy_data_vanilla_mistral.yaml"
    output_dir = "../results/visualizations"
    
    print("Starting visualization generation...")
    print(f"Using config file: {config_path}")
    print(f"Output directory: {output_dir}")
    
    visualizer = PipelineVisualizer(
        config_path=config_path,
        output_dir=output_dir
    )
    
    # Generate all visualizations
    print("Generating visualizations...")
    try:
        visualizer.generate_all_visualizations()
        print(f"Visualizations saved in: {output_dir}")
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")