import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

class EnhancedPipelineVisualizer:
    def __init__(self, config_path: str, output_dir: str = "enhanced_visualization_outputs"):
        """Initialize the enhanced visualizer with config path and output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set base paths - adjust to be relative to CultureBank root
        self.project_base = Path('../../')  # Go up to CultureBank root
        self.result_base = self.project_base / 'data_process_pipeline/results'
        
        # Load all dataframes
        print("Loading pipeline output files...")
        self.df_0 = self.read_pipeline_output('0_culture_relevance_classifier')
        self.df_1 = self.read_pipeline_output('1_knowledge_extractor')
        # self.df_3 = self.read_pipeline_output('3_clustering_component')
        self.df_4 = self.read_pipeline_output('4_cluster_summarizer')
        # self.df_5 = self.read_pipeline_output('5_topic_normalizer')
        self.df_6 = self.read_pipeline_output('6_agreement_calculator')
        self.df_7 = self.read_pipeline_output('7_content_moderation')
        self.df_8 = self.read_pipeline_output('8_final_formatter')
        
        # Load annotation file for moderation if it exists
        try:
            moderation_config = self.config['7_content_moderation']
            annotation_file = moderation_config['output_file_for_manual_annotation'].replace('data_process_pipeline/', '')
            self.df_7_annotation = pd.read_csv(self.project_base / 'data_process_pipeline' / annotation_file)
            print(f"Successfully loaded moderation annotation file")
        except Exception as e:
            print(f"Warning: Could not load moderation annotation file: {str(e)}")
            self.df_7_annotation = self.df_7.copy()  # Use regular output as fallback

    def read_pipeline_output(self, stage_name: str) -> pd.DataFrame:
        """Read the output CSV file for a specific pipeline stage."""
        try:
            stage_config = self.config[stage_name]
            # Remove the 'data_process_pipeline/' prefix if it exists
            output_file = stage_config['output_file'].replace('data_process_pipeline/', '')
            
            # Construct path relative to project base
            file_path = self.project_base / 'data_process_pipeline' / output_file
            
            print(f"Loading {stage_name} from: {file_path}")
            if file_path.exists():
                return pd.read_csv(file_path)
            else:
                raise FileNotFoundError(f"File not found at: {file_path}")
            
        except Exception as e:
            print(f"Error reading output for stage {stage_name}: {str(e)}")
            print(f"Config for stage: {self.config.get(stage_name, 'Stage not found in config')}")
            raise

    def save_plot(self, fig, filename: str):
        """Save a plotly figure to HTML and PNG formats."""
        fig.write_html(self.output_dir / f"{filename}.html")
        fig.write_image(self.output_dir / f"{filename}.png")
        
    def create_pipeline_flow_diagram(self):
        """
        Create a Sankey diagram showing how data flows through the pipeline
        with the count of records at each stage.
        """
        # Count records at each stage
        counts = {
            "Input": len(self.df_0),
            "Culture Relevant": self.df_0['pred_label'].value_counts().get('Yes', 0),
            "Knowledge Extraction": len(self.df_1),
            # "Clustering": len(self.df_3['cluster_id'].unique()),
            "Summarization": len(self.df_4),
            # "Topic Normalization": len(self.df_5),
            "Agreement Calculation": self.df_6['norm'].count(),
            "Content Moderation": len(self.df_7_annotation),
            "Final Output": len(self.df_8)
        }
        
        # Define Sankey diagram nodes and links
        nodes = []
        for stage in counts.keys():
            nodes.append({"name": f"{stage} ({counts[stage]})"})
        
        links = [
            {"source": 0, "target": 1, "value": counts["Culture Relevant"]},
            {"source": 1, "target": 2, "value": counts["Knowledge Extraction"]},
            # {"source": 2, "target": 3, "value": counts["Clustering"]},
            {"source": 2, "target": 3, "value": counts["Summarization"]},
            # {"source": 3, "target": 4, "value": counts["Topic Normalization"]},
            {"source": 3, "target": 4, "value": counts["Agreement Calculation"]},
            {"source": 4, "target": 5, "value": counts["Content Moderation"]},
            {"source": 5, "target": 6, "value": counts["Final Output"]}
        ]
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=[node["name"] for node in nodes],
                color="blue"
            ),
            link=dict(
                source=[link["source"] for link in links],
                target=[link["target"] for link in links],
                value=[link["value"] for link in links]
            )
        )])
        
        fig.update_layout(
            title_text="Pipeline Flow Metrics",
            font_size=12,
            height=600
        )
        
        return fig
    
    def create_culture_relevance_stats(self):
        """
        Create a visualization of culture relevance statistics 
        including prediction scores and distribution.
        """
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "histogram"}]],
            subplot_titles=("Culture Relevance Distribution", "Prediction Score Distribution")
        )
        
        # Count the distribution of culture relevance
        relevance_counts = self.df_0['pred_label'].value_counts()
        
        # Plot 1: Pie chart
        fig.add_trace(
            go.Pie(
                labels=relevance_counts.index,
                values=relevance_counts.values,
                textinfo='percent+label',
                marker=dict(colors=['#2ecc71', '#e74c3c'])
            ),
            row=1, col=1
        )
        
        # Plot 2: Histogram of prediction scores
        fig.add_trace(
            go.Histogram(
                x=self.df_0['pred_score'],
                nbinsx=20,
                marker_color='#3498db'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Culture Relevance Analysis",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_cultural_group_analysis(self):
        """
        Create a visualization of cultural groups and their topics.
        """
        # Use the df_5 for this analysis as it contains normalized topic data
        # Group by cultural group and count topics
        # cultural_data = self.df_5.copy()
        
        # Remove any rows with NaN values in cultural group for analysis
        # cultural_data = cultural_data[cultural_data['cultural group'].notna()]
        
        # Count cultural groups
        # cultural_group_counts = cultural_data['cultural group'].value_counts().reset_index()
        # cultural_group_counts.columns = ['cultural_group', 'count']
        
        # Create a word cloud-like visualization for cultural groups
        # fig = px.treemap(
        #     cultural_group_counts,
        #     path=['cultural_group'],
        #     values='count',
        #     title='Cultural Groups Distribution',
        #     color='count',
        #     color_continuous_scale='Blues'
        # )
        
        # Add topic information to hover
        # topics_by_group = cultural_data.groupby('cultural group')['topic'].apply(list).reset_index()
        # topics_by_group['topic_str'] = topics_by_group['topic'].apply(lambda x: ', '.join(set(x)))
        
        # fig.update_layout(
        #     height=600,
        #     margin=dict(t=50, l=25, r=25, b=25)
        # )
        
        # return fig
        
        # Create a placeholder if no cultural group data is available
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Cultural group analysis not available (Step 5 disabled).",
            showarrow=False,
            font=dict(size=14)
        )
        
        fig.update_layout(
            title_text="Cultural Group Analysis (Disabled)",
            height=500
        )
        
        return fig
    
    def create_topic_visualization(self):
        """
        Create a visualization of topics and their relationships.
        """
        # Extract topics from df_5
        # topic_data = self.df_5[['cluster_id', 'topic', 'representative_topic']].copy()
        
        # Count topics
        # topic_counts = topic_data['representative_topic'].value_counts().reset_index()
        # topic_counts.columns = ['topic', 'count']
        
        # Create horizontal bar chart
        # fig = px.bar(
        #     topic_counts,
        #     y='topic',
        #     x='count',
        #     title='Representative Topics Distribution',
        #     color='count',
        #     color_continuous_scale='Viridis',
        #     orientation='h'
        # )
        
        # fig.update_layout(
        #     height=500,
        #     yaxis={'categoryorder':'total ascending'}
        # )
        
        # return fig
        
        # Create a placeholder if no topic data is available
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Topic visualization not available (Step 5 disabled).",
            showarrow=False,
            font=dict(size=14)
        )
        
        fig.update_layout(
            title_text="Topic Visualization (Disabled)",
            height=500
        )
        
        return fig
    
    def create_cluster_analysis(self):
        """
        Create a multi-faceted analysis of clusters.
        """
        # Combine data from df_3 and df_4 for cluster analysis
        # cluster_data = self.df_4.copy()
        
        # Create a figure with subplots
        # fig = make_subplots(
        #     rows=2, cols=1,
        #     subplot_titles=("Cluster Size Distribution", "Topics by Cluster"),
        #     vertical_spacing=0.2,
        #     specs=[[{"type": "bar"}], [{"type": "heatmap"}]]
        # )
        
        # Plot 1: Cluster size distribution
        # cluster_sizes = cluster_data['cluster_size'].tolist()
        # cluster_ids = cluster_data['cluster_id'].tolist()
        
        # fig.add_trace(
        #     go.Bar(
        #         x=cluster_ids,
        #         y=cluster_sizes,
        #         marker_color='#9b59b6',
        #         name='Cluster Size'
        #     ),
        #     row=1, col=1
        # )
        
        # Plot 2: Topics by cluster as a heatmap
        # Create a cross-tabulation matrix for clusters and topics
        # topic_cluster_data = pd.crosstab(
        #     index=cluster_data['cluster_id'],
        #     columns=cluster_data['topic']
        # ).fillna(0)
        
        # Convert to a format for heatmap
        # z_data = topic_cluster_data.values
        # x_data = topic_cluster_data.columns
        # y_data = topic_cluster_data.index
        
        # fig.add_trace(
        #     go.Heatmap(
        #         z=z_data,
        #         x=x_data,
        #         y=y_data,
        #         colorscale='Viridis',
        #         showscale=True
        #     ),
        #     row=2, col=1
        # )
        
        # fig.update_layout(
        #     height=800,
        #     title_text="Cluster Analysis"
        # )
        
        # return fig
        
        # Create a placeholder if no cluster data is available
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5, y=0.5,
            text="Cluster analysis not available (Step 3 disabled).",
            showarrow=False,
            font=dict(size=14)
        )
        
        fig.update_layout(
            title_text="Cluster Analysis (Disabled)",
            height=500
        )
        
        return fig
    
    def create_agreement_score_analysis(self):
        """
        Create a visualization of agreement scores.
        """
        # Extract agreement scores from df_6
        agreement_data = self.df_6[['cluster_id', 'norm', 'cultural group', 'topic']].copy()
        agreement_data = agreement_data.dropna(subset=['norm'])
        
        # Create a figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Agreement Score Distribution", "Agreement by Cultural Group"),
            specs=[[{"type": "box"}, {"type": "bar"}]]
        )
        
        # Plot 1: Box plot of agreement scores
        fig.add_trace(
            go.Box(
                y=agreement_data['norm'],
                name='Agreement Scores',
                marker_color='#e67e22',
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ),
            row=1, col=1
        )
        
        # Plot 2: Average agreement by cultural group
        avg_agreement_by_group = agreement_data.groupby('cultural group')['norm'].mean().reset_index()
        avg_agreement_by_group = avg_agreement_by_group.sort_values('norm', ascending=False)
        
        fig.add_trace(
            go.Bar(
                x=avg_agreement_by_group['cultural group'],
                y=avg_agreement_by_group['norm'],
                marker_color='#3498db',
                name='Avg Agreement'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            title_text="Agreement Score Analysis"
        )
        
        return fig
    
    def create_temporal_analysis(self):
        """
        Create a visualization showing how insights change over time.
        """
        # Extract time data from df_8
        if 'time_range' in self.df_8.columns:
            # Extract year information from time_range
            fig = go.Figure()
            
            fig.add_annotation(
                x=0.5, y=0.5,
                text="Limited time data available. Please provide timestamp data for detailed temporal analysis.",
                showarrow=False,
                font=dict(size=14)
            )
            
            fig.update_layout(
                title_text="Temporal Analysis (Limited Data)",
                height=500
            )
            
            return fig
        
        else:
            # Create a placeholder if no time data is available
            fig = go.Figure()
            
            fig.add_annotation(
                x=0.5, y=0.5,
                text="No time data available. Please provide timestamp data for temporal analysis.",
                showarrow=False,
                font=dict(size=14)
            )
            
            fig.update_layout(
                title_text="Temporal Analysis (No Data)",
                height=500
            )
            
            return fig
    
    def create_content_moderation_analysis(self):
        """
        Create visualizations for content moderation results.
        """
        # Use df_7_annotation for this analysis
        if 'pred_label' in self.df_7_annotation.columns and 'pred_score' in self.df_7_annotation.columns:
            moderation_data = self.df_7_annotation[['pred_label', 'pred_score', 'keywords_list']].copy()
            
            # Create a figure with subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Content Classification", "Prediction Score Distribution"),
                specs=[[{"type": "pie"}, {"type": "histogram"}]]
            )
            
            # Plot 1: Pie chart of prediction labels
            label_counts = moderation_data['pred_label'].value_counts()
            
            fig.add_trace(
                go.Pie(
                    labels=label_counts.index,
                    values=label_counts.values,
                    textinfo='percent+label',
                    marker=dict(colors=['#2ecc71', '#e74c3c'])
                ),
                row=1, col=1
            )
            
            # Plot 2: Histogram of prediction scores
            fig.add_trace(
                go.Histogram(
                    x=moderation_data['pred_score'],
                    nbinsx=20,
                    marker_color='#3498db'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                height=500,
                title_text="Content Moderation Analysis"
            )
            
            return fig
        else:
            # Create a placeholder if no moderation data is available
            fig = go.Figure()
            
            fig.add_annotation(
                x=0.5, y=0.5,
                text="No moderation data available in the expected format.",
                showarrow=False,
                font=dict(size=14)
            )
            
            fig.update_layout(
                title_text="Content Moderation Analysis (No Data)",
                height=500
            )
            
            return fig
    
    def create_pii_detection_analysis(self):
        """
        Create visualizations for PII detection results.
        """
        # Use df_7_annotation for this analysis if it has PII detection data
        if 'keywords_list' in self.df_7_annotation.columns:
            # Try to extract PII types
            pii_data = self.df_7_annotation.copy()
            
            # Check if the keywords_list column contains dictionary-like strings
            if isinstance(pii_data['keywords_list'].iloc[0], str) and '{' in pii_data['keywords_list'].iloc[0]:
                try:
                    # Extract PII types from string representation of dictionaries
                    pii_types = []
                    for keywords in pii_data['keywords_list']:
                        if isinstance(keywords, str) and '{' in keywords:
                            # Convert string representation to actual dictionary
                            keywords_dict = eval(keywords.replace("'", '"'))
                            pii_types.extend(keywords_dict.values())
                    
                    # Count PII types
                    pii_counts = pd.Series(pii_types).value_counts().reset_index()
                    pii_counts.columns = ['pii_type', 'count']
                    
                    # Create the bar chart
                    fig = px.bar(
                        pii_counts,
                        x='pii_type',
                        y='count',
                        title='PII Type Detection Distribution',
                        color='count',
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(
                        height=500,
                        xaxis_title='PII Type',
                        yaxis_title='Count'
                    )
                    
                    return fig
                except:
                    # If there's an error processing the keywords_list, create a placeholder
                    fig = go.Figure()
                    
                    fig.add_annotation(
                        x=0.5, y=0.5,
                        text="Error processing PII detection data. Check the format of keywords_list.",
                        showarrow=False,
                        font=dict(size=14)
                    )
                    
                    fig.update_layout(
                        title_text="PII Detection Analysis (Error)",
                        height=500
                    )
                    
                    return fig
            else:
                # If no PII data is available in the expected format, create a placeholder
                fig = go.Figure()
                
                fig.add_annotation(
                    x=0.5, y=0.5,
                    text="No PII detection data available in the expected format.",
                    showarrow=False,
                    font=dict(size=14)
                )
                
                fig.update_layout(
                    title_text="PII Detection Analysis (No Data)",
                    height=500
                )
                
                return fig
        else:
            # Create a placeholder if no PII data is available
            fig = go.Figure()
            
            fig.add_annotation(
                x=0.5, y=0.5,
                text="No PII detection data available.",
                showarrow=False,
                font=dict(size=14)
            )
            
            fig.update_layout(
                title_text="PII Detection Analysis (No Data)",
                height=500
            )
            
            return fig
    
    def create_summary_dashboard(self):
        """
        Create a comprehensive dashboard showing key metrics across pipeline stages.
        """
        # Calculate key metrics from various dataframes
        total_statements = len(self.df_0)
        culture_relevant_count = self.df_0['pred_label'].value_counts().get('Yes', 0)
        culture_relevant_percentage = (culture_relevant_count / total_statements) * 100 if total_statements > 0 else 0
        
        # Count unique cultural groups from df_5
        # cultural_groups_count = self.df_5['cultural group'].nunique()
        cultural_groups_count = 0  # Placeholder since df_5 is disabled
        
        # Count unique topics from df_5
        # topics_count = self.df_5['topic'].nunique()
        topics_count = 0  # Placeholder since df_5 is disabled
        
        # Calculate average agreement score if available
        avg_agreement_score = 0
        if 'norm' in self.df_6.columns:
            avg_agreement_score = self.df_6['norm'].dropna().mean()
        
        # Create a figure with indicator gauges
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                  [{'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=("Total Statements", "Culture Relevant %", 
                           "Cultural Groups Count", "Topics Count")
        )
        
        # Add indicators for each metric
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=total_statements,
                title={"text": "Total Statements"},
                number={"font": {"color": "#3498db"}}
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=culture_relevant_percentage,
                title={"text": "Culture Relevant %"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#2ecc71"}
                },
                number={"suffix": "%"}
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=cultural_groups_count,
                title={"text": "Cultural Groups"},
                number={"font": {"color": "#9b59b6"}}
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=topics_count,
                title={"text": "Unique Topics"},
                number={"font": {"color": "#e74c3c"}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="Pipeline Summary Dashboard"
        )
        
        return fig
        
    def generate_all_visualizations(self):
        """Generate all visualizations and save them to the output directory."""
        # Generate and save each visualization
        visualizations = {
            'pipeline_flow_diagram': self.create_pipeline_flow_diagram(),
            'culture_relevance_stats': self.create_culture_relevance_stats(),
            'cultural_group_analysis': self.create_cultural_group_analysis(),
            'topic_visualization': self.create_topic_visualization(),
            'cluster_analysis': self.create_cluster_analysis(),
            'agreement_score_analysis': self.create_agreement_score_analysis(),
            'temporal_analysis': self.create_temporal_analysis(),
            'content_moderation_analysis': self.create_content_moderation_analysis(),
            'pii_detection_analysis': self.create_pii_detection_analysis(),
            'summary_dashboard': self.create_summary_dashboard()
        }
        
        # Save all visualizations
        for name, fig in visualizations.items():
            self.save_plot(fig, name)
            
        return visualizations

# Example usage
if __name__ == "__main__":
    # Use paths relative to current directory (pipeline/)
    config_path = "../configs/config_dummy_data_vanilla_mistral.yaml"
    output_dir = "../results/visualizations"
    
    print("Starting enhanced visualization generation...")
    print(f"Using config file: {config_path}")
    print(f"Output directory: {output_dir}")
    
    # Initialize the enhanced visualizer
    visualizer = EnhancedPipelineVisualizer(
        config_path=config_path,
        output_dir=output_dir
    )
    
    # Generate all visualizations
    print("Generating enhanced visualizations...")
    try:
        visualizations = visualizer.generate_all_visualizations()
        print(f"Enhanced visualizations saved in: {output_dir}")
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
        import traceback
        print(traceback.format_exc())