"""
Python Visualizer Agent - Handles visualization and statistical analysis
"""

import asyncio
import json
import sys
import os
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from base_agent import BaseSwarmAgent
import logging
import base64
from io import BytesIO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)


class PythonVisualizerAgent(BaseSwarmAgent):
    """Agent for data visualization and statistical analysis"""
    
    def __init__(self):
        super().__init__(
            agent_type="python_visualizer",
            capabilities=["visualization", "statistics", "heatmap", "scatter", "distribution", "report_generation"]
        )
        self.figure_cache = {}
    
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute visualization task"""
        task_type = task_data.get('type')
        payload = task_data.get('payload', {})
        
        if task_type == "visualization":
            return await self._create_visualization(payload)
        elif task_type == "statistics":
            return await self._run_statistics(payload)
        elif task_type == "report_generation":
            return await self._generate_report(payload)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _create_visualization(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualization based on type"""
        viz_type = payload.get('viz_type', 'heatmap')
        data_key = payload.get('data_key')
        
        # Load data from memory
        data = self.load_from_memory(data_key)
        if not data:
            return {
                'status': 'error',
                'error': f'Data not found for key: {data_key}'
            }
        
        try:
            if viz_type == "heatmap":
                fig = await self._create_heatmap(data, payload)
            elif viz_type == "scatter":
                fig = await self._create_scatter(data, payload)
            elif viz_type == "distribution":
                fig = await self._create_distribution(data, payload)
            elif viz_type == "trait_network":
                fig = await self._create_trait_network(data, payload)
            else:
                raise ValueError(f"Unknown visualization type: {viz_type}")
            
            # Convert to HTML
            fig_html = fig.to_html(include_plotlyjs='cdn')
            
            # Also save as static image
            fig_bytes = fig.to_image(format='png')
            fig_base64 = base64.b64encode(fig_bytes).decode()
            
            # Cache figure
            cache_key = f"{viz_type}:{data_key}"
            self.figure_cache[cache_key] = {
                'html': fig_html,
                'image': fig_base64
            }
            
            # Save to memory
            self.save_to_memory(f"viz:{task_data['id']}", {
                'type': viz_type,
                'html': fig_html,
                'image': fig_base64
            })
            
            return {
                'status': 'success',
                'visualization': {
                    'type': viz_type,
                    'cache_key': cache_key,
                    'format': 'html+png'
                }
            }
            
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _create_heatmap(self, data: Dict[str, Any], params: Dict[str, Any]) -> go.Figure:
        """Create frequency heatmap"""
        # Extract frequency matrix
        frequencies = data.get('frequencies', {})
        
        # Convert to matrix format
        codons = sorted(frequencies.keys())
        positions = list(range(max(len(seq) for seq in frequencies.values())))
        
        matrix = np.zeros((len(codons), len(positions)))
        for i, codon in enumerate(codons):
            for j, freq in enumerate(frequencies.get(codon, [])):
                matrix[i, j] = freq
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=positions,
            y=codons,
            colorscale='Viridis',
            text=matrix.round(2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title=params.get('title', 'Codon Frequency Heatmap'),
            xaxis_title='Position',
            yaxis_title='Codon',
            width=params.get('width', 1000),
            height=params.get('height', 800)
        )
        
        return fig
    
    async def _create_scatter(self, data: Dict[str, Any], params: Dict[str, Any]) -> go.Figure:
        """Create scatter plot"""
        # Extract trait data
        traits = data.get('traits', [])
        
        if not traits:
            raise ValueError("No trait data available")
        
        # Convert to DataFrame
        df = pd.DataFrame(traits)
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x=params.get('x', 'confidence'),
            y=params.get('y', 'frequency'),
            color=params.get('color', 'trait_type'),
            size=params.get('size', 'impact_score'),
            hover_data=['gene_id', 'position'],
            title=params.get('title', 'Trait Analysis Scatter Plot')
        )
        
        fig.update_layout(
            width=params.get('width', 900),
            height=params.get('height', 600)
        )
        
        return fig
    
    async def _create_distribution(self, data: Dict[str, Any], params: Dict[str, Any]) -> go.Figure:
        """Create distribution plot"""
        values = data.get('values', [])
        
        # Create distribution
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=values,
            name='Distribution',
            nbinsx=params.get('bins', 30),
            opacity=0.7
        ))
        
        # Add normal distribution overlay
        mean = np.mean(values)
        std = np.std(values)
        x_range = np.linspace(min(values), max(values), 100)
        normal_curve = stats.norm.pdf(x_range, mean, std) * len(values) * (max(values) - min(values)) / params.get('bins', 30)
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=normal_curve,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=params.get('title', 'Value Distribution'),
            xaxis_title=params.get('xlabel', 'Value'),
            yaxis_title='Frequency',
            showlegend=True,
            width=params.get('width', 800),
            height=params.get('height', 600)
        )
        
        return fig
    
    async def _create_trait_network(self, data: Dict[str, Any], params: Dict[str, Any]) -> go.Figure:
        """Create trait interaction network"""
        traits = data.get('traits', [])
        interactions = data.get('interactions', [])
        
        # Create network graph
        fig = go.Figure()
        
        # Add nodes (traits)
        node_x = []
        node_y = []
        node_text = []
        
        # Simple circular layout
        n_traits = len(traits)
        for i, trait in enumerate(traits):
            angle = 2 * np.pi * i / n_traits
            x = np.cos(angle)
            y = np.sin(angle)
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{trait['name']}<br>Conf: {trait.get('confidence', 0):.2f}")
        
        # Add edges (interactions)
        edge_x = []
        edge_y = []
        
        for interaction in interactions:
            i1 = interaction['trait1_idx']
            i2 = interaction['trait2_idx']
            edge_x.extend([node_x[i1], node_x[i2], None])
            edge_y.extend([node_y[i1], node_y[i2], None])
        
        # Plot edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='#888'),
            hoverinfo='none'
        ))
        
        # Plot nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=[t.get('impact_score', 10) * 20 for t in traits],
                color=[t.get('confidence', 0.5) for t in traits],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Confidence")
            )
        ))
        
        fig.update_layout(
            title=params.get('title', 'Trait Interaction Network'),
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=params.get('width', 800),
            height=params.get('height', 800)
        )
        
        return fig
    
    async def _run_statistics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run statistical analysis"""
        analysis_type = payload.get('analysis_type', 'summary')
        data_key = payload.get('data_key')
        
        # Load data
        data = self.load_from_memory(data_key)
        if not data:
            return {
                'status': 'error',
                'error': f'Data not found for key: {data_key}'
            }
        
        try:
            if analysis_type == "summary":
                results = await self._summary_statistics(data)
            elif analysis_type == "correlation":
                results = await self._correlation_analysis(data)
            elif analysis_type == "significance":
                results = await self._significance_testing(data, payload)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            # Save results
            self.save_to_memory(f"stats:{task_data['id']}", results)
            
            return {
                'status': 'success',
                'statistics': results
            }
            
        except Exception as e:
            logger.error(f"Statistics error: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def _summary_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics"""
        values = data.get('values', [])
        
        if not values:
            return {'error': 'No values provided'}
        
        values_array = np.array(values)
        
        return {
            'count': len(values),
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'median': float(np.median(values_array)),
            'q1': float(np.percentile(values_array, 25)),
            'q3': float(np.percentile(values_array, 75)),
            'skewness': float(stats.skew(values_array)),
            'kurtosis': float(stats.kurtosis(values_array))
        }
    
    async def _correlation_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run correlation analysis"""
        matrix_data = data.get('matrix', [])
        
        if not matrix_data:
            return {'error': 'No matrix data provided'}
        
        # Convert to numpy array
        matrix = np.array(matrix_data)
        
        # Calculate correlations
        corr_matrix = np.corrcoef(matrix)
        
        # Find significant correlations
        n = matrix.shape[0]
        p_values = np.zeros_like(corr_matrix)
        
        for i in range(n):
            for j in range(i+1, n):
                r = corr_matrix[i, j]
                # Calculate p-value
                t = r * np.sqrt((n-2)/(1-r**2))
                p_values[i, j] = p_values[j, i] = 2 * (1 - stats.t.cdf(abs(t), n-2))
        
        # Multiple testing correction
        from statsmodels.stats.multitest import multipletests
        flat_p = p_values[np.triu_indices(n, k=1)]
        _, corrected_p, _, _ = multipletests(flat_p, method='fdr_bh')
        
        return {
            'correlation_matrix': corr_matrix.tolist(),
            'p_values': p_values.tolist(),
            'significant_pairs': [
                {
                    'pair': [i, j],
                    'correlation': float(corr_matrix[i, j]),
                    'p_value': float(p_values[i, j])
                }
                for i in range(n) for j in range(i+1, n)
                if p_values[i, j] < 0.05
            ]
        }
    
    async def _significance_testing(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Run significance tests"""
        group1 = data.get('group1', [])
        group2 = data.get('group2', [])
        test_type = params.get('test_type', 't_test')
        
        if test_type == 't_test':
            statistic, p_value = stats.ttest_ind(group1, group2)
            test_name = "Independent t-test"
        elif test_type == 'mann_whitney':
            statistic, p_value = stats.mannwhitneyu(group1, group2)
            test_name = "Mann-Whitney U test"
        elif test_type == 'ks_test':
            statistic, p_value = stats.ks_2samp(group1, group2)
            test_name = "Kolmogorov-Smirnov test"
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Effect size
        effect_size = (np.mean(group1) - np.mean(group2)) / np.sqrt((np.var(group1) + np.var(group2)) / 2)
        
        return {
            'test': test_name,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'effect_size': float(effect_size),
            'significant': p_value < 0.05,
            'group1_mean': float(np.mean(group1)),
            'group2_mean': float(np.mean(group2)),
            'group1_std': float(np.std(group1)),
            'group2_std': float(np.std(group2))
        }
    
    async def _generate_report(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis report"""
        analysis_keys = payload.get('analysis_keys', [])
        
        # Collect all results
        report_data = {}
        for key in analysis_keys:
            data = self.load_from_memory(key)
            if data:
                report_data[key] = data
        
        # Generate HTML report
        html_report = self._create_html_report(report_data)
        
        # Save report
        report_key = f"report:{task_data['id']}"
        self.save_to_memory(report_key, {
            'html': html_report,
            'timestamp': datetime.now().isoformat(),
            'included_analyses': analysis_keys
        })
        
        return {
            'status': 'success',
            'report_key': report_key,
            'num_analyses': len(report_data)
        }
    
    def _create_html_report(self, data: Dict[str, Any]) -> str:
        """Create HTML report from analysis data"""
        html = """
        <html>
        <head>
            <title>Genomic Pleiotropy Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                h2 { color: #666; }
                .section { margin: 20px 0; }
                .figure { margin: 15px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Genomic Pleiotropy Analysis Report</h1>
            <p>Generated: {timestamp}</p>
        """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        for key, analysis in data.items():
            html += f"<div class='section'><h2>{key}</h2>"
            
            if isinstance(analysis, dict):
                if 'html' in analysis:
                    html += f"<div class='figure'>{analysis['html']}</div>"
                else:
                    html += "<table>"
                    for k, v in analysis.items():
                        html += f"<tr><th>{k}</th><td>{v}</td></tr>"
                    html += "</table>"
            else:
                html += f"<pre>{json.dumps(analysis, indent=2)}</pre>"
            
            html += "</div>"
        
        html += "</body></html>"
        return html
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect visualizer metrics"""
        return {
            'cached_figures': len(self.figure_cache),
            'figure_types': list(set(k.split(':')[0] for k in self.figure_cache.keys())),
            'memory_usage_mb': sys.getsizeof(self.figure_cache) / 1024 / 1024
        }


# Run agent if executed directly
async def main():
    agent = PythonVisualizerAgent()
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())