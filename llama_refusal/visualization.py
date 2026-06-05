"""
Visualization utilities for mechanistic interpretability.
Generate heatmaps, attention visualizations, and interactive dashboards.
"""

import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json


def plot_activation_heatmap(
    activations: Dict[int, torch.Tensor],
    title: str = "Activation Heatmap",
    save_path: Optional[str] = None,
    interactive: bool = True
) -> go.Figure:
    """
    Plot heatmap of activations across layers.
    
    Args:
        activations: Dict mapping layer_idx -> activation tensor
        title: Plot title
        save_path: Path to save figure
        interactive: Use Plotly (True) or Matplotlib (False)
    
    Returns:
        Plotly figure object
    """
    # Extract activation magnitudes
    layer_indices = sorted(activations.keys())
    activation_norms = []
    
    for layer_idx in layer_indices:
        act = activations[layer_idx]
        # Compute mean activation magnitude across batch and sequence
        norm = act.abs().mean().item()
        activation_norms.append(norm)
    
    if interactive:
        # Create Plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=[activation_norms],
            x=layer_indices,
            y=['Activation Magnitude'],
            colorscale='Viridis',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Layer Index",
            yaxis_title="",
            height=300,
            width=1000
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    else:
        # Create Matplotlib heatmap
        plt.figure(figsize=(12, 3))
        sns.heatmap(
            [activation_norms],
            xticklabels=layer_indices,
            yticklabels=['Magnitude'],
            cmap='viridis',
            cbar_kws={'label': 'Activation Magnitude'}
        )
        plt.title(title)
        plt.xlabel('Layer Index')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return None


def plot_activation_difference(
    harmful_cache,
    harmless_cache,
    component_type: str = 'residual',
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Plot difference in activations between harmful and harmless prompts.
    
    Args:
        harmful_cache: ActivationCache from harmful prompt
        harmless_cache: ActivationCache from harmless prompt
        component_type: 'residual', 'attention', or 'mlp'
        save_path: Path to save figure
    
    Returns:
        Plotly figure
    """
    # Get activations based on component type
    if component_type == 'residual':
        harmful_acts = harmful_cache.hidden_states
        harmless_acts = harmless_cache.hidden_states
    elif component_type == 'attention':
        harmful_acts = harmful_cache.attention_out
        harmless_acts = harmless_cache.attention_out
    else:  # mlp
        harmful_acts = harmful_cache.mlp_out
        harmless_acts = harmless_cache.mlp_out
    
    # Compute differences
    layer_indices = sorted(set(harmful_acts.keys()) & set(harmless_acts.keys()))
    differences = []
    
    for layer_idx in layer_indices:
        diff = (harmful_acts[layer_idx] - harmless_acts[layer_idx]).abs().mean().item()
        differences.append(diff)
    
    # Create bar plot
    fig = go.Figure(data=[
        go.Bar(
            x=layer_indices,
            y=differences,
            marker_color='indianred'
        )
    ])
    
    fig.update_layout(
        title=f"Activation Difference: {component_type.capitalize()}",
        xaxis_title="Layer Index",
        yaxis_title="Mean Absolute Difference",
        height=400,
        width=1000
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_attention_pattern(
    attention_pattern: torch.Tensor,
    layer_idx: int,
    head_idx: Optional[int] = None,
    tokens: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Visualize attention patterns.
    
    Args:
        attention_pattern: Attention weights [batch, heads, seq_len, seq_len]
        layer_idx: Layer index
        head_idx: Specific head to visualize (None = average all heads)
        tokens: Token labels for axes
        save_path: Path to save figure
    
    Returns:
        Plotly figure
    """
    # Handle different attention pattern shapes
    if len(attention_pattern.shape) == 4:
        # [batch, heads, seq_len, seq_len]
        attn = attention_pattern[0]  # Take first batch
        if head_idx is not None:
            attn = attn[head_idx]  # Take specific head
        else:
            attn = attn.mean(dim=0)  # Average over heads
    elif len(attention_pattern.shape) == 3:
        # [heads, seq_len, seq_len] or [batch, seq_len, seq_len]
        if head_idx is not None:
            attn = attention_pattern[head_idx]
        else:
            attn = attention_pattern.mean(dim=0)
    else:
        attn = attention_pattern
    
    # Convert to numpy
    attn_np = attn.cpu().numpy()
    
    # Create labels
    if tokens is None:
        tokens = [f"T{i}" for i in range(attn_np.shape[0])]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=attn_np,
        x=tokens,
        y=tokens,
        colorscale='Blues',
        hoverongaps=False
    ))
    
    title = f"Attention Pattern - Layer {layer_idx}"
    if head_idx is not None:
        title += f", Head {head_idx}"
    else:
        title += " (Averaged over heads)"
    
    fig.update_layout(
        title=title,
        xaxis_title="Key Position",
        yaxis_title="Query Position",
        height=600,
        width=600
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_patching_results(
    results: Dict[str, Any],
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Visualize results from activation patching experiments.
    
    Args:
        results: Output from batch_patching_experiment
        save_path: Path to save figure
    
    Returns:
        Plotly figure
    """
    experiments = results['experiments']
    
    # Organize data by layer and component
    data = {
        'layer': [],
        'component': [],
        'induced_refusal': []
    }
    
    for exp in experiments:
        if 'error' not in exp:
            data['layer'].append(exp['layer_patched'])
            data['component'].append(exp['component_type'])
            
            # Check if patching induced refusal
            induced = (
                exp.get('is_refusal_patched', False) and 
                not exp.get('is_refusal_harmless', True)
            )
            data['induced_refusal'].append(1 if induced else 0)
    
    # Create pivot table for heatmap
    import pandas as pd
    df = pd.DataFrame(data)
    pivot = df.pivot_table(
        index='component',
        columns='layer',
        values='induced_refusal',
        aggfunc='mean',
        fill_value=0
    )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn_r',
        text=pivot.values,
        texttemplate='%{text:.0f}',
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Activation Patching Results<br>(1 = Induced Refusal, 0 = No Effect)",
        xaxis_title="Layer Index",
        yaxis_title="Component Type",
        height=400,
        width=1000
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_layer_importance(
    layer_importance: Dict[int, int],
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Plot importance of each layer in refusal mechanism.
    
    Args:
        layer_importance: Dict mapping layer_idx -> importance score
        save_path: Path to save figure
    
    Returns:
        Plotly figure
    """
    layers = sorted(layer_importance.keys())
    scores = [layer_importance[l] for l in layers]
    
    fig = go.Figure(data=[
        go.Bar(
            x=layers,
            y=scores,
            marker_color='steelblue'
        )
    ])
    
    fig.update_layout(
        title="Layer Importance for Refusal Mechanism",
        xaxis_title="Layer Index",
        yaxis_title="Number of Causal Components",
        height=500,
        width=1000
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def create_circuit_diagram(
    causal_components: List[Dict[str, Any]],
    save_path: Optional[str] = None
) -> go.Figure:
    """
    Create a circuit diagram showing refusal mechanism.
    
    Args:
        causal_components: List of components identified as causal
        save_path: Path to save figure
    
    Returns:
        Plotly figure
    """
    # Group by layer
    layers = {}
    for comp in causal_components:
        layer = comp['layer']
        if layer not in layers:
            layers[layer] = []
        layers[layer].append(comp)
    
    # Create network visualization
    fig = go.Figure()
    
    # Add nodes for each layer
    layer_indices = sorted(layers.keys())
    y_positions = np.linspace(0, 10, len(layer_indices))
    
    for i, layer_idx in enumerate(layer_indices):
        components = layers[layer_idx]
        num_comps = len(components)
        
        # Position components horizontally
        x_positions = np.linspace(-1, 1, num_comps)
        
        for j, comp in enumerate(components):
            # Add node
            fig.add_trace(go.Scatter(
                x=[x_positions[j]],
                y=[y_positions[i]],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color='red' if comp['component'] == 'attention' else 'blue'
                ),
                text=[f"L{layer_idx}-{comp['component'][:3]}"],
                textposition="top center",
                name=f"Layer {layer_idx}",
                showlegend=False
            ))
    
    fig.update_layout(
        title="Refusal Circuit Diagram",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(title="Layer Depth", showgrid=True),
        height=800,
        width=600
    )
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def create_dashboard_summary(
    results: Dict[str, Any],
    analysis: Dict[str, Any],
    output_dir: str = "outputs/figures"
) -> str:
    """
    Create comprehensive HTML dashboard with all visualizations.
    
    Args:
        results: Patching experiment results
        analysis: Analysis of results
        output_dir: Directory to save outputs
    
    Returns:
        Path to HTML dashboard
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate all plots
    patching_fig = plot_patching_results(results)
    importance_fig = plot_layer_importance(analysis['layer_importance'])
    circuit_fig = create_circuit_diagram(analysis['causal_components'])
    
    # Create HTML dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Llama-3 Refusal Mechanism Analysis</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #4CAF50;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #666;
                margin-top: 30px;
            }}
            .summary {{
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .plot {{
                background-color: white;
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric {{
                display: inline-block;
                margin: 10px 20px;
                padding: 15px;
                background-color: #e8f5e9;
                border-radius: 5px;
            }}
            .metric-label {{
                font-size: 14px;
                color: #666;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2e7d32;
            }}
        </style>
    </head>
    <body>
        <h1>Llama-3-8B Refusal Mechanism Analysis</h1>
        
        <div class="summary">
            <h2>Executive Summary</h2>
            <div class="metric">
                <div class="metric-label">Total Experiments</div>
                <div class="metric-value">{analysis['total_experiments']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Causal Components Found</div>
                <div class="metric-value">{analysis['num_causal_components']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Most Important Layer</div>
                <div class="metric-value">{analysis['most_important_layers'][0][0] if analysis['most_important_layers'] else 'N/A'}</div>
            </div>
        </div>
        
        <div class="summary">
            <h2>Key Findings</h2>
            <ul>
                <li><strong>Harmful Prompt:</strong> {results.get('harmful_prompt', 'N/A')}</li>
                <li><strong>Harmless Prompt:</strong> {results.get('harmless_prompt', 'N/A')}</li>
                <li><strong>Causal Components Identified:</strong> {analysis['num_causal_components']}</li>
                <li><strong>Top 3 Most Important Layers:</strong>
                    <ul>
                        {' '.join([f"<li>Layer {layer}: {count} components</li>" for layer, count in analysis['most_important_layers'][:3]])}
                    </ul>
                </li>
            </ul>
        </div>
        
        <div class="plot">
            <h2>Activation Patching Results</h2>
            <div id="patching"></div>
        </div>
        
        <div class="plot">
            <h2>Layer Importance</h2>
            <div id="importance"></div>
        </div>
        
        <div class="plot">
            <h2>Refusal Circuit Diagram</h2>
            <div id="circuit"></div>
        </div>
        
        <script>
            var patchingData = {patching_fig.to_json()};
            var importanceData = {importance_fig.to_json()};
            var circuitData = {circuit_fig.to_json()};
            
            Plotly.newPlot('patching', patchingData.data, patchingData.layout);
            Plotly.newPlot('importance', importanceData.data, importanceData.layout);
            Plotly.newPlot('circuit', circuitData.data, circuitData.layout);
        </script>
    </body>
    </html>
    """
    
    dashboard_path = Path(output_dir) / "dashboard.html"
    with open(dashboard_path, 'w') as f:
        f.write(html_content)
    
    print(f"Dashboard created: {dashboard_path}")
    return str(dashboard_path)


if __name__ == '__main__':
    """Test visualization functions."""
    print("="*60)
    print("TESTING VISUALIZATION")
    print("="*60)
    
    # Create dummy data for testing
    print("\nGenerating test visualizations...")
    
    # Test activation heatmap
    dummy_acts = {i: torch.randn(1, 10, 4096) for i in range(32)}
    fig = plot_activation_heatmap(dummy_acts, title="Test Heatmap")
    print("✓ Created activation heatmap")
    
    # Test layer importance
    dummy_importance = {i: np.random.randint(0, 5) for i in range(32)}
    fig = plot_layer_importance(dummy_importance)
    print("✓ Created layer importance plot")
    
    print("\nVisualization utilities ready!")
