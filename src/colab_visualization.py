"""
Enhanced visualizations for Google Colab.
Creates comprehensive interactive plots for patching results.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
from pathlib import Path


def plot_logit_effects_heatmap(results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Create heatmap of causal effects across layers and components.
    
    Args:
        results: Patching experiment results
        save_path: Path to save HTML
    """
    experiments = [e for e in results['experiments'] if 'error' not in e]
    
    # Create matrix: layers x components
    layers = sorted(set(e['layer'] for e in experiments))
    components = sorted(set(e['component'] for e in experiments))
    
    # Build effect matrix
    effect_matrix = np.zeros((len(components), len(layers)))
    
    for exp in experiments:
        layer_idx = layers.index(exp['layer'])
        comp_idx = components.index(exp['component'])
        effect_matrix[comp_idx, layer_idx] = exp.get('causal_effect', 0)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=effect_matrix,
        x=[f"L{l}" for l in layers],
        y=components,
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(effect_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Causal Effect<br>(Logit Diff)")
    ))
    
    fig.update_layout(
        title={
            'text': "🔥 Causal Effects: Activation Patching<br><sub>Positive = Induces Refusal | Negative = Induces Compliance</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Layer",
        yaxis_title="Component Type",
        width=1200,
        height=400,
        font=dict(size=12)
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✓ Heatmap saved to {save_path}")
    
    return fig


def plot_layer_importance_bar(results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Bar chart of layer importance (max causal effect per layer).
    """
    experiments = [e for e in results['experiments'] if 'error' not in e]
    
    # Group by layer
    layer_effects = {}
    for exp in experiments:
        layer = exp['layer']
        effect = abs(exp.get('causal_effect', 0))
        if layer not in layer_effects:
            layer_effects[layer] = []
        layer_effects[layer].append(effect)
    
    # Compute stats
    layer_stats = []
    for layer, effects in sorted(layer_effects.items()):
        layer_stats.append({
            'layer': layer,
            'max_effect': max(effects),
            'mean_effect': np.mean(effects),
            'num_experiments': len(effects)
        })
    
    df = pd.DataFrame(layer_stats)
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['layer'],
        y=df['max_effect'],
        name='Max Effect',
        marker_color='rgb(220, 50, 50)',
        text=df['max_effect'].round(2),
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        x=df['layer'],
        y=df['mean_effect'],
        name='Mean Effect',
        marker_color='rgb(100, 150, 200)',
        text=df['mean_effect'].round(2),
        textposition='inside'
    ))
    
    fig.update_layout(
        title={
            'text': "📊 Layer Importance Analysis<br><sub>Which layers have the strongest causal effects?</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Layer Index",
        yaxis_title="Causal Effect (Absolute)",
        barmode='group',
        width=1200,
        height=500,
        font=dict(size=12),
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✓ Layer importance chart saved to {save_path}")
    
    return fig


def plot_top_components_scatter(results: Dict[str, Any], top_n: int = 30, save_path: Optional[str] = None):
    """
    Scatter plot of top components by causal effect and KL divergence.
    """
    experiments = [e for e in results['experiments'] if 'error' not in e and 'causal_effect' in e]
    
    # Sort by causal effect
    experiments.sort(key=lambda x: abs(x['causal_effect']), reverse=True)
    top_experiments = experiments[:top_n]
    
    # Prepare data
    data = []
    for exp in top_experiments:
        data.append({
            'layer': exp['layer'],
            'component': exp['component'],
            'causal_effect': exp['causal_effect'],
            'kl_divergence': exp.get('kl_divergence', 0),
            'induced_refusal': exp.get('induced_refusal', False),
            'label': f"L{exp['layer']}-{exp['component']}"
        })
    
    df = pd.DataFrame(data)
    
    # Create scatter
    fig = px.scatter(
        df,
        x='causal_effect',
        y='kl_divergence',
        color='component',
        symbol='induced_refusal',
        size=abs(df['causal_effect']),
        hover_data=['layer', 'component'],
        text='label',
        title=f"🎯 Top {top_n} Causal Components<br><sub>Size = Effect Magnitude | Symbol = Induced Refusal</sub>",
        labels={
            'causal_effect': 'Causal Effect (Logit Difference)',
            'kl_divergence': 'KL Divergence',
            'component': 'Component Type'
        }
    )
    
    fig.update_traces(textposition='top center', textfont_size=8)
    fig.update_layout(
        width=1200,
        height=600,
        font=dict(size=12),
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✓ Scatter plot saved to {save_path}")
    
    return fig


def plot_refusal_cascade(results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Line plot showing how logit difference changes across layers.
    """
    experiments = [e for e in results['experiments'] if 'error' not in e]
    
    # Group by layer and component
    layer_data = {}
    for exp in experiments:
        layer = exp['layer']
        comp = exp['component']
        key = f"{comp}"
        
        if key not in layer_data:
            layer_data[key] = {'layers': [], 'effects': []}
        
        layer_data[key]['layers'].append(layer)
        layer_data[key]['effects'].append(exp.get('causal_effect', 0))
    
    # Create line plot
    fig = go.Figure()
    
    colors = {'attention': 'rgb(220, 50, 50)', 'mlp': 'rgb(50, 150, 220)', 'residual': 'rgb(100, 200, 100)'}
    
    for comp_type, data in layer_data.items():
        # Sort by layer
        sorted_pairs = sorted(zip(data['layers'], data['effects']))
        layers, effects = zip(*sorted_pairs) if sorted_pairs else ([], [])
        
        fig.add_trace(go.Scatter(
            x=list(layers),
            y=list(effects),
            mode='lines+markers',
            name=comp_type.capitalize(),
            line=dict(color=colors.get(comp_type, 'gray'), width=2),
            marker=dict(size=6)
        ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title={
            'text': "🌊 Refusal Cascade Across Layers<br><sub>How each component contributes to refusal behavior</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Layer Index",
        yaxis_title="Causal Effect (Logit Difference)",
        width=1200,
        height=500,
        font=dict(size=12),
        showlegend=True,
        hovermode='x unified'
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✓ Cascade plot saved to {save_path}")
    
    return fig


def plot_logit_stats_comparison(results: Dict[str, Any], save_path: Optional[str] = None):
    """
    Compare harmful, harmless, and patched logit statistics.
    """
    experiments = [e for e in results['experiments'] if 'error' not in e and 'harmful_stats' in e]
    
    if not experiments:
        print("⚠️ No detailed stats available")
        return None
    
    # Take first experiment for detailed view
    exp = max(experiments, key=lambda x: abs(x.get('causal_effect', 0)))
    
    stats_types = ['harmful_stats', 'harmless_stats', 'patched_stats']
    labels = ['Harmful (baseline)', 'Harmless (baseline)', 'Patched (harmful→harmless)']
    
    # Extract data
    logit_diffs = []
    refusal_probs = []
    compliance_probs = []
    
    for stat_type in stats_types:
        if stat_type in exp:
            stats = exp[stat_type]
            logit_diffs.append(stats.get('logit_diff', 0))
            refusal_probs.append(stats.get('refusal_prob_sum', 0))
            compliance_probs.append(stats.get('compliance_prob_sum', 0))
    
    # Create subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Logit Differences', 'Probability Mass'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Logit differences
    fig.add_trace(go.Bar(
        x=labels,
        y=logit_diffs,
        name='Logit Diff',
        marker_color=['red', 'green', 'orange'],
        text=[f"{v:+.2f}" for v in logit_diffs],
        textposition='outside'
    ), row=1, col=1)
    
    # Probability masses
    fig.add_trace(go.Bar(
        x=labels,
        y=refusal_probs,
        name='Refusal Tokens',
        marker_color='rgba(220, 50, 50, 0.7)'
    ), row=1, col=2)
    
    fig.add_trace(go.Bar(
        x=labels,
        y=compliance_probs,
        name='Compliance Tokens',
        marker_color='rgba(50, 150, 220, 0.7)'
    ), row=1, col=2)
    
    fig.update_layout(
        title={
            'text': f"📈 Detailed Logit Analysis<br><sub>Layer {exp['layer']} - {exp['component']} (strongest effect)</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        width=1200,
        height=500,
        font=dict(size=12),
        showlegend=True,
        barmode='group'
    )
    
    fig.update_yaxes(title_text="Logit Difference", row=1, col=1)
    fig.update_yaxes(title_text="Probability Mass", row=1, col=2)
    
    if save_path:
        fig.write_html(save_path)
        print(f"✓ Stats comparison saved to {save_path}")
    
    return fig


def plot_ransomware_analysis(analysis: Dict[str, Any], save_path: Optional[str] = None):
    """
    Visualize ransomware bypass analysis.
    """
    if not analysis or 'layer_differences' not in analysis:
        print("⚠️ No ransomware analysis available")
        return None
    
    layer_diffs = analysis['layer_differences']
    
    # Create bar chart of layer differences
    df = pd.DataFrame(layer_diffs)
    df = df.sort_values('l2_distance', ascending=False).head(20)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['layer'],
        y=df['l2_distance'],
        marker_color='rgb(200, 50, 100)',
        text=df['l2_distance'].round(2),
        textposition='outside'
    ))
    
    fig.update_layout(
        title={
            'text': f"🔓 Ransomware Bypass Analysis<br><sub>L2 Distance between Ransomware (bypassed) and Malware (refused)</sub><br>" + 
                    f"<sub>Bypass Gap: {analysis.get('bypass_gap', 0):+.4f} logits</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Layer Index",
        yaxis_title="L2 Distance (Activation Difference)",
        width=1200,
        height=500,
        font=dict(size=12)
    )
    
    # Add annotation
    fig.add_annotation(
        text=f"Ransomware: {analysis.get('ransomware_logit_diff', 0):+.3f} (complied)<br>" +
             f"Malware: {analysis.get('malware_logit_diff', 0):+.3f} (refused)",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="gray",
        borderwidth=1,
        font=dict(size=10)
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"✓ Ransomware analysis saved to {save_path}")
    
    return fig


def create_comprehensive_dashboard(
    all_results: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    bypass_analysis: Optional[Dict[str, Any]] = None,
    save_dir: str = "outputs/figures"
):
    """
    Create comprehensive dashboard with all visualizations for Colab.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("📊 CREATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60 + "\n")
    
    figures = {}
    
    # Use first result for detailed plots
    if len(all_results) > 0:
        result = all_results[0]
        
        # 1. Heatmap
        print("1️⃣ Creating causal effects heatmap...")
        figures['heatmap'] = plot_logit_effects_heatmap(
            result, 
            save_path=f"{save_dir}/01_causal_heatmap.html"
        )
        
        # 2. Layer importance
        print("2️⃣ Creating layer importance chart...")
        figures['layer_importance'] = plot_layer_importance_bar(
            result,
            save_path=f"{save_dir}/02_layer_importance.html"
        )
        
        # 3. Top components scatter
        print("3️⃣ Creating top components scatter...")
        figures['top_components'] = plot_top_components_scatter(
            result,
            top_n=30,
            save_path=f"{save_dir}/03_top_components.html"
        )
        
        # 4. Refusal cascade
        print("4️⃣ Creating refusal cascade plot...")
        figures['cascade'] = plot_refusal_cascade(
            result,
            save_path=f"{save_dir}/04_refusal_cascade.html"
        )
        
        # 5. Logit stats comparison
        print("5️⃣ Creating logit statistics comparison...")
        figures['logit_stats'] = plot_logit_stats_comparison(
            result,
            save_path=f"{save_dir}/05_logit_stats.html"
        )
    
    # 6. Ransomware analysis
    if bypass_analysis:
        print("6️⃣ Creating ransomware bypass analysis...")
        figures['ransomware'] = plot_ransomware_analysis(
            bypass_analysis,
            save_path=f"{save_dir}/06_ransomware_bypass.html"
        )
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS CREATED")
    print("="*60)
    print(f"\nSaved to: {save_dir}/")
    print("\nFiles:")
    for i, name in enumerate(['causal_heatmap', 'layer_importance', 'top_components', 
                               'refusal_cascade', 'logit_stats', 'ransomware_bypass'], 1):
        filename = f"{i:02d}_{name}.html"
        filepath = Path(save_dir) / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
    
    return figures


def display_in_colab(figures: Dict[str, Any]):
    """
    Display all figures inline in Google Colab.
    """
    try:
        from IPython.display import display
        
        print("\n" + "="*60)
        print("📺 DISPLAYING VISUALIZATIONS IN COLAB")
        print("="*60 + "\n")
        
        for name, fig in figures.items():
            if fig is not None:
                print(f"\n{'='*60}")
                print(f"{name.upper().replace('_', ' ')}")
                print(f"{'='*60}\n")
                display(fig)
    except ImportError:
        print("⚠️ Not in Colab environment, figures saved to files only")


def create_summary_table(analysis: Dict[str, Any]) -> pd.DataFrame:
    """
    Create summary table of top components for easy reading.
    """
    if 'top_30_components' not in analysis:
        return pd.DataFrame()
    
    top_comps = analysis['top_30_components'][:20]
    
    data = []
    for i, comp in enumerate(top_comps, 1):
        data.append({
            'Rank': i,
            'Layer': comp['layer'],
            'Component': comp['component'],
            'Head': comp.get('head', 'N/A'),
            'Causal Effect': f"{comp['causal_effect']:+.4f}",
            'KL Divergence': f"{comp.get('kl_divergence', 0):.4f}",
            'Induced Refusal': '✓' if comp.get('induced_refusal') else ''
        })
    
    return pd.DataFrame(data)
