"""
Experiment 3: Ablation Study
Test necessity of identified refusal components.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import yaml
from model_utils import load_model_and_tokenizer, print_memory_stats
from ablation import systematic_ablation_study, analyze_ablation_results
import plotly.graph_objects as go


def load_patching_results(results_path: str = "outputs/results/02_patching_results.json"):
    """Load patching experiment results."""
    with open(results_path, 'r') as f:
        return json.load(f)


def run_ablation_experiment(config_path: str = "config.yaml"):
    """Run ablation study."""
    
    print("="*80)
    print("EXPERIMENT 3: ABLATION STUDY")
    print("="*80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print("\n[1/5] Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        model_id=config['model']['name'],
        hf_token=None,  # Will prompt user
        use_4bit=config['quantization']['load_in_4bit']
    )
    print_memory_stats("Current memory: ")
    
    # Load patching results
    print("\n[2/4] Loading identified causal components...")
    try:
        patching_results = load_patching_results()
        raw_components = patching_results['combined_analysis']['ranked_components']
        
        # 🔥 ФИЛЬТР: Убираем residual stream из кандидатов на удаление
        causal_components = [
            c for c in raw_components 
            if c['component'] in ['mlp', 'attention'] # Оставляем только нейроны и внимание
        ]
        
        print(f"Found {len(raw_components)} components, filtered down to {len(causal_components)} (removed residuals)")
    except FileNotFoundError:
        print("Warning: Patching results not found. Using default components.")
        # Default to middle and late layers
        causal_components = [
            {'layer': i, 'component': 'attention'} 
            for i in [15, 16, 20, 24, 28]
        ] + [
            {'layer': i, 'component': 'mlp'} 
            for i in [15, 16, 20, 24, 28]
        ]
    
    # Prepare test prompts (harmful ones)
    print("\n[3/4] Preparing test prompts...")
    with open("data/prompts.json", 'r') as f:
        data = json.load(f)
    
    harmful_prompts = [p['harmful'] for p in data['prompt_pairs']]
    print(f"Testing with {len(harmful_prompts)} harmful prompts")
    
    # Run ablation study
    print("\n[4/4] Running systematic ablation study...")
    print("This will test ablating components individually and cumulatively...")
    
    results = systematic_ablation_study(
        model=model,
        tokenizer=tokenizer,
        causal_components=causal_components[:20],  # Top 20 components
        test_prompts=harmful_prompts,
        save_path="outputs/results/03_ablation_results.json"
    )
    
    # Analyze results
    print("\n" + "="*80)
    print("ABLATION ANALYSIS")
    print("="*80)
    
    analysis = analyze_ablation_results(results)
    
    print(f"\nBaseline Refusal Rate: {analysis['baseline_refusal_rate']*100:.1f}%")
    print(f"Total Ablation Experiments: {analysis['summary']['total_experiments']}")
    print(f"Maximum Refusal Rate Drop: {analysis['summary']['max_refusal_rate_drop']*100:.1f}%")
    
    if analysis['summary']['components_needed_for_50pct_drop']:
        print(f"Components needed for 50% reduction: "
              f"{analysis['summary']['components_needed_for_50pct_drop']}")
    
    # Print most effective individual ablations
    print(f"\nMost Effective Individual Component Ablations:")
    print(f"{'Layer':<10} {'Component':<15} {'Refusal Rate Drop':<20}")
    print("-" * 45)
    
    for exp in analysis['most_effective_individual'][:10]:
        layer = exp['ablated_components'][0]['layer'] if exp['ablated_components'] else 'N/A'
        comp = exp['ablated_components'][0]['component'] if exp['ablated_components'] else 'N/A'
        drop = exp.get('refusal_rate_change', 0) * 100
        print(f"{layer:<10} {comp:<15} {drop:.1f}%")
    
    # Plot cumulative results
    if analysis['cumulative_results']:
        print(f"\nCumulative Ablation Results:")
        print(f"{'Num Components':<18} {'Refusal Rate':<15} {'Drop from Baseline':<20}")
        print("-" * 53)
        
        for exp in analysis['cumulative_results']:
            num = exp.get('num_ablated', 0)
            rate = exp.get('refusal_rate', 0) * 100
            drop = exp.get('refusal_rate_change', 0) * 100
            print(f"{num:<18} {rate:.1f}%{'':<10} {drop:.1f}%")
        
        # Create visualization
        num_ablated = [exp.get('num_ablated', 0) for exp in analysis['cumulative_results']]
        refusal_rates = [exp.get('refusal_rate', 0) * 100 for exp in analysis['cumulative_results']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=num_ablated,
            y=refusal_rates,
            mode='lines+markers',
            name='Refusal Rate',
            line=dict(color='red', width=3),
            marker=dict(size=10)
        ))
        
        fig.add_hline(
            y=analysis['baseline_refusal_rate'] * 100,
            line_dash="dash",
            line_color="gray",
            annotation_text="Baseline"
        )
        
        fig.update_layout(
            title="Cumulative Ablation Effect on Refusal Rate",
            xaxis_title="Number of Ablated Components",
            yaxis_title="Refusal Rate (%)",
            height=500,
            width=800
        )
        
        output_dir = Path(config['paths']['figures_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_dir / "ablation_cumulative.html"))
        print(f"\n✓ Saved cumulative ablation plot")
    
    # Save analysis
    results_dir = Path(config['paths']['results_dir'])
    analysis_path = results_dir / "03_ablation_analysis.json"
    
    with open(analysis_path, 'w') as f:
        # Make JSON-serializable
        save_data = {
            'baseline_refusal_rate': analysis['baseline_refusal_rate'],
            'summary': analysis['summary'],
            'top_10_most_effective': [
                {
                    'ablated_components': exp['ablated_components'],
                    'refusal_rate_change': exp.get('refusal_rate_change', 0)
                }
                for exp in analysis['most_effective_individual'][:10]
            ]
        }
        json.dump(save_data, f, indent=2)
    
    print(f"\n✓ Analysis saved to: {analysis_path}")
    print_memory_stats("\nFinal memory: ")
    
    return results, analysis


if __name__ == '__main__':
    try:
        results, analysis = run_ablation_experiment()
        print("\n" + "="*80)
        print("✓ ABLATION STUDY COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nKey Insights:")
        print(f"- Baseline refusal rate: {analysis['baseline_refusal_rate']*100:.1f}%")
        if analysis['summary']['max_refusal_rate_drop'] > 0:
            print(f"- Maximum reduction achieved: {analysis['summary']['max_refusal_rate_drop']*100:.1f}%")
            print(f"- This proves identified components are causally responsible for refusal")
        else:
            print("- Note: Limited refusal reduction observed")
            print("- May need to refine component identification or try different ablation strategies")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
