"""
Experiment 2: Activation Patching
Identify refusal circuits through activation patching.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import yaml
from model_utils import load_model_and_tokenizer, print_memory_stats
from patching import (
    batch_patching_experiment,
    analyze_patching_results,
    run_with_cache
)
from visualization import (
    plot_patching_results,
    plot_layer_importance,
    create_circuit_diagram,
    create_dashboard_summary
)


def load_baseline_results(results_path: str = "outputs/results/01_baseline_results.json"):
    """Load baseline experiment results."""
    with open(results_path, 'r') as f:
        return json.load(f)


def select_test_pairs(baseline_results, num_pairs: int = 5):
    """Select prompt pairs for patching experiments."""
    harmful = baseline_results['harmful_results']
    harmless = baseline_results['harmless_results']
    
    # Select pairs where harmful refused and harmless complied
    selected = []
    for h, hl in zip(harmful, harmless):
        if h['is_refusal'] and not hl['is_refusal']:
            selected.append({
                'harmful': h['prompt'],
                'harmless': hl['prompt'],
                'category': h['category']
            })
            if len(selected) >= num_pairs:
                break
    
    return selected


def run_patching_experiment(config_path: str = "config.yaml"):
    """Run activation patching experiment."""
    
    print("="*80)
    print("EXPERIMENT 2: ACTIVATION PATCHING")
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
    
    # Load baseline results
    print("\n[2/5] Loading baseline results...")
    try:
        baseline = load_baseline_results()
        test_pairs = select_test_pairs(baseline, num_pairs=5)
        print(f"Selected {len(test_pairs)} prompt pairs for testing")
    except FileNotFoundError:
        print("Warning: Baseline results not found, using default pairs")
        # Load from prompts.json
        with open("data/prompts.json", 'r') as f:
            data = json.load(f)
        test_pairs = [
            {'harmful': p['harmful'], 'harmless': p['harmless'], 'category': p['category']}
            for p in data['prompt_pairs'][:5]
        ]
    
    # Run patching experiments
    print("\n[3/5] Running activation patching experiments...")
    print("This will test each layer and component type...")
    
    all_results = []
    
    for i, pair in enumerate(test_pairs):
        print(f"\n--- Pair {i+1}/{len(test_pairs)}: {pair['category']} ---")
        print(f"Harmful: {pair['harmful'][:60]}...")
        print(f"Harmless: {pair['harmless'][:60]}...")
        
        # Run batch patching
        results = batch_patching_experiment(
            model=model,
            tokenizer=tokenizer,
            harmful_prompt=pair['harmful'],
            harmless_prompt=pair['harmless'],
            layers_to_test=list(range(0, 32, 2)),  # Test every 2nd layer to save time
            component_types=['residual', 'attention', 'mlp'],
            max_new_tokens=50,
            save_path=None
        )
        
        results['category'] = pair['category']
        results['pair_id'] = i
        all_results.append(results)
    
    # Analyze results
    print("\n[4/5] Analyzing results...")
    
    combined_analysis = {
        'per_pair_analysis': [],
        'aggregated_causal_components': []
    }
    
    for result in all_results:
        analysis = analyze_patching_results(result)
        combined_analysis['per_pair_analysis'].append({
            'category': result['category'],
            'pair_id': result['pair_id'],
            'analysis': analysis
        })
        
        # Aggregate causal components
        combined_analysis['aggregated_causal_components'].extend(
            analysis['causal_components']
        )
    
    # Count occurrences of each component
    from collections import Counter
    component_counts = Counter()
    for comp in combined_analysis['aggregated_causal_components']:
        key = (comp['layer'], comp['component'])
        component_counts[key] += 1
    
    # Rank components by frequency
    ranked_components = [
        {
            'layer': layer,
            'component': comp,
            'frequency': count,
            'percentage': count / len(test_pairs) * 100
        }
        for (layer, comp), count in component_counts.most_common(20)
    ]
    
    combined_analysis['ranked_components'] = ranked_components
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    print(f"\nTotal causal components found: {len(combined_analysis['aggregated_causal_components'])}")
    print(f"Unique components: {len(component_counts)}")
    
    print(f"\nTop 10 Most Consistent Refusal Components:")
    print(f"{'Layer':<10} {'Component':<15} {'Frequency':<12} {'Percentage':<12}")
    print("-" * 50)
    for comp in ranked_components[:10]:
        print(f"{comp['layer']:<10} {comp['component']:<15} "
              f"{comp['frequency']:<12} {comp['percentage']:.1f}%")
    
    # Layer importance
    layer_importance = {}
    for comp in combined_analysis['aggregated_causal_components']:
        layer = comp['layer']
        layer_importance[layer] = layer_importance.get(layer, 0) + 1
    
    print(f"\nMost Important Layers:")
    for layer, count in sorted(layer_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  Layer {layer}: {count} causal components")
    
    # Generate visualizations
    print("\n[5/5] Generating visualizations...")
    
    output_dir = Path(config['paths']['figures_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot for first pair (most representative)
    if all_results:
        fig1 = plot_patching_results(
            all_results[0],
            save_path=str(output_dir / "patching_heatmap.html")
        )
        print(f"✓ Saved patching heatmap")
        
        fig2 = plot_layer_importance(
            layer_importance,
            save_path=str(output_dir / "layer_importance.html")
        )
        print(f"✓ Saved layer importance plot")
        
        fig3 = create_circuit_diagram(
            ranked_components[:15],
            save_path=str(output_dir / "circuit_diagram.html")
        )
        print(f"✓ Saved circuit diagram")
        
        # Create dashboard
        analysis_sample = analyze_patching_results(all_results[0])
        dashboard_path = create_dashboard_summary(
            all_results[0],
            analysis_sample,
            output_dir=str(output_dir)
        )
        print(f"✓ Created interactive dashboard: {dashboard_path}")
    
    # Save results
    results_dir = Path(config['paths']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = results_dir / "02_patching_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'all_experiments': [
                {
                    'category': r['category'],
                    'pair_id': r['pair_id'],
                    'harmful_prompt': r['harmful_prompt'],
                    'harmless_prompt': r['harmless_prompt'],
                    'num_experiments': len(r['experiments'])
                }
                for r in all_results
            ],
            'combined_analysis': combined_analysis
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    print_memory_stats("\nFinal memory: ")
    
    return all_results, combined_analysis


if __name__ == '__main__':
    try:
        results, analysis = run_patching_experiment()
        print("\n" + "="*80)
        print("✓ PATCHING EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*80)
        print("\nNext steps:")
        print("1. Review the interactive dashboard in outputs/figures/dashboard.html")
        print("2. Examine identified refusal components")
        print("3. Run ablation study with: python experiments/03_ablation.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
