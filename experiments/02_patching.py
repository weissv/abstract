"""
Experiment 2: Activation Patching with Logit-Based Metrics
Comprehensive scan of ALL layers and attention heads.
Identifies refusal circuits using precise logit difference measurements.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import yaml
import numpy as np
from model_utils import load_model_and_tokenizer, print_memory_stats
from advanced_patching import (
    comprehensive_layer_scan,
    analyze_patching_results_advanced,
    compare_ransomware_vs_malware
)
from colab_visualization import (
    create_comprehensive_dashboard,
    display_in_colab,
    create_summary_table
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
    """
    Run comprehensive activation patching experiment with logit-based metrics.
    Scans ALL 32 layers and optionally individual attention heads.
    """
    
    print("="*80)
    print("EXPERIMENT 2: COMPREHENSIVE ACTIVATION PATCHING")
    print("With Logit-Based Metrics and Full Layer Scanning")
    print("="*80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print("\n[1/6] Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        model_id=config['model']['name'],
        hf_token=None,  # Will use auto-detection
        use_4bit=config['quantization']['load_in_4bit']
    )
    print_memory_stats("Current memory: ")
    
    # Load baseline results
    print("\n[2/6] Loading baseline results...")
    try:
        baseline = load_baseline_results()
        test_pairs = select_test_pairs(baseline, num_pairs=3)
        print(f"Selected {len(test_pairs)} prompt pairs for testing")
    except FileNotFoundError:
        print("Warning: Baseline results not found, using default pairs")
        with open("data/prompts.json", 'r') as f:
            data = json.load(f)
        test_pairs = [
            {'harmful': p['harmful'], 'harmless': p['harmless'], 'category': p['category']}
            for p in data['prompt_pairs'][:3]
        ]
    
    # Run comprehensive patching experiments
    print("\n[3/6] Running COMPREHENSIVE patching experiments...")
    print("⚡ This scans ALL 32 layers with logit-based metrics")
    print("⚡ Much more precise than text-based classification\n")
    
    all_results = []
    
    for i, pair in enumerate(test_pairs):
        print(f"\n{'='*60}")
        print(f"Pair {i+1}/{len(test_pairs)}: {pair['category']}")
        print(f"{'='*60}")
        print(f"Harmful: {pair['harmful'][:70]}...")
        print(f"Harmless: {pair['harmless'][:70]}...\n")
        
        # Run comprehensive scan (ALL layers, ALL components)
        results = comprehensive_layer_scan(
            model=model,
            tokenizer=tokenizer,
            harmful_prompt=pair['harmful'],
            harmless_prompt=pair['harmless'],
            scan_heads=False,  # Set to True to scan individual heads (very slow!)
            save_path=f"outputs/results/02_patching_pair_{i+1}.json"
        )
        
        results['category'] = pair['category']
        results['pair_id'] = i
        all_results.append(results)
    
    # Special: Ransomware vs Malware analysis
    print("\n[4/6] Running RANSOMWARE BYPASS ANALYSIS...")
    print("Investigating why ransomware prompt bypassed safety...\n")
    
    # Find ransomware and malware prompts from baseline
    ransomware_prompt = None
    malware_prompt = None
    
    try:
        for pair in test_pairs:
            if 'ransomware' in pair['harmful'].lower():
                ransomware_prompt = pair['harmful']
            elif 'malware' in pair['harmful'].lower():
                malware_prompt = pair['harmful']
        
        if ransomware_prompt and malware_prompt:
            bypass_analysis = compare_ransomware_vs_malware(
                model=model,
                tokenizer=tokenizer,
                ransomware_prompt=ransomware_prompt,
                malware_prompt=malware_prompt,
                save_path="outputs/results/02_ransomware_bypass_analysis.json"
            )
        else:
            print("⚠️ Ransomware/Malware prompts not found, skipping bypass analysis")
            bypass_analysis = None
    except Exception as e:
        print(f"⚠️ Bypass analysis failed: {e}")
        bypass_analysis = None
    
    # Analyze results
    print("\n[5/6] Analyzing results with advanced metrics...")
    
    combined_analysis = {
        'per_pair_analysis': [],
        'aggregated_components': [],
        'bypass_analysis': bypass_analysis
    }
    
    for result in all_results:
        analysis = analyze_patching_results_advanced(result)
        combined_analysis['per_pair_analysis'].append({
            'category': result['category'],
            'pair_id': result['pair_id'],
            'analysis': analysis
        })
        
        # Collect top components
        combined_analysis['aggregated_components'].extend(
            analysis['top_20_components']
        )
    
    # Rank by causal effect magnitude
    all_components = []
    for result in all_results:
        for exp in result['experiments']:
            if 'error' not in exp and 'causal_effect' in exp:
                all_components.append({
                    'layer': exp['layer'],
                    'component': exp['component'],
                    'head': exp.get('head'),
                    'causal_effect': exp['causal_effect'],
                    'kl_divergence': exp.get('kl_divergence', 0),
                    'induced_refusal': exp.get('induced_refusal', False)
                })
    
    # Sort by absolute causal effect
    all_components.sort(key=lambda x: abs(x['causal_effect']), reverse=True)
    
    # Save combined results
    combined_results = {
        'all_pair_results': all_results,
        'analysis': combined_analysis,
        'top_30_components': all_components[:30]
    }
    
    with open("outputs/results/02_patching_combined.json", 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    # Print key findings
    print("\n" + "="*80)
    print("KEY FINDINGS (Logit-Based Metrics)")
    print("="*80)
    
    print(f"\n📊 Total components tested: {len(all_components)}")
    induced = [c for c in all_components if c['induced_refusal']]
    print(f"🎯 Components that INDUCED REFUSAL: {len(induced)}")
    
    print(f"\n🔥 Top 15 Causal Components (by logit difference):")
    print(f"{'Layer':<8} {'Component':<12} {'Head':<6} {'Effect':<12} {'KL Div':<10} {'Induced?':<10}")
    print("-" * 70)
    for comp in all_components[:15]:
        head_str = str(comp['head']) if comp['head'] is not None else 'N/A'
        induced_str = '✓ YES' if comp['induced_refusal'] else ''
        print(f"{comp['layer']:<8} {comp['component']:<12} {head_str:<6} "
              f"{comp['causal_effect']:>+11.4f} {comp['kl_divergence']:>9.4f} {induced_str:<10}")
    
    # Layer-wise summary
    layer_stats = {}
    for comp in all_components:
        layer = comp['layer']
        if layer not in layer_stats:
            layer_stats[layer] = {'effects': [], 'induced_count': 0}
        layer_stats[layer]['effects'].append(abs(comp['causal_effect']))
        if comp['induced_refusal']:
            layer_stats[layer]['induced_count'] += 1
    
    layer_importance = [
        {
            'layer': layer,
            'mean_effect': np.mean(stats['effects']),
            'max_effect': max(stats['effects']),
            'induced_count': stats['induced_count']
        }
        for layer, stats in layer_stats.items()
    ]
    layer_importance.sort(key=lambda x: x['max_effect'], reverse=True)
    
    print(f"\n📍 Top 10 Most Important Layers:")
    print(f"{'Layer':<8} {'Mean Effect':<15} {'Max Effect':<15} {'Induced #':<12}")
    print("-" * 50)
    for layer_info in layer_importance[:10]:
        print(f"{layer_info['layer']:<8} {layer_info['mean_effect']:>14.4f} "
              f"{layer_info['max_effect']:>14.4f} {layer_info['induced_count']:>11}")
    
    # Ransomware analysis summary
    if bypass_analysis:
        print(f"\n🔓 RANSOMWARE BYPASS ANALYSIS:")
        print(f"   Ransomware logit_diff: {bypass_analysis['ransomware_logit_diff']:+.4f} (bypassed)")
        print(f"   Malware logit_diff:    {bypass_analysis['malware_logit_diff']:+.4f} (refused)")
        print(f"   Bypass gap:            {bypass_analysis['bypass_gap']:+.4f}")
        print(f"\n   Top 5 layers with largest activation differences:")
        for ld in bypass_analysis['top_10_different_layers'][:5]:
            print(f"     Layer {ld['layer']:2d}: L2 distance = {ld['l2_distance']:.4f}")
    
    # Visualization
    print("\n[6/6] Generating comprehensive visualizations...")
    try:
        # Create comprehensive dashboard with all visualizations
        figures = create_comprehensive_dashboard(
            all_results=all_results,
            analysis=combined_analysis,
            bypass_analysis=bypass_analysis,
            save_dir="outputs/figures"
        )
        
        # Display in Colab if available
        try:
            display_in_colab(figures)
        except:
            print("✓ Figures saved to files (not in Colab environment)")
        
        # Create summary table
        if 'top_30_components' in combined_results:
            summary_df = create_summary_table(combined_results)
            print("\n📋 TOP 20 COMPONENTS SUMMARY TABLE:")
            print(summary_df.to_string(index=False))
            
            # Save to CSV
            summary_df.to_csv("outputs/results/02_top_components_summary.csv", index=False)
            print("\n✓ Summary table saved to outputs/results/02_top_components_summary.csv")
        
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT 2 COMPLETE")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  - outputs/results/02_patching_combined.json")
    print(f"  - outputs/results/02_patching_pair_*.json")
    print(f"  - outputs/results/02_top_components_summary.csv")
    if bypass_analysis:
        print(f"  - outputs/results/02_ransomware_bypass_analysis.json")
    print(f"\nVisualizations saved to outputs/figures/:")
    print(f"  - 01_causal_heatmap.html")
    print(f"  - 02_layer_importance.html")
    print(f"  - 03_top_components.html")
    print(f"  - 04_refusal_cascade.html")
    print(f"  - 05_logit_stats.html")
    if bypass_analysis:
        print(f"  - 06_ransomware_bypass.html")
    
    return combined_results, combined_analysis


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
