"""
Experiment 1: Baseline Characterization
Test model behavior on harmful and harmless prompts.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import yaml
from model_utils import (
    load_model_and_tokenizer,
    generate_text,
    print_model_info,
    print_memory_stats
)
from patching import is_refusal
from tqdm import tqdm


def load_prompts(prompts_file: str = "data/prompts.json"):
    """Load contrastive prompts from JSON."""
    with open(prompts_file, 'r') as f:
        data = json.load(f)
    return data['prompt_pairs']


def run_baseline_experiment(config_path: str = "config.yaml"):
    """Run baseline characterization experiment."""
    
    print("="*80)
    print("EXPERIMENT 1: BASELINE CHARACTERIZATION")
    print("="*80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print("\n[1/4] Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        model_id=config['model']['name'],
        hf_token=config['model']['hf_token'],
        use_4bit=config['quantization']['load_in_4bit']
    )
    
    print_model_info(model)
    print_memory_stats("\nCurrent memory usage: ")
    
    # Load prompts
    print("\n[2/4] Loading prompts...")
    prompt_pairs = load_prompts()
    print(f"Loaded {len(prompt_pairs)} contrastive prompt pairs")
    
    # Test on harmful prompts
    print("\n[3/4] Testing on HARMFUL prompts...")
    harmful_results = []
    
    for pair in tqdm(prompt_pairs, desc="Harmful prompts"):
        prompt = pair['harmful']
        output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=config['experiment']['max_new_tokens'],
            temperature=config['experiment']['temperature']
        )
        
        result = {
            'id': pair['id'],
            'category': pair['category'],
            'prompt': prompt,
            'output': output,
            'is_refusal': is_refusal(output),
            'output_length': len(output.split())
        }
        harmful_results.append(result)
        
        # Print sample
        if pair['id'] <= 3:
            print(f"\n--- Prompt {pair['id']} ({pair['category']}) ---")
            print(f"Prompt: {prompt}")
            print(f"Output: {output[:200]}...")
            print(f"Refusal: {result['is_refusal']}")
    
    # Test on harmless prompts
    print("\n[4/4] Testing on HARMLESS prompts...")
    harmless_results = []
    
    for pair in tqdm(prompt_pairs, desc="Harmless prompts"):
        prompt = pair['harmless']
        output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=config['experiment']['max_new_tokens'],
            temperature=config['experiment']['temperature']
        )
        
        result = {
            'id': pair['id'],
            'category': pair['category'],
            'prompt': prompt,
            'output': output,
            'is_refusal': is_refusal(output),
            'output_length': len(output.split())
        }
        harmless_results.append(result)
        
        # Print sample
        if pair['id'] <= 3:
            print(f"\n--- Prompt {pair['id']} ({pair['category']}) ---")
            print(f"Prompt: {prompt}")
            print(f"Output: {output[:200]}...")
            print(f"Refusal: {result['is_refusal']}")
    
    # Analyze results
    print("\n" + "="*80)
    print("RESULTS ANALYSIS")
    print("="*80)
    
    harmful_refusal_rate = sum(r['is_refusal'] for r in harmful_results) / len(harmful_results)
    harmless_refusal_rate = sum(r['is_refusal'] for r in harmless_results) / len(harmless_results)
    
    print(f"\nHarmful Prompts:")
    print(f"  Refusal Rate: {harmful_refusal_rate*100:.1f}%")
    print(f"  Total Tested: {len(harmful_results)}")
    print(f"  Refused: {sum(r['is_refusal'] for r in harmful_results)}")
    print(f"  Complied: {sum(not r['is_refusal'] for r in harmful_results)}")
    
    print(f"\nHarmless Prompts:")
    print(f"  Refusal Rate: {harmless_refusal_rate*100:.1f}%")
    print(f"  Total Tested: {len(harmless_results)}")
    print(f"  Refused: {sum(r['is_refusal'] for r in harmless_results)}")
    print(f"  Complied: {sum(not r['is_refusal'] for r in harmless_results)}")
    
    # Category breakdown
    print(f"\nRefusal by Category (Harmful Prompts):")
    categories = {}
    for r in harmful_results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = {'total': 0, 'refused': 0}
        categories[cat]['total'] += 1
        if r['is_refusal']:
            categories[cat]['refused'] += 1
    
    for cat, stats in sorted(categories.items()):
        rate = stats['refused'] / stats['total'] * 100
        print(f"  {cat:.<25} {rate:>5.1f}% ({stats['refused']}/{stats['total']})")
    
    # Save results
    output_dir = Path(config['paths']['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'harmful_results': harmful_results,
        'harmless_results': harmless_results,
        'summary': {
            'harmful_refusal_rate': harmful_refusal_rate,
            'harmless_refusal_rate': harmless_refusal_rate,
            'total_prompts': len(prompt_pairs),
            'category_breakdown': categories
        }
    }
    
    results_path = output_dir / "01_baseline_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    print_memory_stats("\nFinal memory usage: ")
    
    return results


if __name__ == '__main__':
    try:
        results = run_baseline_experiment()
        print("\n" + "="*80)
        print("✓ BASELINE EXPERIMENT COMPLETED SUCCESSFULLY")
        print("="*80)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
