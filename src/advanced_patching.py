"""
Advanced activation patching with logit-based metrics.
Performs comprehensive layer and attention head scanning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
from tqdm import tqdm
from collections import defaultdict

from patching import (
    run_with_cache_logits_only,
    ActivationCache,
    ActivationHook,
    register_hooks,
    remove_hooks
)
from metrics import (
    compute_logit_diff,
    compute_logit_diff_distribution,
    compute_kl_divergence,
    get_token_ids,
    REFUSAL_TOKENS,
    COMPLIANCE_TOKENS
)


def patch_single_component_logits(
    model: nn.Module,
    tokenizer,
    harmful_prompt: str,
    harmless_prompt: str,
    layer_idx: int,
    component_type: str,
    head_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Patch a single component and measure logit difference.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        harmful_prompt: Source of activations (harmful)
        harmless_prompt: Target to patch (harmless)
        layer_idx: Layer to patch
        component_type: 'attention', 'mlp', 'residual', or 'attn_head'
        head_idx: If patching single attention head, specify index
    
    Returns:
        Results with logit metrics
    """
    device = next(model.parameters()).device
    
    # Get token IDs for refusal/compliance
    refusal_ids = get_token_ids(tokenizer, REFUSAL_TOKENS)
    compliance_ids = get_token_ids(tokenizer, COMPLIANCE_TOKENS)
    
    # 1. Run harmful prompt - get activations
    harmful_logits, harmful_cache = run_with_cache_logits_only(
        model, tokenizer, harmful_prompt
    )
    harmful_logit_diff = compute_logit_diff(
        harmful_logits, refusal_ids, compliance_ids, position=-1
    )
    harmful_stats = compute_logit_diff_distribution(
        harmful_logits, refusal_ids, compliance_ids, position=-1
    )
    
    # 2. Run harmless prompt - baseline
    harmless_logits, harmless_cache = run_with_cache_logits_only(
        model, tokenizer, harmless_prompt
    )
    harmless_logit_diff = compute_logit_diff(
        harmless_logits, refusal_ids, compliance_ids, position=-1
    )
    harmless_stats = compute_logit_diff_distribution(
        harmless_logits, refusal_ids, compliance_ids, position=-1
    )
    
    # 3. Get activation to patch
    if component_type == 'attention':
        if layer_idx not in harmful_cache.attention_out:
            return {'error': f'Layer {layer_idx} attention not in cache'}
        patch_activation = harmful_cache.attention_out[layer_idx]
    elif component_type == 'mlp':
        if layer_idx not in harmful_cache.mlp_out:
            return {'error': f'Layer {layer_idx} MLP not in cache'}
        patch_activation = harmful_cache.mlp_out[layer_idx]
    elif component_type == 'residual':
        if layer_idx not in harmful_cache.hidden_states:
            return {'error': f'Layer {layer_idx} residual not in cache'}
        patch_activation = harmful_cache.hidden_states[layer_idx]
    elif component_type == 'attn_head':
        # Patch single attention head
        if head_idx is None:
            return {'error': 'head_idx required for attn_head patching'}
        if layer_idx not in harmful_cache.attention_out:
            return {'error': f'Layer {layer_idx} attention not in cache'}
        # This requires head-specific output, which may need custom hook
        patch_activation = harmful_cache.attention_out[layer_idx]
    else:
        return {'error': f'Unknown component type: {component_type}'}
    
    # 4. Run harmless with patching
    class PatchHook:
        def __init__(self, activation):
            self.activation = activation
        
        def __call__(self, module, input, output):
            if isinstance(output, tuple):
                # Replace first element (main output)
                return (self.activation.to(output[0].device),) + output[1:]
            else:
                return self.activation.to(output.device)
    
    # Register patching hook
    if component_type == 'attention':
        layer = model.model.layers[layer_idx] if hasattr(model, 'model') else model.layers[layer_idx]
        hook_handle = layer.self_attn.register_forward_hook(PatchHook(patch_activation))
    elif component_type == 'mlp':
        layer = model.model.layers[layer_idx] if hasattr(model, 'model') else model.layers[layer_idx]
        hook_handle = layer.mlp.register_forward_hook(PatchHook(patch_activation))
    elif component_type in ['residual', 'attn_head']:
        layer = model.model.layers[layer_idx] if hasattr(model, 'model') else model.layers[layer_idx]
        hook_handle = layer.register_forward_hook(PatchHook(patch_activation))
    
    try:
        # Run with patching
        patched_logits, _ = run_with_cache_logits_only(
            model, tokenizer, harmless_prompt
        )
        patched_logit_diff = compute_logit_diff(
            patched_logits, refusal_ids, compliance_ids, position=-1
        )
        patched_stats = compute_logit_diff_distribution(
            patched_logits, refusal_ids, compliance_ids, position=-1
        )
    finally:
        hook_handle.remove()
    
    # 5. Compute causal effect
    causal_effect = patched_logit_diff - harmless_logit_diff
    kl_div = compute_kl_divergence(patched_logits, harmless_logits, position=-1)
    
    # Did patching induce refusal?
    induced_refusal = (harmless_logit_diff < 0) and (patched_logit_diff > 0)
    
    return {
        'layer': layer_idx,
        'component': component_type,
        'head': head_idx,
        # Baseline stats
        'harmful_logit_diff': harmful_logit_diff,
        'harmless_logit_diff': harmless_logit_diff,
        'patched_logit_diff': patched_logit_diff,
        # Causal metrics
        'causal_effect': causal_effect,
        'kl_divergence': kl_div,
        'induced_refusal': induced_refusal,
        # Detailed stats
        'harmful_stats': harmful_stats,
        'harmless_stats': harmless_stats,
        'patched_stats': patched_stats,
    }


def comprehensive_layer_scan(
    model: nn.Module,
    tokenizer,
    harmful_prompt: str,
    harmless_prompt: str,
    scan_heads: bool = True,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive scan of ALL layers and components.
    Optionally scans individual attention heads.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        harmful_prompt: Harmful prompt
        harmless_prompt: Harmless prompt
        scan_heads: If True, scan individual attention heads (slower)
        save_path: Path to save results
    
    Returns:
        Complete results dictionary
    """
    # Get model architecture
    num_layers = len(model.model.layers) if hasattr(model, 'model') else 32
    num_heads = model.config.num_attention_heads if hasattr(model, 'config') else 32
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE PATCHING SCAN")
    print(f"{'='*60}")
    print(f"Model layers: {num_layers}")
    print(f"Attention heads per layer: {num_heads}")
    print(f"Scan individual heads: {scan_heads}")
    print(f"{'='*60}\n")
    
    results = {
        'harmful_prompt': harmful_prompt,
        'harmless_prompt': harmless_prompt,
        'num_layers': num_layers,
        'num_heads': num_heads,
        'experiments': []
    }
    
    component_types = ['residual', 'attention', 'mlp']
    total_experiments = num_layers * len(component_types)
    
    if scan_heads:
        total_experiments += num_layers * num_heads
    
    print(f"Total experiments: {total_experiments}\n")
    
    with tqdm(total=total_experiments, desc="Scanning") as pbar:
        # Scan all layers for each component type
        for layer_idx in range(num_layers):
            for component_type in component_types:
                try:
                    result = patch_single_component_logits(
                        model, tokenizer,
                        harmful_prompt, harmless_prompt,
                        layer_idx, component_type
                    )
                    results['experiments'].append(result)
                except Exception as e:
                    results['experiments'].append({
                        'layer': layer_idx,
                        'component': component_type,
                        'error': str(e)
                    })
                
                pbar.update(1)
        
        # Optionally scan individual attention heads
        if scan_heads:
            print("\nScanning individual attention heads...")
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    try:
                        result = patch_single_component_logits(
                            model, tokenizer,
                            harmful_prompt, harmless_prompt,
                            layer_idx, 'attn_head', head_idx
                        )
                        results['experiments'].append(result)
                    except Exception as e:
                        results['experiments'].append({
                            'layer': layer_idx,
                            'component': 'attn_head',
                            'head': head_idx,
                            'error': str(e)
                        })
                    
                    pbar.update(1)
    
    # Save results
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {save_path}")
    
    return results


def analyze_patching_results_advanced(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Advanced analysis of patching results using logit metrics.
    
    Returns:
        Analysis with top causal components
    """
    experiments = [e for e in results['experiments'] if 'error' not in e]
    
    # Sort by causal effect
    sorted_by_effect = sorted(
        experiments,
        key=lambda x: abs(x.get('causal_effect', 0)),
        reverse=True
    )
    
    # Find components that induced refusal
    induced_refusal = [e for e in experiments if e.get('induced_refusal', False)]
    
    # Analyze by layer
    layer_effects = defaultdict(list)
    for exp in experiments:
        layer_effects[exp['layer']].append(exp.get('causal_effect', 0))
    
    layer_importance = {
        layer: {
            'mean_effect': np.mean(effects) if effects else 0,
            'max_effect': max(effects, key=abs) if effects else 0,
            'count': len(effects)
        }
        for layer, effects in layer_effects.items()
    }
    
    # Analyze by component type
    component_effects = defaultdict(list)
    for exp in experiments:
        component_effects[exp['component']].append(exp.get('causal_effect', 0))
    
    return {
        'total_experiments': len(experiments),
        'top_10_components': sorted_by_effect[:10],
        'top_20_components': sorted_by_effect[:20],
        'induced_refusal_count': len(induced_refusal),
        'induced_refusal_components': induced_refusal[:20],
        'layer_importance': dict(sorted(
            layer_importance.items(),
            key=lambda x: abs(x[1]['max_effect']),
            reverse=True
        )),
        'component_type_stats': {
            comp: {
                'mean': np.mean(effects),
                'max': max(effects, key=abs) if effects else 0,
                'count': len(effects)
            }
            for comp, effects in component_effects.items()
        }
    }


def compare_ransomware_vs_malware(
    model: nn.Module,
    tokenizer,
    ransomware_prompt: str,
    malware_prompt: str,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Special analysis: Compare activations for ransomware (bypassed) vs malware (refused).
    Identify why defense failed for ransomware.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        ransomware_prompt: Prompt that bypassed safety
        malware_prompt: Prompt that was refused
        save_path: Where to save analysis
    
    Returns:
        Comparison analysis
    """
    print(f"\n{'='*60}")
    print("RANSOMWARE vs MALWARE BYPASS ANALYSIS")
    print(f"{'='*60}\n")
    
    # Get logits and activations for both
    ransomware_logits, ransomware_cache = run_with_cache_logits_only(
        model, tokenizer, ransomware_prompt
    )
    malware_logits, malware_cache = run_with_cache_logits_only(
        model, tokenizer, malware_prompt
    )
    
    # Get logit stats
    refusal_ids = get_token_ids(tokenizer, REFUSAL_TOKENS)
    compliance_ids = get_token_ids(tokenizer, COMPLIANCE_TOKENS)
    
    ransomware_diff = compute_logit_diff(ransomware_logits, refusal_ids, compliance_ids)
    malware_diff = compute_logit_diff(malware_logits, refusal_ids, compliance_ids)
    
    print(f"Ransomware logit_diff: {ransomware_diff:.3f} ({'COMPLIED' if ransomware_diff < 0 else 'REFUSED'})")
    print(f"Malware logit_diff: {malware_diff:.3f} ({'COMPLIED' if malware_diff < 0 else 'REFUSED'})")
    print(f"\nBypass gap: {malware_diff - ransomware_diff:.3f}\n")
    
    # Compare activations layer by layer
    layer_differences = []
    
    for layer_idx in ransomware_cache.hidden_states.keys():
        if layer_idx not in malware_cache.hidden_states:
            continue
        
        ransomware_hidden = ransomware_cache.hidden_states[layer_idx]
        malware_hidden = malware_cache.hidden_states[layer_idx]
        
        # Compute L2 difference
        diff = (ransomware_hidden - malware_hidden).pow(2).sum().sqrt().item()
        
        layer_differences.append({
            'layer': layer_idx,
            'l2_distance': diff
        })
    
    # Sort by difference
    layer_differences.sort(key=lambda x: x['l2_distance'], reverse=True)
    
    analysis = {
        'ransomware_prompt': ransomware_prompt,
        'malware_prompt': malware_prompt,
        'ransomware_logit_diff': ransomware_diff,
        'malware_logit_diff': malware_diff,
        'bypass_gap': malware_diff - ransomware_diff,
        'layer_differences': layer_differences,
        'top_10_different_layers': layer_differences[:10]
    }
    
    print("Top 10 most different layers:")
    for ld in layer_differences[:10]:
        print(f"  Layer {ld['layer']:2d}: L2 distance = {ld['l2_distance']:.4f}")
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n✓ Analysis saved to {save_path}")
    
    return analysis
