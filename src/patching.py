"""
Activation patching utilities for mechanistic interpretability.
Implements causal tracing to identify refusal circuits in Llama-3.
Uses logit-based metrics for more precise measurement.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from collections import defaultdict
import json
from pathlib import Path
from tqdm import tqdm

# Import logit-based metrics
from metrics import (
    compute_logit_diff,
    compute_logit_diff_distribution,
    compute_kl_divergence,
    compute_js_divergence,
    get_token_ids,
    analyze_first_token_distribution,
    is_refusal_by_logits,
    REFUSAL_TOKENS,
    COMPLIANCE_TOKENS
)


@dataclass
class ActivationCache:
    """Store activations from a model forward pass."""
    
    residual_stream: Dict[int, torch.Tensor]  # Layer -> residual activations
    attention_out: Dict[int, torch.Tensor]    # Layer -> attention output
    mlp_out: Dict[int, torch.Tensor]          # Layer -> MLP output
    attention_pattern: Dict[int, torch.Tensor] # Layer -> attention patterns
    hidden_states: Dict[int, torch.Tensor]    # Layer -> hidden states
    
    def __init__(self):
        self.residual_stream = {}
        self.attention_out = {}
        self.mlp_out = {}
        self.attention_pattern = {}
        self.hidden_states = {}
    
    def save(self, path: str):
        """Save cache to disk."""
        cache_dict = {
            'residual_stream': {k: v.cpu() for k, v in self.residual_stream.items()},
            'attention_out': {k: v.cpu() for k, v in self.attention_out.items()},
            'mlp_out': {k: v.cpu() for k, v in self.mlp_out.items()},
            'attention_pattern': {k: v.cpu() for k, v in self.attention_pattern.items()},
            'hidden_states': {k: v.cpu() for k, v in self.hidden_states.items()},
        }
        torch.save(cache_dict, path)
    
    @classmethod
    def load(cls, path: str):
        """Load cache from disk."""
        cache_dict = torch.load(path)
        cache = cls()
        cache.residual_stream = cache_dict.get('residual_stream', {})
        cache.attention_out = cache_dict.get('attention_out', {})
        cache.mlp_out = cache_dict.get('mlp_out', {})
        cache.attention_pattern = cache_dict.get('attention_pattern', {})
        cache.hidden_states = cache_dict.get('hidden_states', {})
        return cache


class ActivationHook:
    """Hook to capture activations during forward pass."""
    
    def __init__(self, cache: ActivationCache, layer_idx: int, component_type: str):
        self.cache = cache
        self.layer_idx = layer_idx
        self.component_type = component_type
        self.activation = None
    
    def __call__(self, module, input, output):
        """Hook function called during forward pass."""
        # Store activation based on component type
        if self.component_type == 'residual':
            # For residual stream, we want the hidden states
            if isinstance(output, tuple):
                self.cache.hidden_states[self.layer_idx] = output[0].detach()
            else:
                self.cache.hidden_states[self.layer_idx] = output.detach()
        
        elif self.component_type == 'attention':
            # For attention, capture the output and patterns
            if isinstance(output, tuple):
                self.cache.attention_out[self.layer_idx] = output[0].detach()
                # Attention patterns are typically in output[1] or output[2]
                if len(output) > 2 and output[2] is not None:
                    self.cache.attention_pattern[self.layer_idx] = output[2].detach()
            else:
                self.cache.attention_out[self.layer_idx] = output.detach()
        
        elif self.component_type == 'mlp':
            # For MLP, capture the output
            if isinstance(output, tuple):
                self.cache.mlp_out[self.layer_idx] = output[0].detach()
            else:
                self.cache.mlp_out[self.layer_idx] = output.detach()
        
        self.activation = output
        return output


class PatchingHook:
    """Hook to patch activations during forward pass."""
    
    def __init__(self, patch_activation: torch.Tensor, layer_idx: int):
        self.patch_activation = patch_activation
        self.layer_idx = layer_idx
    
    def __call__(self, module, input, output):
        """Replace activation with patched version."""
        # Handle tuple outputs (common in transformer layers)
        if isinstance(output, tuple):
            # Replace the main activation tensor
            output_list = list(output)
            output_list[0] = self.patch_activation
            return tuple(output_list)
        else:
            return self.patch_activation


def register_hooks(model: nn.Module, cache: ActivationCache) -> List:
    """
    Register forward hooks to capture activations.
    
    Args:
        model: PyTorch model
        cache: ActivationCache to store activations
    
    Returns:
        List of hook handles
    """
    hooks = []
    
    # For Llama models, we need to hook into specific layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        
        for layer_idx, layer in enumerate(layers):
            # Hook residual stream (layer output)
            hook = layer.register_forward_hook(
                ActivationHook(cache, layer_idx, 'residual')
            )
            hooks.append(hook)
            
            # Hook attention
            if hasattr(layer, 'self_attn'):
                hook = layer.self_attn.register_forward_hook(
                    ActivationHook(cache, layer_idx, 'attention')
                )
                hooks.append(hook)
            
            # Hook MLP
            if hasattr(layer, 'mlp'):
                hook = layer.mlp.register_forward_hook(
                    ActivationHook(cache, layer_idx, 'mlp')
                )
                hooks.append(hook)
    
    return hooks


def remove_hooks(hooks: List):
    """Remove all registered hooks."""
    for hook in hooks:
        hook.remove()


def run_with_cache_logits_only(
    model: nn.Module,
    tokenizer,
    prompt: str,
) -> Tuple[torch.Tensor, ActivationCache]:
    """
    Run model WITHOUT generation - just get logits for next token.
    Much faster and more precise for patching experiments.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
    
    Returns:
        Tuple of (logits, activation_cache)
        logits shape: [batch, seq_len, vocab_size]
    """
    cache = ActivationCache()
    hooks = register_hooks(model, cache)
    
    try:
        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, 'apply_chat_template'):
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Single forward pass (no generation)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # [batch, seq_len, vocab_size]
        
    finally:
        remove_hooks(hooks)
    
    return logits, cache


def run_with_cache(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7
) -> Tuple[str, ActivationCache, torch.Tensor]:
    """
    Run model and cache all activations (with generation).
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Tuple of (generated_text, activation_cache, logits)
    """
    cache = ActivationCache()
    hooks = register_hooks(model, cache)
    
    try:
        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, 'apply_chat_template'):
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Get logits
        logits = torch.stack(outputs.scores, dim=1) if hasattr(outputs, 'scores') else None
        
        # Decode
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        if formatted_prompt in generated_text:
            generated_text = generated_text[len(formatted_prompt):].strip()
        
    finally:
        remove_hooks(hooks)
    
    return generated_text, cache, logits


def compute_activation_diff(
    cache_harmful: ActivationCache,
    cache_harmless: ActivationCache,
    component_type: str = 'all'
) -> Dict[int, torch.Tensor]:
    """
    Compute difference in activations between harmful and harmless prompts.
    
    Args:
        cache_harmful: Activations from harmful prompt
        cache_harmless: Activations from harmless prompt
        component_type: 'all', 'attention', 'mlp', or 'residual'
    
    Returns:
        Dictionary mapping layer_idx -> activation difference
    """
    diff = {}
    
    if component_type in ['all', 'attention']:
        for layer_idx in cache_harmful.attention_out.keys():
            if layer_idx in cache_harmless.attention_out:
                diff[f'attn_{layer_idx}'] = (
                    cache_harmful.attention_out[layer_idx] - 
                    cache_harmless.attention_out[layer_idx]
                )
    
    if component_type in ['all', 'mlp']:
        for layer_idx in cache_harmful.mlp_out.keys():
            if layer_idx in cache_harmless.mlp_out:
                diff[f'mlp_{layer_idx}'] = (
                    cache_harmful.mlp_out[layer_idx] - 
                    cache_harmless.mlp_out[layer_idx]
                )
    
    if component_type in ['all', 'residual']:
        for layer_idx in cache_harmful.hidden_states.keys():
            if layer_idx in cache_harmless.hidden_states:
                diff[f'resid_{layer_idx}'] = (
                    cache_harmful.hidden_states[layer_idx] - 
                    cache_harmless.hidden_states[layer_idx]
                )
    
    return diff


def activation_patching_experiment(
    model: nn.Module,
    tokenizer,
    harmful_prompt: str,
    harmless_prompt: str,
    layer_to_patch: int,
    component_type: str = 'residual',
    max_new_tokens: int = 50
) -> Dict[str, Any]:
    """
    Perform activation patching experiment.
    
    This swaps activations from harmful prompt into harmless prompt at a specific layer
    to test if the model output changes from helpful -> refusal.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        harmful_prompt: Harmful prompt (source of activations)
        harmless_prompt: Harmless prompt (target for patching)
        layer_to_patch: Which layer to patch
        component_type: 'attention', 'mlp', or 'residual'
        max_new_tokens: Max tokens to generate
    
    Returns:
        Dictionary with results
    """
    # 1. Run harmful prompt and cache activations
    print(f"Running harmful prompt: '{harmful_prompt[:50]}...'")
    harmful_output, harmful_cache, harmful_logits = run_with_cache(
        model, tokenizer, harmful_prompt, max_new_tokens=max_new_tokens
    )
    
    # 2. Run harmless prompt and cache activations (baseline)
    print(f"Running harmless prompt: '{harmless_prompt[:50]}...'")
    harmless_output, harmless_cache, harmless_logits = run_with_cache(
        model, tokenizer, harmless_prompt, max_new_tokens=max_new_tokens
    )
    
    # 3. Get the activation to patch
    if component_type == 'attention':
        if layer_to_patch not in harmful_cache.attention_out:
            raise ValueError(f"Layer {layer_to_patch} not in cache")
        patch_activation = harmful_cache.attention_out[layer_to_patch]
    elif component_type == 'mlp':
        if layer_to_patch not in harmful_cache.mlp_out:
            raise ValueError(f"Layer {layer_to_patch} not in cache")
        patch_activation = harmful_cache.mlp_out[layer_to_patch]
    else:  # residual
        if layer_to_patch not in harmful_cache.hidden_states:
            raise ValueError(f"Layer {layer_to_patch} not in cache")
        patch_activation = harmful_cache.hidden_states[layer_to_patch]
    
    # 4. Run harmless prompt with patched activation
    print(f"Running patched inference (layer {layer_to_patch}, {component_type})")
    
    # Register patching hook
    target_layer = model.model.layers[layer_to_patch]
    if component_type == 'attention':
        target_module = target_layer.self_attn
    elif component_type == 'mlp':
        target_module = target_layer.mlp
    else:
        target_module = target_layer
    
    patch_hook = target_module.register_forward_hook(
        PatchingHook(patch_activation, layer_to_patch)
    )
    
    try:
        patched_output, patched_cache, patched_logits = run_with_cache(
            model, tokenizer, harmless_prompt, max_new_tokens=max_new_tokens
        )
    finally:
        patch_hook.remove()
    
    # 5. Analyze results
    results = {
        'harmful_output': harmful_output,
        'harmless_output': harmless_output,
        'patched_output': patched_output,
        'layer_patched': layer_to_patch,
        'component_type': component_type,
        'is_refusal_harmful': is_refusal(harmful_output),
        'is_refusal_harmless': is_refusal(harmless_output),
        'is_refusal_patched': is_refusal(patched_output),
        'patching_effect': 'harmful->refusal' if is_refusal(patched_output) and not is_refusal(harmless_output) else 'no effect'
    }
    
    return results


def is_refusal(text: str) -> bool:
    """Robust refusal detection."""
    t = text.lower()
    # Список стоп-фраз с вариациями апострофов
    keywords = [
        "i cannot", "i can't", "i can’t",  # <--- ВАЖНО: кривой апостроф
        "i am unable", "i'm unable", "i’m unable",
        "i apologize", "i cannot assist", "cannot comply",
        "against my programming", "violates my safety",
        "harmful activities", "illegal acts"
    ]
    return any(k in t for k in keywords)


def batch_patching_experiment(
    model: nn.Module,
    tokenizer,
    harmful_prompt: str,
    harmless_prompt: str,
    layers_to_test: Optional[List[int]] = None,
    component_types: List[str] = ['residual', 'attention', 'mlp'],
    max_new_tokens: int = 50,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run patching experiments across multiple layers and components.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        harmful_prompt: Harmful prompt
        harmless_prompt: Harmless prompt
        layers_to_test: List of layer indices to test (None = all layers)
        component_types: Types of components to patch
        max_new_tokens: Max tokens to generate
        save_path: Path to save results
    
    Returns:
        Dictionary with all results
    """
    # Get number of layers
    num_layers = len(model.model.layers) if hasattr(model, 'model') else 32
    
    if layers_to_test is None:
        layers_to_test = list(range(num_layers))
    
    all_results = {
        'harmful_prompt': harmful_prompt,
        'harmless_prompt': harmless_prompt,
        'experiments': []
    }
    
    total_experiments = len(layers_to_test) * len(component_types)
    
    with tqdm(total=total_experiments, desc="Patching experiments") as pbar:
        for layer_idx in layers_to_test:
            for component_type in component_types:
                try:
                    result = activation_patching_experiment(
                        model=model,
                        tokenizer=tokenizer,
                        harmful_prompt=harmful_prompt,
                        harmless_prompt=harmless_prompt,
                        layer_to_patch=layer_idx,
                        component_type=component_type,
                        max_new_tokens=max_new_tokens
                    )
                    all_results['experiments'].append(result)
                    
                except Exception as e:
                    print(f"Error in layer {layer_idx}, {component_type}: {e}")
                    all_results['experiments'].append({
                        'layer_patched': layer_idx,
                        'component_type': component_type,
                        'error': str(e)
                    })
                
                pbar.update(1)
    
    # Save results
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to {save_path}")
    
    return all_results


def analyze_patching_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze results from batch patching experiments.
    
    Args:
        results: Output from batch_patching_experiment
    
    Returns:
        Analysis summary
    """
    experiments = results['experiments']
    
    # Find experiments that caused refusal
    causal_components = []
    for exp in experiments:
        if 'error' not in exp:
            if exp.get('is_refusal_patched') and not exp.get('is_refusal_harmless'):
                causal_components.append({
                    'layer': exp['layer_patched'],
                    'component': exp['component_type'],
                    'effect': 'induced_refusal'
                })
    
    # Rank by layer
    layer_importance = defaultdict(int)
    for comp in causal_components:
        layer_importance[comp['layer']] += 1
    
    analysis = {
        'total_experiments': len(experiments),
        'causal_components': causal_components,
        'num_causal_components': len(causal_components),
        'layer_importance': dict(sorted(layer_importance.items())),
        'most_important_layers': sorted(
            layer_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
    }
    
    return analysis


if __name__ == '__main__':
    """Test patching functionality."""
    print("="*60)
    print("TESTING ACTIVATION PATCHING")
    print("="*60)
    
    # This is a placeholder - real testing requires loaded model
    print("\nTo test patching:")
    print("1. Load model using model_utils.py")
    print("2. Run activation_patching_experiment() with contrastive prompts")
    print("3. Analyze results to find refusal circuits")
    print("\nExample usage in experiments/02_patching.py")