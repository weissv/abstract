"""
Ablation study utilities for testing necessity of identified components.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np


class AblationManager:
    """Manage ablation of specific model components."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.ablated_components = []
    
    def ablate_attention_head(
        self,
        layer_idx: int,
        head_idx: int,
        ablation_type: str = "zero"
    ):
        """
        Ablate a specific attention head.
        
        Args:
            layer_idx: Layer index
            head_idx: Head index within layer
            ablation_type: 'zero' or 'mean'
        """
        def ablation_hook(module, input, output):
            """Hook to zero out specific attention head."""
            # output is typically (hidden_states, attention_weights, ...)
            if isinstance(output, tuple):
                hidden = output[0]
                
                # Llama has num_heads attention heads
                num_heads = module.num_heads
                head_dim = hidden.shape[-1] // num_heads
                
                # Reshape to separate heads
                batch_size, seq_len, _ = hidden.shape
                hidden = hidden.view(batch_size, seq_len, num_heads, head_dim)
                
                # Ablate specific head
                if ablation_type == "zero":
                    hidden[:, :, head_idx, :] = 0
                elif ablation_type == "mean":
                    mean_val = hidden[:, :, head_idx, :].mean()
                    hidden[:, :, head_idx, :] = mean_val
                
                # Reshape back
                hidden = hidden.view(batch_size, seq_len, num_heads * head_dim)
                
                # Return modified output
                return (hidden,) + output[1:]
            return output
        
        # Register hook on attention layer
        layer = self.model.model.layers[layer_idx]
        hook = layer.self_attn.register_forward_hook(ablation_hook)
        self.hooks.append(hook)
        self.ablated_components.append({
            'type': 'attention_head',
            'layer': layer_idx,
            'head': head_idx,
            'ablation_type': ablation_type
        })
    
    def ablate_mlp_neurons(
        self,
        layer_idx: int,
        neuron_indices: List[int],
        ablation_type: str = "zero"
    ):
        """
        Ablate specific MLP neurons.
        
        Args:
            layer_idx: Layer index
            neuron_indices: List of neuron indices to ablate
            ablation_type: 'zero' or 'mean'
        """
        def ablation_hook(module, input, output):
            """Hook to ablate specific neurons."""
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            # Ablate neurons
            for neuron_idx in neuron_indices:
                if ablation_type == "zero":
                    hidden[:, :, neuron_idx] = 0
                elif ablation_type == "mean":
                    mean_val = hidden[:, :, neuron_idx].mean()
                    hidden[:, :, neuron_idx] = mean_val
            
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        
        # Register hook on MLP layer
        layer = self.model.model.layers[layer_idx]
        hook = layer.mlp.register_forward_hook(ablation_hook)
        self.hooks.append(hook)
        self.ablated_components.append({
            'type': 'mlp_neurons',
            'layer': layer_idx,
            'neurons': neuron_indices,
            'ablation_type': ablation_type
        })
    
    def ablate_layer(
        self,
        layer_idx: int,
        component: str = "all",
        ablation_type: str = "zero"
    ):
        """
        Ablate entire layer or component.
        
        Args:
            layer_idx: Layer index
            component: 'all', 'attention', or 'mlp'
            ablation_type: 'zero' or 'mean'
        """
        def ablation_hook(module, input, output):
            """Hook to ablate entire component."""
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            if ablation_type == "zero":
                hidden = torch.zeros_like(hidden)
            elif ablation_type == "mean":
                hidden = torch.full_like(hidden, hidden.mean())
            
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        
        layer = self.model.model.layers[layer_idx]
        
        if component in ["all", "attention"]:
            hook = layer.self_attn.register_forward_hook(ablation_hook)
            self.hooks.append(hook)
        
        if component in ["all", "mlp"]:
            hook = layer.mlp.register_forward_hook(ablation_hook)
            self.hooks.append(hook)
        
        self.ablated_components.append({
            'type': 'layer',
            'layer': layer_idx,
            'component': component,
            'ablation_type': ablation_type
        })
    
    def clear_ablations(self):
        """Remove all ablation hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.ablated_components = []
    
    def get_ablated_components(self) -> List[Dict[str, Any]]:
        """Get list of currently ablated components."""
        return self.ablated_components


def test_ablation_effect(
    model: nn.Module,
    tokenizer,
    ablation_manager: AblationManager,
    test_prompts: List[str],
    baseline_outputs: Optional[List[str]] = None,
    max_new_tokens: int = 100
) -> Dict[str, Any]:
    """
    Test effect of ablations on model outputs.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        ablation_manager: AblationManager with ablations registered
        test_prompts: List of prompts to test
        baseline_outputs: Optional baseline outputs without ablation
        max_new_tokens: Max tokens to generate
    
    Returns:
        Dictionary with results
    """
    from model_utils import generate_text
    from patching import is_refusal
    
    results = {
        'ablated_components': ablation_manager.get_ablated_components(),
        'test_results': []
    }
    
    for i, prompt in enumerate(tqdm(test_prompts, desc="Testing ablation")):
        # Generate with ablation
        try:
            output = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens
            )
            
            is_refusal_output = is_refusal(output)
            
            test_result = {
                'prompt': prompt,
                'output': output,
                'is_refusal': is_refusal_output
            }
            
            # Compare with baseline if provided
            if baseline_outputs and i < len(baseline_outputs):
                baseline = baseline_outputs[i]
                is_refusal_baseline = is_refusal(baseline)
                test_result['baseline_output'] = baseline
                test_result['baseline_is_refusal'] = is_refusal_baseline
                test_result['behavior_changed'] = (is_refusal_output != is_refusal_baseline)
            
            results['test_results'].append(test_result)
            
        except Exception as e:
            results['test_results'].append({
                'prompt': prompt,
                'error': str(e)
            })
    
    return results


def systematic_ablation_study(
    model: nn.Module,
    tokenizer,
    causal_components: List[Dict[str, Any]],
    test_prompts: List[str],
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Conduct systematic ablation study of identified causal components.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        causal_components: List of components identified as causal
        test_prompts: Prompts to test (should include harmful ones)
        save_path: Path to save results
    
    Returns:
        Ablation study results
    """
    from model_utils import generate_text
    from patching import is_refusal
    
    # 1. Get baseline outputs (no ablation)
    print("Getting baseline outputs...")
    baseline_outputs = []
    baseline_refusal_rate = 0
    
    for prompt in tqdm(test_prompts, desc="Baseline"):
        output = generate_text(model, tokenizer, prompt, max_new_tokens=100)
        baseline_outputs.append(output)
        if is_refusal(output):
            baseline_refusal_rate += 1
    
    baseline_refusal_rate /= len(test_prompts)
    
    all_results = {
        'baseline_refusal_rate': baseline_refusal_rate,
        'baseline_outputs': baseline_outputs,
        'ablation_experiments': []
    }
    
    # 2. Test ablating individual components
    print("\nTesting individual component ablations...")
    
    for comp in tqdm(causal_components[:20], desc="Individual ablations"):  # Limit to top 20
        ablation_mgr = AblationManager(model)
        
        # Ablate component
        if comp.get('component') == 'attention':
            # For now, ablate entire attention layer (would need head index for specific head)
            ablation_mgr.ablate_layer(comp['layer'], component='attention')
        elif comp.get('component') == 'mlp':
            ablation_mgr.ablate_layer(comp['layer'], component='mlp')
        else:
            ablation_mgr.ablate_layer(comp['layer'], component='all')
        
        # Test
        results = test_ablation_effect(
            model=model,
            tokenizer=tokenizer,
            ablation_manager=ablation_mgr,
            test_prompts=test_prompts,
            baseline_outputs=baseline_outputs
        )
        
        # Calculate refusal rate
        refusal_count = sum(1 for r in results['test_results'] if r.get('is_refusal', False))
        refusal_rate = refusal_count / len(test_prompts)
        
        results['refusal_rate'] = refusal_rate
        results['refusal_rate_change'] = baseline_refusal_rate - refusal_rate
        
        all_results['ablation_experiments'].append(results)
        
        # Clear ablations
        ablation_mgr.clear_ablations()
    
    # 3. Test cumulative ablation (top components)
    print("\nTesting cumulative ablation of top components...")
    
    for top_k in [5, 10, 15, 20]:
        if top_k > len(causal_components):
            continue
        
        ablation_mgr = AblationManager(model)
        
        # Ablate top-k components
        for comp in causal_components[:top_k]:
            if comp.get('component') == 'attention':
                ablation_mgr.ablate_layer(comp['layer'], component='attention')
            elif comp.get('component') == 'mlp':
                ablation_mgr.ablate_layer(comp['layer'], component='mlp')
        
        # Test
        results = test_ablation_effect(
            model=model,
            tokenizer=tokenizer,
            ablation_manager=ablation_mgr,
            test_prompts=test_prompts,
            baseline_outputs=baseline_outputs
        )
        
        refusal_count = sum(1 for r in results['test_results'] if r.get('is_refusal', False))
        refusal_rate = refusal_count / len(test_prompts)
        
        results['refusal_rate'] = refusal_rate
        results['refusal_rate_change'] = baseline_refusal_rate - refusal_rate
        results['num_ablated'] = top_k
        results['ablation_type'] = 'cumulative'
        
        all_results['ablation_experiments'].append(results)
        
        ablation_mgr.clear_ablations()
    
    # Save results
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        save_data = {
            'baseline_refusal_rate': all_results['baseline_refusal_rate'],
            'ablation_experiments': [
                {
                    'refusal_rate': exp['refusal_rate'],
                    'refusal_rate_change': exp['refusal_rate_change'],
                    'ablated_components': exp['ablated_components'],
                    'num_changed_behaviors': sum(
                        1 for r in exp['test_results'] 
                        if r.get('behavior_changed', False)
                    )
                }
                for exp in all_results['ablation_experiments']
            ]
        }
        
        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Results saved to {save_path}")
    
    return all_results


def analyze_ablation_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze ablation study results.
    
    Args:
        results: Output from systematic_ablation_study
    
    Returns:
        Analysis summary
    """
    baseline_rate = results['baseline_refusal_rate']
    experiments = results['ablation_experiments']
    
    # Find most effective ablations
    individual_ablations = [
        exp for exp in experiments 
        if exp.get('ablation_type') != 'cumulative'
    ]
    
    cumulative_ablations = [
        exp for exp in experiments 
        if exp.get('ablation_type') == 'cumulative'
    ]
    
    # Sort by effect size
    individual_ablations.sort(
        key=lambda x: abs(x.get('refusal_rate_change', 0)),
        reverse=True
    )
    
    analysis = {
        'baseline_refusal_rate': baseline_rate,
        'most_effective_individual': individual_ablations[:10] if individual_ablations else [],
        'cumulative_results': cumulative_ablations,
        'summary': {
            'total_experiments': len(experiments),
            'max_refusal_rate_drop': max(
                (exp.get('refusal_rate_change', 0) for exp in experiments),
                default=0
            ),
            'components_needed_for_50pct_drop': None
        }
    }
    
    # Find minimum components needed for 50% reduction
    for exp in cumulative_ablations:
        if exp.get('refusal_rate_change', 0) >= baseline_rate * 0.5:
            analysis['summary']['components_needed_for_50pct_drop'] = exp.get('num_ablated')
            break
    
    return analysis


if __name__ == '__main__':
    """Test ablation functionality."""
    print("="*60)
    print("TESTING ABLATION UTILITIES")
    print("="*60)
    
    print("\nAblation utilities ready for use!")
    print("Example usage:")
    print("  1. Create AblationManager(model)")
    print("  2. Register ablations (heads, neurons, layers)")
    print("  3. Run test_ablation_effect() to measure impact")
    print("  4. Use systematic_ablation_study() for comprehensive analysis")
