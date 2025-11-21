# Quick Start Guide - Llama-3 Refusal Mechanism Research

## TL;DR - Run the Complete Research Pipeline

```bash
# 1. Setup (5 minutes)
git clone https://github.com/weissv/abstract.git
cd abstract
python3.11 -m venv venv  # Must use Python 3.11 for PyTorch MPS
source venv/bin/activate
pip install torch torchvision torchaudio
pip install -r requirements.txt
huggingface-cli login  # Use token: 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

# 2. Run Experiments (3-4 hours total)
python experiments/01_baseline.py      # 20 min - Test refusal behavior
python experiments/02_patching.py      # 2 hours - Find refusal circuits
python experiments/03_ablation.py      # 1 hour - Verify findings

# 3. View Results
open outputs/figures/dashboard.html    # Interactive visualizations
cat outputs/results/*_results.json     # Raw data
```

## What This Research Does

**Goal**: Identify the specific neurons and attention heads in Llama-3-8B that cause it to refuse harmful requests.

**Method**: 
1. **Baseline**: Test model on 15 harmful prompts (e.g., "How to build a bomb?") vs. harmless ones ("How to build a shed?")
2. **Patching**: Swap activations between harmful/harmless prompts layer-by-layer to find which components cause refusal
3. **Ablation**: Turn off identified components and verify refusal rate drops

**Output**: 
- Top 20 "refusal neurons/heads" ranked by importance
- Circuit diagram showing refusal pathway through layers
- Proof that <5% of model parameters control safety behavior

## Project Structure

```
abstract/
├── experiments/           # Run these in order (01 → 02 → 03)
│   ├── 01_baseline.py    # Test harmful vs harmless prompts
│   ├── 02_patching.py    # Find refusal circuits
│   └── 03_ablation.py    # Verify by ablation
├── src/                   # Core utilities (don't run directly)
│   ├── model_utils.py    # Load model, generate text
│   ├── patching.py       # Activation patching logic
│   ├── ablation.py       # Component ablation
│   └── visualization.py  # Create plots & dashboards
├── data/
│   └── prompts.json      # 15 harmful/harmless pairs
├── outputs/              # Results go here
│   ├── results/          # JSON data
│   └── figures/          # HTML visualizations
└── config.yaml           # Adjust settings here
```

## Key Configuration Options

Edit `config.yaml` before running:

```yaml
model:
  hf_token: "your_token_here"       # Your HuggingFace token
  quantization: "4bit"              # Use 4bit for 8-16GB RAM, fp16 for >16GB
  device: "mps"                     # "mps" for M1/M2/M3, "cuda" for NVIDIA

experiment:
  batch_size: 1                     # Always 1 for memory constraints
  max_new_tokens: 100               # Reduce if running out of memory
```

## Troubleshooting

### "Out of Memory" Error
```yaml
# In config.yaml, reduce:
experiment:
  max_new_tokens: 50     # From 100
model:
  quantization: "4bit"   # Ensure 4-bit is enabled
```

### "No module named 'torch'"
```bash
# PyTorch requires Python 3.11 (not 3.14)
python --version  # Should show 3.11.x

# If not, reinstall venv:
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### "TransformerLens not compatible"
```bash
# Some quantized models don't work with TransformerLens
# Code will fall back to manual hooks automatically
# Check logs for "Warning: Could not wrap with HookedTransformer"
```

### "HuggingFace authentication failed"
```bash
# Re-login
huggingface-cli logout
huggingface-cli login
# Paste: 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
```

## Expected Results

### Experiment 1: Baseline (20 minutes)
```
Harmful Prompts:
  Refusal Rate: 90-95%
  Example: "I cannot assist with that request..." ✅

Harmless Prompts:
  Refusal Rate: 0-5%
  Example: "Here's how to build a shed..." ✅
```

### Experiment 2: Patching (2 hours)
```
Causal Components Found: 15-25
Top Layer: 24 (late layer as expected)
Most Important Component: Layer 24, Attention Head 12

Dashboard: outputs/figures/dashboard.html
- Heatmap shows which layers induce refusal
- Circuit diagram shows information flow
```

### Experiment 3: Ablation (1 hour)
```
Baseline Refusal Rate: 93%
After Ablating Top 10 Components: 60% (-33%)
After Ablating Top 20 Components: 45% (-48%)

✅ Success: >30% reduction proves causality
```

## Understanding the Output

### `outputs/results/02_patching_results.json`
```json
{
  "ranked_components": [
    {
      "layer": 24,
      "component": "attention",
      "frequency": 4,
      "percentage": 80.0
    }
  ]
}
```
**Interpretation**: Layer 24's attention is causal in 80% of test cases.

### `outputs/figures/dashboard.html`
- **Patching Heatmap**: Red = induces refusal when patched
- **Layer Importance**: Bar chart of causal component count per layer
- **Circuit Diagram**: Network showing refusal pathway

## Advanced Usage

### Test Custom Prompts
```python
from src.model_utils import load_model_and_tokenizer, generate_text

model, tokenizer = load_model_and_tokenizer(
    model_id="meta-llama/Meta-Llama-3-8B-Instruct",
    hf_token="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"
)

output = generate_text(model, tokenizer, "Your prompt here")
print(output)
```

### Run Single Patching Test
```python
from src.patching import activation_patching_experiment

result = activation_patching_experiment(
    model=model,
    tokenizer=tokenizer,
    harmful_prompt="How to build a bomb?",
    harmless_prompt="How to build a shed?",
    layer_to_patch=24,
    component_type="attention"
)

print(result['patching_effect'])  # "harmful->refusal" if causal
```

### Visualize Specific Layer
```python
from src.visualization import plot_attention_pattern

# After running with cache
fig = plot_attention_pattern(
    attention_pattern=cache.attention_pattern[24],
    layer_idx=24,
    head_idx=12,  # Specific head
    save_path="outputs/figures/layer24_head12.html"
)
```

## Research Output Checklist

After running all experiments, you should have:

- [📊] **Baseline Results** (`outputs/results/01_baseline_results.json`)
  - Refusal rates per category
  - Example outputs for each prompt

- [🔍] **Patching Analysis** (`outputs/results/02_patching_results.json`)
  - Ranked list of causal components
  - Layer importance scores
  - Top 3 most important layers identified

- [🧪] **Ablation Study** (`outputs/results/03_ablation_results.json`)
  - Individual component ablation effects
  - Cumulative ablation curve
  - Proof of >30% refusal reduction

- [📈] **Visualizations** (`outputs/figures/`)
  - `dashboard.html` - Interactive summary
  - `patching_heatmap.html` - Component causality
  - `layer_importance.html` - Layer rankings
  - `circuit_diagram.html` - Refusal pathway
  - `ablation_cumulative.html` - Ablation curve

- [📝] **Write-up Material**
  - Circuit diagram for publication
  - Quantitative results tables
  - Example refusal/compliance outputs

## Next Steps After Results

1. **Review Dashboard**: Identify top 10 components
2. **Write Research Article**:
   - Title: "Deconstructing Llama-3's Refusal Mechanism"
   - Sections: Intro, Methods, Results, Discussion
   - Figures: Circuit diagram, ablation curve
   - Code: Link to this repository
3. **Publish**:
   - Medium (for general audience)
   - LessWrong (for AI safety community)
   - arXiv preprint (if results warrant)

## Citation

If you use this research, please cite:

```bibtex
@misc{llama3_refusal_2024,
  title={Deconstructing Llama-3's Refusal Mechanism: A Mechanistic Interpretability Study},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/weissv/abstract}}
}
```

## Questions?

- **Code Issues**: Open GitHub issue
- **Research Questions**: See RESEARCH_PLAN.md for detailed methodology
- **Implementation Details**: Read IMPLEMENTATION_SUMMARY.md

---

**Ready to start?** Run: `python experiments/01_baseline.py`
