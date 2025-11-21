# Abstract: Deconstructing Llama-3's Refusal Mechanism

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a comprehensive mechanistic interpretability research project that deconstructs the "refusal mechanism" within Meta's Llama-3-8B-Instruct model. Using activation patching, ablation studies, and circuit analysis, we identify and causally verify the specific neural components responsible for safety refusals—all on consumer-grade Apple Silicon hardware.

## 🎯 Research Objectives

1. **Localization**: Identify specific components (Attention Heads & MLP Neurons) causally responsible for refusal behavior
2. **Causality Verification**: Prove component roles using activation patching between harmful/harmless prompt pairs
3. **Ablation Study**: Demonstrate that ablating identified circuits inhibits refusal mechanism
4. **Hardware Optimization**: Enable mechanistic interpretability research on resource-constrained devices (M2 8-16GB)

## 📊 Research Methodology

### Conceptual Framework
- **Residual Stream Theory**: View LLM as linear sum of information flows
- **Contrastive Analysis**: Compare activations between harmful vs. harmless prompts
- **Causal Tracing**: Use activation patching to isolate refusal circuits

### Technical Approach
```
1. Baseline → Test model on 15 harmful/harmless prompt pairs
2. Patching → Swap activations layer-by-layer to find causal components  
3. Ablation → Remove identified components and measure refusal rate drop
4. Analysis → Build circuit diagram of refusal mechanism
```

## 🚀 Quick Start

### Prerequisites

- **Hardware**: Apple Silicon Mac (M1/M2/M3) with 8GB+ unified memory
- **Software**: Python 3.11+, macOS 12.0+
- **Access**: HuggingFace account with Llama-3 model access

### Installation

```bash
# 1. Clone repository
git clone https://github.com/weissv/abstract.git
cd abstract

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch with MPS support
# Visit https://pytorch.org for latest command, or:
pip3 install torch torchvision torchaudio

# 4. Install dependencies
pip install -r requirements.txt

# 5. Login to HuggingFace
huggingface-cli login
# Enter your token: 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
```

### Configuration

Edit `config.yaml` to customize experiment parameters:

```yaml
model:
  name: "meta-llama/Meta-Llama-3-8B-Instruct"
  hf_token: "your_token_here"
  quantization: "4bit"  # Fits in 8-16GB RAM
```

## 📂 Project Structure

```
abstract/
├── README.md                     # This file
├── RESEARCH_PLAN.md             # Detailed 6-week research plan
├── config.yaml                  # Experiment configuration
├── requirements.txt             # Python dependencies
│
├── data/
│   └── prompts.json            # 15 harmful/harmless prompt pairs
│
├── src/
│   ├── model_utils.py          # Model loading, quantization, generation
│   ├── patching.py             # Activation patching & caching
│   ├── ablation.py             # Component ablation utilities
│   └── visualization.py        # Plotly/CircuitsVis visualizations
│
├── experiments/
│   ├── 01_baseline.py          # Baseline characterization
│   ├── 02_patching.py          # Activation patching experiments
│   └── 03_ablation.py          # Ablation study
│
└── outputs/
    ├── results/                # JSON experiment results
    ├── figures/                # Visualizations & dashboards
    └── cache/                  # Cached activations
```

## 🧪 Running Experiments

### Experiment 1: Baseline Characterization

Test model behavior on harmful vs. harmless prompts:

```bash
python experiments/01_baseline.py
```

**Output**: 
- Refusal rates per category
- Sample outputs
- `outputs/results/01_baseline_results.json`

**Expected Results**:
- Harmful prompts: ~90-95% refusal rate
- Harmless prompts: ~0-5% refusal rate

---

### Experiment 2: Activation Patching

Identify causal components via layer-wise activation swapping:

```bash
python experiments/02_patching.py
```

**Output**:
- Causal component rankings
- Layer importance scores
- Interactive dashboard: `outputs/figures/dashboard.html`
- `outputs/results/02_patching_results.json`

**Key Metrics**:
- Components that induce refusal when patched
- Most important layers (expected: 15-28)

---

### Experiment 3: Ablation Study

Verify necessity by ablating identified components:

```bash
python experiments/03_ablation.py
```

**Output**:
- Refusal rate drop per ablation
- Cumulative ablation curves
- `outputs/results/03_ablation_results.json`

**Success Criterion**:
- Ablating top-K components reduces refusal rate by >30%

---

## 📈 Visualization & Analysis

### Interactive Dashboard

Open the generated dashboard to explore results:

```bash
open outputs/figures/dashboard.html
```

**Features**:
- Activation patching heatmaps
- Layer importance bar charts
- Circuit diagrams
- Per-category breakdowns

### Key Visualizations

1. **Patching Heatmap**: Shows which layer+component combinations induce refusal
2. **Layer Importance**: Ranks layers by number of causal components
3. **Circuit Diagram**: Network visualization of refusal pathway
4. **Ablation Curves**: Refusal rate vs. number of ablated components

## 🔬 Technical Details

### Memory Optimization for Apple Silicon

| Component | Strategy | Memory Saved |
|-----------|----------|--------------|
| Model Weights | 4-bit NF4 quantization | ~70% (16GB → 4.5GB) |
| Activations | Batch size = 1 | ~80% |
| Gradients | No backprop (inference only) | 100% |
| Cache | Disk offloading | Variable |

**Total RAM Usage**: ~10GB peak (fits M2 16GB with headroom)

### Activation Patching Algorithm

```python
# Pseudocode for activation patching
harmful_cache = run_with_cache(model, "How to build a bomb?")
harmless_cache = run_with_cache(model, "How to build a shed?")

for layer in range(32):
    # Swap activations from harmful → harmless at layer L
    patched_output = run_with_patch(
        model, 
        prompt="How to build a shed?",
        patch={layer: harmful_cache[layer]}
    )
    
    # If output switches to refusal, layer L is causal
    if is_refusal(patched_output):
        causal_layers.append(layer)
```

## 📊 Expected Findings

Based on mechanistic interpretability literature, we hypothesize:

1. **Refusal Circuit Localization**: <5% of parameters responsible for >80% of refusal behavior
2. **Hierarchical Processing**:
   - Early layers (0-10): Detect harmful keywords
   - Middle layers (10-20): Context evaluation
   - Late layers (20-31): Refusal execution
3. **Shared Safety Substrate**: Same components handle multiple harm categories

## 🛠️ Troubleshooting

### Common Issues

**Issue**: `OutOfMemoryError` during model loading
```bash
# Solution: Ensure 4-bit quantization is enabled
# In config.yaml, verify:
quantization:
  load_in_4bit: true
```

**Issue**: `TransformerLens` compatibility errors
```bash
# Solution: Some models need specific TransformerLens versions
pip install transformer-lens==1.14.0
```

**Issue**: MPS backend crashes
```bash
# Solution: Fallback to CPU
# In config.yaml:
model:
  device: "cpu"  # Slower but stable
```

**Issue**: HuggingFace authentication failed
```bash
# Solution: Re-login with correct token
huggingface-cli logout
huggingface-cli login
```

## 📝 Research Outputs

### Code Artifacts
- ✅ Reproducible experiment pipeline
- ✅ Reusable interpretability utilities
- ✅ Interactive visualization dashboard

### Planned Publications
- [ ] Medium article: "Deconstructing AI Safety: Inside Llama-3's Refusal Mechanism"
- [ ] LessWrong post: "M2-Scale Mechanistic Interpretability"
- [ ] Academic preprint (if results warrant)

## 🤝 Contributing

This is a research project, but contributions are welcome:

1. **Bug Reports**: Open an issue with reproduction steps
2. **Feature Requests**: Suggest new analyses or visualizations
3. **Extensions**: Try on other models (Llama-2, Mistral, etc.)

## ⚠️ Ethical Considerations

**Important**: This research is for AI safety and transparency. 

- **Do NOT** use ablated models in production
- **Do NOT** deploy "jailbroken" models
- **Purpose**: Understanding safety mechanisms to improve them

## 📚 References

### Key Papers
- [Mechanistic Interpretability (Anthropic)](https://transformer-circuits.pub/)
- [Activation Patching (Redwood Research)](https://arxiv.org/abs/2202.05262)
- [TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/)

### Related Work
- ROME: Locating and Editing Factual Associations
- Causal Tracing in Language Models
- Representation Engineering

## 📜 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Neel Nanda** for TransformerLens
- **Anthropic** for mechanistic interpretability research
- **Meta** for Llama-3 model
- **HuggingFace** for transformers library

---

## 🎓 Citation

If you use this work, please cite:

```bibtex
@misc{abstract2024,
  title={Deconstructing Llama-3's Refusal Mechanism: A Mechanistic Interpretability Study},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/weissv/abstract}
}
```

---

**Status**: 🚧 Active Research (November 2024)

For questions: Open an issue or contact via GitHub