# 🔬 Llama-3.1 Refusal Mechanism Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weissv/abstract/blob/main/llama_refusal_analysis.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

A comprehensive mechanistic interpretability study to deconstruct the refusal mechanism in Meta's **Llama-3.1-8B-Instruct** model using activation patching and ablation techniques.

---

## 🎯 Overview

This project implements a rigorous mechanistic interpretability pipeline to understand how Llama-3.1 decides whether to refuse harmful requests. By identifying and analyzing the specific attention heads and MLP components responsible for refusal behavior, we gain insights into AI safety mechanisms.

### Key Features

- 🔍 **Activation Patching**: Identify causal components responsible for refusal decisions
- ✂️ **Ablation Studies**: Verify necessity of identified components
- 📊 **Interactive Visualizations**: Explore results with Plotly dashboards
- 🎨 **Circuit Diagrams**: Visualize the refusal mechanism as a computational graph
- ⚡ **Optimized for Colab**: Works on free T4 GPU with 4-bit quantization
- 📦 **Reproducible**: Complete pipeline from data to results

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

Click the badge to run in Colab with zero setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weissv/abstract/blob/main/llama_refusal_analysis.ipynb)

**Requirements:**
- Google account
- HuggingFace account with Llama-3.1 access ([request here](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct))
- HuggingFace API token ([create here](https://huggingface.co/settings/tokens))

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/weissv/abstract.git
cd abstract

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token
export HF_TOKEN="your_token_here"

# Run experiments
python experiments/01_baseline.py
python experiments/02_patching.py
python experiments/03_ablation.py
```

**Requirements:**
- Python 3.8+
- CUDA-capable GPU with 15GB+ VRAM (or 8GB with 4-bit quantization)
- 50GB disk space

---

## 📁 Project Structure

```
abstract/
├── llama_refusal_analysis.ipynb  # Main Colab notebook
├── src/
│   ├── model_utils.py           # Model loading & utilities
│   ├── patching.py              # Activation patching implementation
│   ├── ablation.py              # Ablation study tools
│   └── visualization.py         # Plotting and dashboards
├── experiments/
│   ├── 01_baseline.py           # Baseline characterization
│   ├── 02_patching.py           # Activation patching experiments
│   └── 03_ablation.py           # Ablation studies
├── data/
│   └── prompts.json             # Contrastive harmful/harmless prompts
├── config.yaml                  # Configuration settings
├── requirements.txt             # Python dependencies
└── outputs/                     # Generated results (git-ignored)
    ├── results/                 # JSON experiment results
    ├── figures/                 # Visualizations & dashboards
    └── cache/                   # Cached activations
```

---

## 🔬 Methodology

### 1. Baseline Characterization

Test the model on 15 pairs of contrastive prompts (harmful vs. harmless) to establish baseline refusal rates.

**Expected Results:**
- Harmful refusal rate: 90-95%
- Harmless refusal rate: 0-5%

### 2. Activation Patching

Systematically patch activations from harmless runs into harmful runs to identify which components cause refusal behavior.

**Process:**
1. Run harmful prompt and cache activations
2. Run harmless prompt and cache activations
3. For each layer/component:
   - Replace harmful activations with harmless ones
   - Measure change in refusal probability
4. Rank components by causal effect

**Expected Findings:**
- 15-25 highly causal attention heads
- Concentrated in layers 10-25
- Both attention and MLP components involved

### 3. Ablation Study

Verify that identified components are necessary by removing them and measuring refusal rate drop.

**Success Criteria:**
- Ablating top components reduces refusal rate by >30%
- Cumulative effect shows diminishing returns
- Results reproducible across prompt categories

---

## 📊 Results & Outputs

### Experiment Outputs

| File | Description |
|------|-------------|
| `outputs/results/01_baseline_results.json` | Baseline refusal rates per category |
| `outputs/results/02_patching_results.json` | Causal effect scores for all components |
| `outputs/results/03_ablation_results.json` | Ablation impact measurements |
| `outputs/figures/patching_dashboard.html` | Interactive exploration dashboard |
| `outputs/figures/refusal_circuit.png` | Circuit diagram of refusal mechanism |

---

## ⚙️ Configuration

Edit `config.yaml` to customize experiments:

```yaml
model:
  name: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  quantization: "4bit"  # Options: 4bit, 8bit, fp16, fp32
  device: "cuda"

quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_quant_type: "nf4"

experiment:
  batch_size: 1
  num_prompt_pairs: 15
  max_new_tokens: 100
```

---

## 📈 Performance

### Memory Usage

| Configuration | VRAM | Runtime (3 experiments) |
|---------------|------|-------------------------|
| fp16 (full) | ~15GB | ~2 hours |
| 4-bit quantization | ~5GB | ~3 hours |
| 8-bit quantization | ~8GB | ~2.5 hours |

---

## 🎓 Citation

If you use this research in your work, please cite:

```bibtex
@misc{llama31_refusal_analysis,
  title={Mechanistic Analysis of Refusal Behavior in Llama-3.1-8B-Instruct},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/weissv/abstract}
}
```

---

## 📄 License

This project is licensed under the MIT License.

**Note:** Llama-3.1 model usage is subject to Meta's [License Agreement](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).

---

## ⚠️ Ethical Considerations

This research aims to improve AI safety by understanding refusal mechanisms. Please use responsibly:

- ✅ **DO**: Use for safety research and interpretability studies
- ✅ **DO**: Share findings to improve AI alignment
- ❌ **DON'T**: Use to bypass safety measures in production systems
- ❌ **DON'T**: Apply findings to harm users or violate policies

---

## 🐛 Troubleshooting

### Common Issues

**Q: "CUDA out of memory" error**
- A: Enable 4-bit quantization in `config.yaml`

**Q: "HuggingFace token invalid"**
- A: Ensure you've accepted Llama-3.1 license and token has read permissions

**Q: "Model download slow"**
- A: First run downloads ~15GB model. Subsequent runs use cached version

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Open a Pull Request

---

**⭐ If you find this useful, please star the repo!**

Built with ❤️ for the AI safety community
