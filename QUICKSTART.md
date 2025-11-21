# Quick Start Guide

## 🚀 5-Minute Setup (Google Colab)

1. **Open the notebook**:
   - Click: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weissv/abstract/blob/main/llama_refusal_analysis.ipynb)

2. **Enable GPU**:
   - Runtime → Change runtime type → GPU (T4)

3. **Get HuggingFace Token**:
   - Visit: https://huggingface.co/settings/tokens
   - Create a token with "read" access
   - Accept Llama-3.1 license: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct

4. **Run all cells**:
   - Runtime → Run all
   - Enter your HF token when prompted

5. **Wait for results** (~3 hours total):
   - Baseline: ~15 minutes
   - Patching: ~2 hours  
   - Ablation: ~45 minutes

6. **Download results**:
   - Last cell downloads `results.zip`

## 💻 Local Setup (with CUDA GPU)

```bash
# Clone repository
git clone https://github.com/weissv/abstract.git
cd abstract

# Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt

# Set token
export HF_TOKEN="your_hf_token"

# Run experiments
python experiments/01_baseline.py
python experiments/02_patching.py
python experiments/03_ablation.py

# View results
ls outputs/results/
ls outputs/figures/
```

## 📊 Expected Output

After running all experiments, you'll have:

```
outputs/
├── results/
│   ├── 01_baseline_results.json    # Refusal rates
│   ├── 02_patching_results.json    # Causal components
│   └── 03_ablation_results.json    # Verification results
├── figures/
│   ├── patching_dashboard.html     # Interactive explorer
│   ├── refusal_circuit.png         # Circuit diagram
│   └── activation_heatmap.html     # Layer analysis
└── cache/
    └── activations/                # Cached activations
```

## 🎯 Success Criteria

✅ Baseline: 90%+ harmful refusal rate  
✅ Patching: 15-25 causal components identified  
✅ Ablation: >30% refusal reduction

## ⚡ Quick Test (5 prompts, ~10 minutes)

Edit `config.yaml`:
```yaml
experiment:
  num_prompt_pairs: 5  # Instead of 15
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "CUDA out of memory" | Enable 4-bit quantization in config.yaml |
| "Token invalid" | Ensure you accepted Llama-3.1 license |
| "Module not found" | Restart runtime after pip install |

## 📚 Next Steps

- Read [CONTRIBUTING.md](CONTRIBUTING.md) to extend the research
- Check [examples/](examples/) for analysis notebooks
- Join discussions: https://github.com/weissv/abstract/discussions
