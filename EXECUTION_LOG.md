# Execution Log

## Environment Setup - COMPLETED ✓

**Date**: Current session
**Status**: SUCCESS

### Steps Completed:

1. **Python Environment** ✓
   - Installed Python 3.12 via Homebrew
   - Created virtual environment with `python3.12 -m venv venv`
   - Installed all dependencies:
     - PyTorch 2.2.2
     - Transformers 4.57.1
     - Accelerate 1.11.0
     - HuggingFace Hub 0.36.0
     - NumPy 1.26.4 (downgraded from 2.3.5 for PyTorch compatibility)
     - Plotly, Pandas, Jupyter, Datasets, etc.

2. **Configuration** ✓
   - Updated config.yaml to use fp16 instead of 4-bit quantization (bitsandbytes requires CUDA)
   - Created output directories: outputs/results, outputs/figures, outputs/cache
   - Logged into HuggingFace with provided token

3. **Code Fixes** ✓
   - Added TransformerLens import guard to handle missing library
   - Model loading falls back to HuggingFace transformers only

## Experiment 1: Baseline Characterization - IN PROGRESS

**Status**: Running in background (Terminal ID: 63df939c-57b7-4c26-b8d3-c204a2a0f83c)

### Current Activity:
- Downloading Llama-3-8B-Instruct model (~15GB)
- 4 safetensors files being fetched from HuggingFace

### Expected Next Steps:
1. Model loads to MPS device
2. Test 15 harmful prompts
3. Test 15 harmless prompts
4. Calculate refusal rates
5. Save results to `outputs/results/01_baseline_results.json`

### Expected Results:
- Harmful refusal rate: 90-95%
- Harmless refusal rate: 0-5%
- Runtime: ~20 minutes

## Experiment 2: Activation Patching - PENDING

Waits for Experiment 1 completion.

## Experiment 3: Ablation Study - PENDING

Waits for Experiment 2 completion.

---

## Technical Notes:

### Environment:
- OS: macOS Sonoma
- Hardware: Apple M2 MacBook Air (8-16GB RAM)
- Python: 3.12.12
- Device: MPS (Apple Metal Performance Shaders)

### Issue Resolved:
- **Problem**: Python 3.14 incompatible with PyTorch
- **Solution**: Installed Python 3.12 via Homebrew
- **Problem**: NumPy 2.3.5 incompatible with PyTorch 2.2.2
- **Solution**: Downgraded to NumPy 1.26.4
- **Problem**: bitsandbytes requires CUDA
- **Solution**: Disabled 4-bit quantization, using fp16 instead

### Memory Usage:
- Model size: ~15GB (fp16)
- Expected peak RAM: ~18-20GB
- MPS handles memory allocation automatically
