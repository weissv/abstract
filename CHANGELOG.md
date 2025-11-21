# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-11-21

### Major Refactoring for Google Colab & HuggingFace Publishing

#### Added
- **Google Colab Support**: Full optimization for T4 GPU with 4-bit quantization
- **Interactive Notebook**: `llama_refusal_analysis.ipynb` with one-click execution
- **Token Management**: User prompt for HuggingFace token (removed hardcoded values)
- **Documentation**: Comprehensive README, CONTRIBUTING, QUICKSTART guides
- **License**: MIT License with third-party attributions
- **Setup Script**: `setup.py` for pip installation
- **.gitignore**: Comprehensive ignore patterns for clean repo

#### Changed
- **Model**: Updated from Llama-3 to **Llama-3.1-8B-Instruct**
- **Hardware**: Optimized for NVIDIA T4 (removed Mac M2 specifics)
- **Device**: CUDA-first (removed MPS/Mac-specific code)
- **Config**: Removed hardcoded `hf_token`, now prompts user at runtime
- **Requirements**: Streamlined dependencies for Colab environment
- **Model Loading**: Enhanced error handling and progress indicators
- **Memory Stats**: CUDA-optimized memory monitoring

#### Removed
- Mac-specific configuration (MPS device handling)
- Hardcoded HuggingFace tokens from config files
- Old documentation files (RESEARCH_PLAN, IMPLEMENTATION_SUMMARY, etc.)
- TransformerLens requirement (now optional)

#### Fixed
- NumPy version compatibility (<2.0.0 for PyTorch)
- Bitsandbytes CUDA dependency issues
- Token authentication flow

---

## [0.1.0] - 2025-11-20

### Initial Implementation

#### Added
- Baseline characterization experiment
- Activation patching implementation  
- Ablation study tools
- Visualization dashboards
- 15 contrastive prompt pairs
- Configuration system
- Research methodology documentation

---

## Future Roadmap

### [1.1.0] - Planned
- [ ] Add more diverse prompt categories
- [ ] Implement circuit discovery algorithms
- [ ] Add attention pattern visualizations
- [ ] Create tutorial notebooks
- [ ] Add unit tests

### [1.2.0] - Planned
- [ ] Support for larger Llama models (70B)
- [ ] Multi-GPU distributed patching
- [ ] Real-time dashboard updates
- [ ] API for programmatic access

---

**Format**: Based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)  
**Versioning**: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
