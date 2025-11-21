# Contributing to Llama-3.1 Refusal Analysis

Thank you for your interest in contributing! This project welcomes contributions from the community.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/abstract.git
   cd abstract
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🔧 Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install pytest black flake8 mypy
```

## 📝 Code Guidelines

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints where possible
- Maximum line length: 100 characters
- Use docstrings for all functions/classes

### Example:

```python
def process_activations(
    activations: torch.Tensor,
    layer: int,
    component: str = "attn"
) -> Dict[str, torch.Tensor]:
    """
    Process activations from a specific layer and component.
    
    Args:
        activations: Tensor of shape (batch, seq_len, hidden_dim)
        layer: Layer index (0-31 for Llama-3.1-8B)
        component: Component type ('attn' or 'mlp')
    
    Returns:
        Dictionary containing processed activations
    """
    # Implementation here
    pass
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📊 Experiment Contributions

When adding new experiments:

1. Create script in `experiments/` directory
2. Follow naming convention: `XX_description.py`
3. Update `config.yaml` if needed
4. Add documentation to README
5. Include expected outputs

## 🐛 Bug Reports

Please include:

- Python version
- GPU type and VRAM
- Full error traceback
- Steps to reproduce
- Expected vs actual behavior

## ✨ Feature Requests

Please describe:

- Use case and motivation
- Proposed implementation approach
- Expected behavior
- Alternative solutions considered

## 📥 Pull Request Process

1. **Update documentation** for any changed functionality
2. **Add tests** for new features
3. **Ensure all tests pass** locally
4. **Update CHANGELOG** (if applicable)
5. **Request review** from maintainers

### PR Checklist:

- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

## 🎯 Areas for Contribution

We especially welcome contributions in:

- 🔍 **New analysis methods**: Additional interpretability techniques
- 📊 **Visualizations**: New plotting methods or dashboards
- ⚡ **Optimization**: Performance improvements
- 📚 **Documentation**: Tutorials, examples, guides
- 🧪 **Testing**: Additional test coverage
- 🛠️ **Tooling**: Development utilities

## 💬 Questions?

- Open a [GitHub Discussion](https://github.com/weissv/abstract/discussions)
- Check existing [Issues](https://github.com/weissv/abstract/issues)

## 📜 Code of Conduct

Please be respectful and constructive. We're all here to learn and improve AI safety research.

---

Thank you for contributing! 🙏
