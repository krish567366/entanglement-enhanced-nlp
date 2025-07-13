# Contributing to Entanglement-Enhanced NLP

Thank you for your interest in contributing to the Entanglement-Enhanced NLP framework! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Types of Contributions

We welcome several types of contributions:

- **Bug reports and fixes**
- **Feature requests and implementations**
- **Documentation improvements**
- **Performance optimizations**
- **Test coverage enhancements**
- **Examples and tutorials**

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/krish567366/entanglement-enhanced-nlp.git
   cd entanglement-enhanced-nlp
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

5. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🔧 Development Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9.0 or higher
- Git for version control

### Development Installation

```bash
# Clone and enter directory
git clone https://github.com/krish567366/entanglement-enhanced-nlp.git
cd entanglement-enhanced-nlp

# Install in development mode with all dependencies
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=entanglement_enhanced_nlp --cov-report=html

# Run specific test file
pytest tests/test_entangled_embedding.py -v

# Run performance tests
pytest tests/test_performance.py -v --benchmark-only
```

### Code Quality

We maintain high code quality standards:

```bash
# Format code with black
black entanglement_enhanced_nlp/ tests/

# Check code style with flake8
flake8 entanglement_enhanced_nlp/ tests/

# Type checking with mypy
mypy entanglement_enhanced_nlp/

# Sort imports with isort
isort entanglement_enhanced_nlp/ tests/
```

## 📝 Code Style Guidelines

### Python Code Style

- Follow **PEP 8** guidelines
- Use **Black** for code formatting
- Maximum line length: **88 characters**
- Use **type hints** for all function signatures
- Write comprehensive **docstrings** for all public functions

### Example Function:

```python
def compute_entanglement_correlation(
    embeddings: torch.Tensor,
    correlation_strength: float = 0.7,
    decoherence_rate: float = 0.1
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute quantum entanglement correlations between token embeddings.
    
    This function implements quantum-inspired correlation computation
    based on Bell state entanglement principles.
    
    Args:
        embeddings: Input token embeddings of shape (batch_size, seq_len, hidden_dim)
        correlation_strength: Strength of quantum correlations (0.0-1.0)
        decoherence_rate: Rate of quantum decoherence (0.0-1.0)
    
    Returns:
        Tuple containing:
        - correlation_matrix: Pairwise correlations of shape (batch_size, seq_len, seq_len)
        - statistics: Dictionary with correlation statistics
    
    Raises:
        ValueError: If correlation_strength or decoherence_rate are out of valid range
        
    Example:
        >>> embeddings = torch.randn(2, 10, 768)
        >>> correlations, stats = compute_entanglement_correlation(embeddings)
        >>> print(f"Average correlation: {stats['average_correlation']:.4f}")
    """
    if not (0.0 <= correlation_strength <= 1.0):
        raise ValueError(f"correlation_strength must be in [0.0, 1.0], got {correlation_strength}")
    
    # Implementation here...
    return correlation_matrix, statistics
```

### Documentation Style

- Use **Markdown** for documentation files
- Include **code examples** in docstrings
- Add **mathematical notation** using LaTeX for complex formulas
- Provide **complete usage examples** for new features

## 🧪 Testing Guidelines

### Test Structure

- Place tests in `tests/` directory
- Mirror the package structure in test files
- Use descriptive test names that explain what is being tested

### Test Example:

```python
class TestEntangledEmbedding:
    """Test cases for EntangledEmbedding class."""
    
    @pytest.fixture
    def embedding_layer(self):
        """Create test EntangledEmbedding instance."""
        return EntangledEmbedding(
            vocab_size=1000,
            embedding_dim=128,
            entanglement_depth=2
        )
    
    def test_forward_pass_shape(self, embedding_layer):
        """Test that forward pass returns correct tensor shapes."""
        input_ids = torch.randint(0, 1000, (2, 10))
        embeddings, correlations = embedding_layer(input_ids, return_correlations=True)
        
        assert embeddings.shape == (2, 10, 128)
        assert correlations.shape == (2, 10, 10)
    
    def test_correlation_symmetry(self, embedding_layer):
        """Test that correlation matrices are symmetric."""
        input_ids = torch.randint(0, 1000, (1, 5))
        _, correlations = embedding_layer(input_ids, return_correlations=True)
        
        assert torch.allclose(correlations, correlations.transpose(-2, -1))
```

### Performance Tests

Include benchmarks for performance-critical functions:

```python
def test_entanglement_computation_performance(benchmark):
    """Benchmark entanglement correlation computation."""
    embedder = EntangledEmbedding(vocab_size=10000, embedding_dim=768)
    input_ids = torch.randint(0, 10000, (8, 512))  # Large batch
    
    def compute_entanglement():
        return embedder(input_ids, return_correlations=True)
    
    result = benchmark(compute_entanglement)
    embeddings, correlations = result
    
    # Verify correctness
    assert embeddings.shape == (8, 512, 768)
    assert correlations.shape == (8, 512, 512)
```

## 📚 Documentation Contributions

### Documentation Structure

- **API docs**: Comprehensive function/class documentation
- **Tutorials**: Step-by-step guides for common use cases
- **Theory**: Mathematical foundations and quantum concepts
- **Examples**: Practical implementation examples

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Serve docs locally
mkdocs serve

# Build static documentation
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

## 🚀 Pull Request Process

### Before Submitting

1. **Update tests** for any new functionality
2. **Update documentation** for API changes
3. **Run full test suite** and ensure all tests pass
4. **Check code style** with automated tools
5. **Update CHANGELOG.md** if applicable

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have run the performance benchmarks and verified no significant regressions

## Documentation
- [ ] I have updated the documentation accordingly
- [ ] I have added docstrings for any new functions/classes
- [ ] I have updated the API reference if needed

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings
```

### Review Process

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by at least one maintainer
3. **Documentation review** for user-facing changes
4. **Performance review** for optimization PRs

## 🐛 Bug Reports

### Creating Good Bug Reports

Include the following information:

```markdown
**Environment:**
- OS: [e.g., Ubuntu 20.04, Windows 10, macOS 12.0]
- Python version: [e.g., 3.9.7]
- PyTorch version: [e.g., 1.12.0]
- Package version: [e.g., 1.0.0]

**Description:**
Clear and concise description of the bug.

**Steps to Reproduce:**
1. Initialize model with parameters...
2. Call function with inputs...
3. Observe error/unexpected behavior...

**Expected Behavior:**
Description of what you expected to happen.

**Actual Behavior:**
Description of what actually happened.

**Code Sample:**
```python
# Minimal code sample that reproduces the issue
from entanglement_enhanced_nlp import EntangledEmbedding
embedder = EntangledEmbedding(vocab_size=1000, embedding_dim=768)
# ... rest of code
```

**Error Output:**
```
Full error traceback here
```

**Additional Context:**
Any other relevant information.
```

## 🌟 Feature Requests

### Proposing New Features

1. **Check existing issues** to avoid duplicates
2. **Create detailed proposal** with use cases
3. **Discuss implementation approach** with maintainers
4. **Consider backward compatibility** implications

### Feature Request Template

```markdown
**Feature Description:**
Clear description of the proposed feature.

**Motivation:**
Why is this feature needed? What problem does it solve?

**Proposed Implementation:**
High-level description of how this could be implemented.

**Use Cases:**
Specific examples of how this feature would be used.

**Alternatives Considered:**
Other approaches you've considered.

**Additional Context:**
Links to papers, related issues, etc.
```

## 🏆 Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md** file
- **GitHub releases** notes
- **Documentation acknowledgments**
- **Academic citations** (for significant contributions)

## 📞 Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: [bajpaikrishna715@gmail.com](mailto:bajpaikrishna715@gmail.com) for direct contact

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the advancement of quantum-inspired NLP! 🚀
