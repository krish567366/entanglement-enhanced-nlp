# Installation Guide

## 📦 Quick Installation

Install the latest stable version from PyPI:

```bash
pip install entanglement-enhanced-nlp
```

## 🔧 System Requirements

### Python Version

- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- Compatible with Python 3.8, 3.9, 3.10, 3.11

### Operating Systems

- **Linux** (Ubuntu 18.04+, CentOS 7+)
- **macOS** (10.14+)
- **Windows** (10+)

### Hardware Requirements

- **Minimum**: 4GB RAM, 2GB disk space
- **Recommended**: 8GB+ RAM, GPU support for large models
- **Optional**: CUDA-compatible GPU for accelerated processing

## 📋 Dependencies

### Core Dependencies

The package automatically installs these required dependencies:

```txt
torch>=1.9.0
numpy>=1.21.0
transformers>=4.20.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
click>=8.0.0
pyyaml>=6.0
tqdm>=4.62.0
```

### Optional Dependencies

#### For Advanced Quantum Simulation

```bash
pip install pennylane>=0.30.0
```

#### For Enhanced Visualization

```bash
pip install plotly>=5.0.0
pip install networkx>=2.6.0
```

#### For Development

```bash
pip install pytest>=7.0.0
pip install pytest-cov>=4.0.0
pip install black>=22.0.0
pip install flake8>=5.0.0
```

## 🛠 Installation Methods

### Method 1: PyPI Installation (Recommended)

```bash
# Install the latest stable version
pip install entanglement-enhanced-nlp

# Install with optional dependencies
pip install "entanglement-enhanced-nlp[quantum]"
pip install "entanglement-enhanced-nlp[viz]"
pip install "entanglement-enhanced-nlp[dev]"
pip install "entanglement-enhanced-nlp[all]"
```

### Method 2: Development Installation

```bash
# Clone the repository
git clone https://github.com/krish567366/entanglement-enhanced-nlp.git
cd entanglement-enhanced-nlp

# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Method 3: Conda Environment

```bash
# Create a new conda environment
conda create -n eenlp python=3.9
conda activate eenlp

# Install the package
pip install entanglement-enhanced-nlp
```

### Method 4: Docker Installation

```bash
# Pull the pre-built Docker image
docker pull krish567366/entanglement-enhanced-nlp:latest

# Or build from source
git clone https://github.com/krish567366/entanglement-enhanced-nlp.git
cd entanglement-enhanced-nlp
docker build -t eenlp .
```

## ✅ Verification

### Quick Test

After installation, verify everything works:

```python
import entanglement_enhanced_nlp as eenlp
print(f"EENLP version: {eenlp.__version__}")

# Test basic functionality
from entanglement_enhanced_nlp import EntangledEmbedding
embedder = EntangledEmbedding(vocab_size=1000, embedding_dim=128)
print("✅ EntangledEmbedding initialized successfully!")
```

### CLI Tool Test

```bash
# Check CLI installation
eenlp-cli --version

# Run basic help
eenlp-cli --help
```

### Run Test Suite

```bash
# Install test dependencies
pip install pytest

# Run tests
pytest tests/ -v
```

## 🐛 Troubleshooting

### Common Issues

#### ImportError: No module named 'torch'

```bash
# Install PyTorch first
pip install torch torchvision torchaudio
```

#### CUDA Compatibility Issues

```bash
# Install CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or install specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Permission Errors on Linux/macOS

```bash
# Use user installation
pip install --user entanglement-enhanced-nlp

# Or use virtual environment
python -m venv eenlp_env
source eenlp_env/bin/activate  # On Windows: eenlp_env\Scripts\activate
pip install entanglement-enhanced-nlp
```

#### Memory Issues

```bash
# For systems with limited memory, install minimal version
pip install entanglement-enhanced-nlp --no-deps
pip install torch numpy click  # Install only essential dependencies
```

### Platform-Specific Notes

#### Windows

- Ensure Visual Studio Build Tools are installed for some dependencies
- Use Anaconda/Miniconda for easier dependency management

#### macOS

- May require Xcode Command Line Tools: `xcode-select --install`
- Use Homebrew for system-level dependencies if needed

#### Linux

- Ubuntu/Debian: `sudo apt-get install python3-dev python3-pip`
- CentOS/RHEL: `sudo yum install python3-devel python3-pip`

## 🔄 Upgrading

### Upgrade to Latest Version

```bash
pip install --upgrade entanglement-enhanced-nlp
```

### Check Current Version

```python
import entanglement_enhanced_nlp
print(entanglement_enhanced_nlp.__version__)
```

### Version Compatibility

- **v0.1.x**: Initial release, basic entanglement features
- **v0.2.x**: Added quantum contextualizer and CLI tools
- **v1.0.x**: Stable API, full transformer integration

## 🚀 Next Steps

After successful installation:

1. **[Quick Start Guide](usage.md)**: Learn basic usage patterns
2. **[API Documentation](api.md)**: Explore all available classes and functions
3. **[CLI Tool](cli.md)**: Master the command-line interface
4. **[Examples](examples.md)**: Try out practical examples and tutorials

## 💬 Support

If you encounter installation issues:

- **GitHub Issues**: [Report bugs and issues](https://github.com/krish567366/entanglement-enhanced-nlp/issues)
- **Discussions**: [Community support](https://github.com/krish567366/entanglement-enhanced-nlp/discussions)
- **Email**: [bajpaikrishna715@gmail.com](mailto:bajpaikrishna715@gmail.com)
