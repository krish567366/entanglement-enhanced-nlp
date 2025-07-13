# CLI Tool Reference

The `eenlp-cli` command-line interface provides powerful tools for analyzing text with quantum-enhanced NLP capabilities, processing datasets, and generating visualizations.

## 🚀 Installation and Setup

After installing the package, the CLI tool is available globally:

```bash
# Verify installation
eenlp-cli --version

# Get help
eenlp-cli --help
```

## 📝 Command Structure

```bash
eenlp-cli [GLOBAL_OPTIONS] COMMAND [COMMAND_OPTIONS] [ARGUMENTS]
```

### Global Options

- `--verbose, -v`: Enable verbose output
- `--quiet, -q`: Suppress non-essential output  
- `--config, -c`: Specify configuration file path
- `--device`: Set device (cpu/cuda/auto)
- `--version`: Show version information

## 🔧 Core Commands

### `analyze` - Text Analysis

Perform quantum-enhanced analysis on text files or direct input.

```bash
eenlp-cli analyze [OPTIONS] INPUT
```

#### Options

- `--input, -i TEXT`: Input text file path or direct text
- `--output, -o PATH`: Output file for results (JSON format)
- `--model-config PATH`: Path to model configuration file
- `--correlation-strength FLOAT`: Set entanglement correlation strength (0.0-1.0)
- `--entanglement-depth INT`: Number of entanglement layers (1-10)
- `--decoherence-rate FLOAT`: Quantum decoherence rate (0.0-1.0)
- `--return-correlations`: Include correlation matrices in output
- `--visualize`: Generate visualization plots
- `--format TEXT`: Output format (json/csv/txt)

#### Examples

**Basic text analysis:**

```bash
# Analyze text from file
eenlp-cli analyze --input document.txt --output results.json

# Analyze direct text input
eenlp-cli analyze --input "Quantum entanglement enables non-local correlations" --visualize
```

**Advanced analysis with custom parameters:**

```bash
eenlp-cli analyze \
  --input research_paper.txt \
  --output analysis_results.json \
  --correlation-strength 0.9 \
  --entanglement-depth 5 \
  --decoherence-rate 0.05 \
  --return-correlations \
  --visualize
```

**Batch processing multiple files:**

```bash
# Process all text files in directory
for file in *.txt; do
  eenlp-cli analyze --input "$file" --output "${file%.txt}_analysis.json"
done
```

#### Output Format

```json
{
  "input_text": "Sample text for analysis...",
  "statistics": {
    "average_correlation": 0.4521,
    "maximum_correlation": 0.8934,
    "entanglement_entropy": 2.1567,
    "quantum_coherence": 0.7234,
    "decoherence_rate": 0.1000,
    "correlation_strength": 0.8000
  },
  "token_analysis": [
    {
      "token": "quantum",
      "position": 0,
      "entanglement_score": 0.6789,
      "correlations": {...}
    }
  ],
  "correlations": [[...]]  // If --return-correlations is used
}
```

### `process` - Dataset Processing

Process large datasets with quantum-enhanced NLP models.

```bash
eenlp-cli process [OPTIONS] DATASET
```

#### Options

- `--dataset, -d PATH`: Dataset file path (CSV, JSON, JSONL)
- `--text-column TEXT`: Column name containing text (default: 'text')
- `--output-dir PATH`: Output directory for processed results
- `--batch-size INT`: Processing batch size (default: 32)
- `--model-config PATH`: Model configuration file
- `--save-embeddings`: Save computed embeddings
- `--save-correlations`: Save correlation matrices
- `--parallel-workers INT`: Number of parallel workers
- `--progress`: Show progress bar

#### Examples

**Process CSV dataset:**

```bash
eenlp-cli process \
  --dataset data.csv \
  --text-column "content" \
  --output-dir ./processed_results \
  --batch-size 64 \
  --save-embeddings \
  --parallel-workers 4 \
  --progress
```

**Process JSON Lines dataset:**

```bash
eenlp-cli process \
  --dataset tweets.jsonl \
  --text-column "tweet_text" \
  --output-dir ./twitter_analysis \
  --save-correlations \
  --model-config custom_config.yaml
```

#### Output Structure

```bash
processed_results/
├── embeddings/
│   ├── batch_0001.pt
│   ├── batch_0002.pt
│   └── ...
├── correlations/
│   ├── batch_0001_corr.pt
│   └── ...
├── statistics.json
└── summary_report.html
```

### `visualize` - Generate Visualizations

Create visual representations of entanglement patterns and quantum correlations.

```bash
eenlp-cli visualize [OPTIONS] INPUT
```

#### Options

- `--input, -i PATH`: Input results file (JSON from analyze command)
- `--output-dir PATH`: Output directory for visualizations
- `--plot-types TEXT`: Comma-separated plot types (heatmap,network,evolution,3d)
- `--correlation-threshold FLOAT`: Threshold for network visualization (default: 0.3)
- `--network-layout TEXT`: Network layout algorithm (spring/circular/kamada_kawai)
- `--colormap TEXT`: Colormap for heatmaps (viridis/plasma/coolwarm)
- `--dpi INT`: Image resolution (default: 300)
- `--format TEXT`: Image format (png/pdf/svg)
- `--interactive`: Generate interactive plots

#### Examples

**Basic visualization:**

```bash
eenlp-cli visualize --input analysis_results.json --output-dir ./plots
```

**Advanced visualization with custom settings:**

```bash
eenlp-cli visualize \
  --input analysis_results.json \
  --output-dir ./visualizations \
  --plot-types "heatmap,network,evolution" \
  --correlation-threshold 0.4 \
  --network-layout spring \
  --colormap plasma \
  --dpi 300 \
  --format pdf \
  --interactive
```

**Batch visualization for multiple analyses:**

```bash
# Visualize all analysis results
for result in *_analysis.json; do
  eenlp-cli visualize --input "$result" --output-dir "./plots/${result%.json}"
done
```

### `config` - Configuration Management

Manage configuration files and presets for different use cases.

```bash
eenlp-cli config [SUBCOMMAND] [OPTIONS]
```

#### Subcommands

**`create`** - Create new configuration file:

```bash
eenlp-cli config create --template basic --output config.yaml
eenlp-cli config create --template advanced --output advanced_config.yaml
eenlp-cli config create --template multilingual --output multilingual_config.yaml
```

**`validate`** - Validate configuration file:

```bash
eenlp-cli config validate --config config.yaml
```

**`list-templates`** - List available configuration templates:

```bash
eenlp-cli config list-templates
```

#### Configuration File Format

```yaml
# Basic configuration template
model:
  vocab_size: 50000
  embedding_dim: 768
  entanglement_depth: 3
  correlation_strength: 0.8
  decoherence_rate: 0.1
  max_position_embeddings: 512

quantum:
  num_qubits: 8
  evolution_steps: 5
  measurement_basis: "computational"
  enable_error_correction: false

processing:
  batch_size: 32
  use_gpu: true
  mixed_precision: true
  gradient_checkpointing: false

analysis:
  compute_correlations: true
  correlation_threshold: 0.1
  save_statistics: true
  generate_plots: false
```

### `serve` - API Server

Start a REST API server for quantum-enhanced NLP processing.

```bash
eenlp-cli serve [OPTIONS]
```

#### Options

- `--host TEXT`: Server host (default: localhost)
- `--port INT`: Server port (default: 8000)
- `--workers INT`: Number of worker processes
- `--model-config PATH`: Model configuration file
- `--cors`: Enable CORS headers
- `--api-key TEXT`: Require API key authentication

#### Example

```bash
# Start server with custom configuration
eenlp-cli serve \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 4 \
  --model-config production_config.yaml \
  --cors
```

#### API Endpoints

**POST /analyze**

```json
{
  "text": "Input text for analysis",
  "options": {
    "correlation_strength": 0.8,
    "return_correlations": true
  }
}
```

**POST /batch-process**

```json
{
  "texts": ["Text 1", "Text 2", "..."],
  "options": {
    "batch_size": 16,
    "save_embeddings": false
  }
}
```

**GET /health**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

## 🔧 Advanced Usage Patterns

### Pipeline Composition

Chain multiple commands for complex workflows:

```bash
#!/bin/bash
# Advanced processing pipeline

# Step 1: Process dataset
eenlp-cli process \
  --dataset raw_data.csv \
  --output-dir ./step1_processed \
  --save-embeddings

# Step 2: Analyze processed results
for batch in ./step1_processed/embeddings/*.pt; do
  eenlp-cli analyze \
    --input "$batch" \
    --output "./step2_analysis/$(basename $batch .pt)_analysis.json" \
    --visualize
done

# Step 3: Generate comprehensive visualizations
eenlp-cli visualize \
  --input "./step2_analysis/*_analysis.json" \
  --output-dir ./final_visualizations \
  --plot-types "heatmap,network,evolution" \
  --interactive
```

### Custom Scripts Integration

```python
#!/usr/bin/env python3
"""Custom script using eenlp-cli as subprocess."""

import subprocess
import json
from pathlib import Path

def analyze_document(text_file, output_dir):
    """Analyze document using CLI and return results."""
    
    # Run analysis
    result = subprocess.run([
        'eenlp-cli', 'analyze',
        '--input', str(text_file),
        '--output', f'{output_dir}/analysis.json',
        '--return-correlations',
        '--visualize'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        # Load and return results
        with open(f'{output_dir}/analysis.json') as f:
            return json.load(f)
    else:
        raise RuntimeError(f"Analysis failed: {result.stderr}")

# Usage
results = analyze_document('document.txt', './output')
print(f"Average correlation: {results['statistics']['average_correlation']}")
```

### Monitoring and Logging

Enable detailed logging for debugging and monitoring:

```bash
# Enable verbose logging
export EENLP_LOG_LEVEL=DEBUG
export EENLP_LOG_FILE=./eenlp.log

# Run with detailed logging
eenlp-cli --verbose analyze --input large_document.txt --output results.json

# Monitor progress
tail -f ./eenlp.log
```

### Performance Optimization

```bash
# GPU acceleration
eenlp-cli analyze \
  --device cuda \
  --input large_text.txt \
  --batch-size 128

# Memory-efficient processing
eenlp-cli process \
  --dataset huge_dataset.csv \
  --batch-size 16 \
  --parallel-workers 1 \
  --model-config memory_efficient.yaml

# Fast processing without correlations
eenlp-cli analyze \
  --input text.txt \
  --correlation-strength 0.0 \
  --entanglement-depth 1
```

## 🐛 Troubleshooting

### Common Issues

**Memory errors with large datasets:**

```bash
# Reduce batch size
eenlp-cli process --dataset large.csv --batch-size 8

# Use CPU instead of GPU
eenlp-cli analyze --device cpu --input text.txt
```

**CUDA out of memory:**

```bash
# Enable gradient checkpointing
eenlp-cli analyze --model-config memory_efficient.yaml --input text.txt
```

**Slow processing:**

```bash
# Disable correlations for speed
eenlp-cli analyze --correlation-strength 0.0 --input text.txt

# Use sparse correlation computation
eenlp-cli analyze --model-config sparse_config.yaml --input text.txt
```

### Debug Mode

```bash
# Enable debug output
eenlp-cli --verbose analyze --input text.txt

# Check configuration
eenlp-cli config validate --config my_config.yaml

# Test with minimal example
echo "Test text" | eenlp-cli analyze --input -
```

### Getting Help

```bash
# General help
eenlp-cli --help

# Command-specific help
eenlp-cli analyze --help
eenlp-cli process --help
eenlp-cli visualize --help

# List all available options
eenlp-cli config list-templates
```

The CLI tool provides a comprehensive interface for leveraging quantum-enhanced NLP capabilities, from simple text analysis to large-scale dataset processing and advanced visualization generation.
