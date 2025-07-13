# Usage Guide

This guide provides comprehensive examples and tutorials for using the Entanglement-Enhanced NLP framework.

## 🚀 Quick Start

### Basic EntangledEmbedding Usage

```python
import torch
from entanglement_enhanced_nlp import EntangledEmbedding

# Initialize entangled embeddings
embedder = EntangledEmbedding(
    vocab_size=10000,
    embedding_dim=768,
    entanglement_depth=3,
    correlation_strength=0.8,
    decoherence_rate=0.1
)

# Create sample input (batch_size=2, sequence_length=10)
input_ids = torch.randint(0, 10000, (2, 10))

# Get entangled embeddings
embeddings, correlations = embedder(input_ids, return_correlations=True)

print(f"Embeddings shape: {embeddings.shape}")  # (2, 10, 768)
print(f"Correlations shape: {correlations.shape}")  # (2, 10, 10)
```

### Quantum Contextualizer Integration

```python
from entanglement_enhanced_nlp import QuantumContextualizer

# Initialize quantum contextualizer
contextualizer = QuantumContextualizer(
    hidden_dim=768,
    num_qubits=8,
    decoherence_rate=0.1,
    evolution_steps=5
)

# Apply quantum contextualization
quantum_embeddings = contextualizer(embeddings)
print(f"Quantum embeddings shape: {quantum_embeddings.shape}")  # (2, 10, 768)
```

## 🔗 Advanced Entanglement Features

### Entanglement Statistics Analysis

```python
# Get detailed entanglement statistics
stats = embedder.get_entanglement_statistics(input_ids)

print("Entanglement Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value:.4f}")

# Output:
# average_correlation: 0.3421
# maximum_correlation: 0.8934
# entanglement_entropy: 2.1567
# quantum_coherence: 0.7234
# decoherence_rate: 0.1000
# correlation_strength: 0.8000
```

### Custom Entanglement Parameters

```python
# Create embedder with custom entanglement configuration
custom_embedder = EntangledEmbedding(
    vocab_size=5000,
    embedding_dim=512,
    entanglement_depth=5,        # Deeper entanglement layers
    correlation_strength=0.9,    # Stronger correlations
    decoherence_rate=0.05,      # Slower decoherence
    max_position_embeddings=1024 # Longer sequences
)

# Process text with enhanced entanglement
text_tokens = torch.randint(0, 5000, (1, 50))
enhanced_embeddings, _ = custom_embedder(text_tokens)
```

## 🌊 Quantum State Evolution

### Controlled Evolution Steps

```python
from entanglement_enhanced_nlp.core.quantum_contextualizer import QuantumContextualizer

# Initialize with specific evolution parameters
contextualizer = QuantumContextualizer(
    hidden_dim=768,
    num_qubits=12,
    evolution_steps=10,
    measurement_basis='computational'
)

# Apply step-by-step evolution
embeddings = torch.randn(1, 20, 768)
evolved_states = []

for step in range(5):
    embeddings = contextualizer._apply_evolution_step(embeddings, step)
    evolved_states.append(embeddings.clone())
    
print(f"Evolution trajectory: {len(evolved_states)} steps")
```

### Quantum Measurement Effects

```python
# Simulate quantum measurements
measurement_results = contextualizer.measure_quantum_states(embeddings)

print("Measurement Results:")
print(f"  Probabilities shape: {measurement_results['probabilities'].shape}")
print(f"  Collapsed states shape: {measurement_results['collapsed_states'].shape}")
print(f"  Measurement entropy: {measurement_results['entropy']:.4f}")
```

## 🔄 HuggingFace Integration

### Extending BERT with Entanglement

```python
from transformers import BertModel, BertTokenizer
from entanglement_enhanced_nlp import EntangledTransformer

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Create entangled transformer wrapper
entangled_bert = EntangledTransformer(
    base_model=bert_model,
    entanglement_layers=[6, 9, 11],  # Add entanglement to specific layers
    correlation_strength=0.7
)

# Process text with quantum-enhanced BERT
text = "Quantum entanglement enables non-local correlations in language."
inputs = tokenizer(text, return_tensors='pt')

# Get quantum-enhanced outputs
outputs = entangled_bert(**inputs)
quantum_embeddings = outputs.last_hidden_state

print(f"Quantum BERT output shape: {quantum_embeddings.shape}")
```

### Fine-tuning with Entanglement

```python
import torch.nn as nn
from torch.optim import AdamW

# Create classification head with entangled features
class EntangledClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.entangled_bert = entangled_bert
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.entangled_bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled_output)

# Initialize classifier
classifier = EntangledClassifier(hidden_dim=768, num_classes=2)
optimizer = AdamW(classifier.parameters(), lr=2e-5)

# Training loop with entanglement
classifier.train()
for batch in train_dataloader:
    optimizer.zero_grad()
    
    outputs = classifier(batch['input_ids'], batch['attention_mask'])
    loss = nn.CrossEntropyLoss()(outputs, batch['labels'])
    
    loss.backward()
    optimizer.step()
```

## 📊 Visualization and Analysis

### Entanglement Heatmaps

```python
from entanglement_enhanced_nlp.visualization import EntanglementVisualizer

# Initialize visualizer
visualizer = EntanglementVisualizer()

# Create entanglement heatmap
text_tokens = ["quantum", "entanglement", "enables", "non-local", "correlations"]
embeddings, correlations = embedder(torch.tensor([tokenizer.convert_tokens_to_ids(text_tokens)]))

# Generate visualization
fig = visualizer.plot_correlation_heatmap(
    correlations[0],  # First batch item
    token_labels=text_tokens,
    title="Token Entanglement Correlations"
)
fig.show()
```

### Quantum State Evolution Tracking

```python
# Track quantum state evolution over time
evolution_data = visualizer.track_quantum_evolution(
    initial_embeddings=embeddings,
    contextualizer=contextualizer,
    steps=10
)

# Plot evolution trajectory
fig = visualizer.plot_evolution_trajectory(
    evolution_data,
    metrics=['entropy', 'coherence', 'entanglement']
)
fig.show()
```

### Entanglement Network Graphs

```python
# Create entanglement network visualization
network_fig = visualizer.create_entanglement_network(
    correlations[0],
    token_labels=text_tokens,
    threshold=0.3,  # Only show correlations > 0.3
    layout='spring'
)
network_fig.show()
```

## 🛠 Custom Extensions

### Creating Custom Entanglement Patterns

```python
import torch
import torch.nn as nn

class CustomEntanglementPattern(nn.Module):
    """Custom entanglement pattern for domain-specific applications."""
    
    def __init__(self, embedding_dim, pattern_type='circular'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pattern_type = pattern_type
        
        # Define custom entanglement matrices
        if pattern_type == 'circular':
            self.entanglement_matrix = self._create_circular_pattern()
        elif pattern_type == 'hierarchical':
            self.entanglement_matrix = self._create_hierarchical_pattern()
    
    def _create_circular_pattern(self):
        """Create circular entanglement pattern."""
        matrix = torch.zeros(self.embedding_dim, self.embedding_dim)
        for i in range(self.embedding_dim):
            next_idx = (i + 1) % self.embedding_dim
            matrix[i, next_idx] = 1.0
            matrix[next_idx, i] = 1.0
        return nn.Parameter(matrix)
    
    def forward(self, embeddings):
        """Apply custom entanglement pattern."""
        return torch.matmul(embeddings, self.entanglement_matrix)

# Use custom pattern with EntangledEmbedding
custom_pattern = CustomEntanglementPattern(768, 'circular')
embedder.register_custom_pattern(custom_pattern)
```

### Implementing Quantum Error Correction

```python
class QuantumErrorCorrection(nn.Module):
    """Implement quantum error correction for embeddings."""
    
    def __init__(self, embedding_dim, correction_strength=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.correction_strength = correction_strength
        
        # Error correction parameters
        self.syndrome_detector = nn.Linear(embedding_dim, embedding_dim // 4)
        self.error_corrector = nn.Linear(embedding_dim // 4, embedding_dim)
    
    def forward(self, embeddings):
        """Apply quantum error correction."""
        # Detect error syndromes
        syndromes = torch.tanh(self.syndrome_detector(embeddings))
        
        # Generate corrections
        corrections = self.error_corrector(syndromes)
        
        # Apply corrections with strength parameter
        corrected_embeddings = embeddings + self.correction_strength * corrections
        
        return corrected_embeddings

# Integrate error correction
error_corrector = QuantumErrorCorrection(768)
corrected_embeddings = error_corrector(quantum_embeddings)
```

## 🎯 Best Practices

### Performance Optimization

```python
# Use mixed precision for faster training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    embeddings, correlations = embedder(input_ids, return_correlations=True)
    quantum_embeddings = contextualizer(embeddings)
    
# Scale gradients for stability
loss = compute_loss(quantum_embeddings, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Memory Management

```python
# Use gradient checkpointing for large models
embedder.gradient_checkpointing_enable()
contextualizer.gradient_checkpointing_enable()

# Clear intermediate states when not needed
embeddings, _ = embedder(input_ids, return_correlations=False)  # Skip correlations
del intermediate_states  # Explicitly delete large tensors
```

### Hyperparameter Tuning

```python
# Optimal parameter ranges based on experiments
optimal_configs = {
    'small_models': {
        'entanglement_depth': 2,
        'correlation_strength': 0.6,
        'decoherence_rate': 0.15
    },
    'large_models': {
        'entanglement_depth': 4,
        'correlation_strength': 0.8,
        'decoherence_rate': 0.05
    },
    'multilingual': {
        'entanglement_depth': 3,
        'correlation_strength': 0.9,
        'decoherence_rate': 0.1
    }
}

# Select configuration based on use case
config = optimal_configs['large_models']
embedder = EntangledEmbedding(**config, vocab_size=vocab_size, embedding_dim=768)
```

## 🔍 Debugging and Troubleshooting

### Monitoring Entanglement Health

```python
# Check for common issues
def diagnose_entanglement(embedder, input_ids):
    stats = embedder.get_entanglement_statistics(input_ids)
    
    issues = []
    if stats['average_correlation'] < 0.1:
        issues.append("Low correlation - increase correlation_strength")
    if stats['quantum_coherence'] < 0.3:
        issues.append("Low coherence - decrease decoherence_rate")
    if stats['entanglement_entropy'] > 3.0:
        issues.append("High entropy - reduce entanglement_depth")
    
    return issues

# Run diagnostics
issues = diagnose_entanglement(embedder, input_ids)
for issue in issues:
    print(f"⚠️  {issue}")
```

### Gradient Flow Analysis

```python
# Check gradient magnitudes
def analyze_gradients(model):
    total_norm = 0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            print(f"{name}: {param_norm.item():.6f}")
    
    total_norm = total_norm ** (1. / 2)
    print(f"Total gradient norm: {total_norm:.6f}")
    return total_norm

# Monitor gradients during training
gradient_norm = analyze_gradients(embedder)
```

This usage guide provides comprehensive examples for leveraging the quantum-inspired features of the Entanglement-Enhanced NLP framework. For more advanced topics, check out the [API Reference](api.md) and [Theory](theory.md) sections.
