# Examples and Tutorials

This section provides practical examples and tutorials for using the Entanglement-Enhanced NLP framework in various scenarios.

## 🚀 Quick Start Examples

### Basic Entangled Embeddings

```python
import torch
from entanglement_enhanced_nlp import EntangledEmbedding

# Initialize entangled embeddings
embedder = EntangledEmbedding(
    vocab_size=10000,
    embedding_dim=768,
    entanglement_depth=3,
    correlation_strength=0.8
)

# Sample text tokens
text_tokens = torch.randint(0, 10000, (1, 20))  # Batch size 1, sequence length 20

# Get entangled embeddings with correlations
embeddings, correlations = embedder(text_tokens, return_correlations=True)

print(f"Embeddings shape: {embeddings.shape}")  # [1, 20, 768]
print(f"Correlations shape: {correlations.shape}")  # [1, 20, 20]

# Analyze entanglement statistics
stats = embedder.get_entanglement_statistics(text_tokens)
for metric, value in stats.items():
    print(f"{metric}: {value:.4f}")
```

### Quantum Contextualizer

```python
from entanglement_enhanced_nlp import QuantumContextualizer

# Initialize quantum contextualizer
contextualizer = QuantumContextualizer(
    hidden_dim=768,
    num_qubits=8,
    evolution_steps=5,
    decoherence_rate=0.1
)

# Apply quantum contextualization
quantum_embeddings = contextualizer(embeddings)

# Measure quantum states
measurement_results = contextualizer.measure_quantum_states(quantum_embeddings)
print(f"Measurement entropy: {measurement_results['entropy']:.4f}")
```

## 🔄 HuggingFace Integration

### Quantum-Enhanced BERT

```python
from transformers import BertModel, BertTokenizer
from entanglement_enhanced_nlp import EntangledTransformer
import torch

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Create quantum-enhanced version
quantum_bert = EntangledTransformer(
    base_model=bert_model,
    entanglement_layers=[6, 9, 11],  # Add entanglement to layers 6, 9, 11
    correlation_strength=0.7
)

# Process text
text = "Quantum mechanics reveals fascinating non-local correlations between particles."
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# Get quantum-enhanced representations
with torch.no_grad():
    outputs = quantum_bert(**inputs)
    quantum_embeddings = outputs.last_hidden_state

print(f"Quantum BERT output shape: {quantum_embeddings.shape}")

# Extract entanglement correlations
correlations = quantum_bert.get_entanglement_correlations()
print(f"Average entanglement: {correlations.mean().item():.4f}")
```

### Fine-tuning for Classification

```python
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

class QuantumTextClassifier(nn.Module):
    def __init__(self, quantum_bert, num_classes, hidden_dim=768):
        super().__init__()
        self.quantum_bert = quantum_bert
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.quantum_bert(input_ids, attention_mask=attention_mask)
        # Use CLS token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

# Initialize classifier
classifier = QuantumTextClassifier(quantum_bert, num_classes=2)
optimizer = AdamW(classifier.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
classifier.train()
for epoch in range(3):
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        outputs = classifier(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Average Loss: {total_loss/len(train_dataloader):.4f}")
```

## 📊 Analysis and Visualization

### Correlation Analysis

```python
from entanglement_enhanced_nlp.analysis import CorrelationAnalyzer
from entanglement_enhanced_nlp.visualization import EntanglementVisualizer
import matplotlib.pyplot as plt

# Initialize analyzer and visualizer
analyzer = CorrelationAnalyzer()
visualizer = EntanglementVisualizer()

# Sample text for analysis
sample_text = [
    "quantum", "entanglement", "enables", "non-local", 
    "correlations", "between", "distant", "particles"
]

# Tokenize and get embeddings
token_ids = torch.tensor([tokenizer.convert_tokens_to_ids(sample_text)])
embeddings, correlations = embedder(token_ids, return_correlations=True)

# Analyze correlations
mutual_info = analyzer.compute_mutual_information(embeddings)
entanglement_entropy = analyzer.compute_entanglement_entropy(correlations)

print(f"Mutual Information shape: {mutual_info.shape}")
print(f"Entanglement Entropy: {entanglement_entropy:.4f}")

# Create visualizations
fig1 = visualizer.plot_correlation_heatmap(
    correlations[0],  # First batch item
    token_labels=sample_text,
    title="Token Entanglement Correlations"
)

fig2 = visualizer.create_entanglement_network(
    correlations[0],
    token_labels=sample_text,
    threshold=0.3,
    layout='spring'
)

plt.show()
```

### Evolution Tracking

```python
# Track quantum state evolution
initial_embeddings = embeddings.clone()
evolution_data = []

# Apply evolution step by step
current_state = initial_embeddings
for step in range(5):
    evolved_state = contextualizer._apply_evolution_step(current_state, step)
    
    # Compute evolution metrics
    fidelity = torch.abs(torch.sum(current_state * evolved_state.conj(), dim=-1))
    entropy = analyzer.compute_entanglement_entropy(evolved_state)
    
    evolution_data.append({
        'step': step,
        'fidelity': fidelity.mean().item(),
        'entropy': entropy,
        'state': evolved_state.clone()
    })
    
    current_state = evolved_state

# Visualize evolution
fig = visualizer.plot_quantum_evolution(
    evolution_data,
    metrics=['fidelity', 'entropy']
)
plt.show()
```

## 🌐 Multilingual Applications

### Cross-lingual Entanglement

```python
from transformers import AutoTokenizer, AutoModel

# Load multilingual model
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = AutoModel.from_pretrained('bert-base-multilingual-cased')

# Create quantum-enhanced multilingual model
multilingual_quantum = EntangledTransformer(
    base_model=bert_model,
    entanglement_layers=[8, 10],
    correlation_strength=0.85
)

# Process texts in different languages
texts = {
    'en': "Quantum entanglement is a fascinating phenomenon.",
    'es': "El entrelazamiento cuántico es un fenómeno fascinante.",
    'fr': "L'intrication quantique est un phénomène fascinant.",
    'de': "Quantenverschränkung ist ein faszinierendes Phänomen."
}

# Analyze cross-lingual correlations
embeddings_by_lang = {}
for lang, text in texts.items():
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = multilingual_quantum(**inputs)
        embeddings_by_lang[lang] = outputs.last_hidden_state

# Compute cross-lingual similarities
lang_pairs = [('en', 'es'), ('en', 'fr'), ('en', 'de')]
for lang1, lang2 in lang_pairs:
    emb1 = embeddings_by_lang[lang1].mean(dim=1)  # Average over tokens
    emb2 = embeddings_by_lang[lang2].mean(dim=1)
    
    similarity = torch.cosine_similarity(emb1, emb2, dim=-1)
    print(f"{lang1}-{lang2} similarity: {similarity.item():.4f}")
```

## 🎯 Domain-Specific Applications

### Scientific Literature Analysis

```python
# Configure for scientific domain
scientific_config = {
    'vocab_size': 50000,
    'embedding_dim': 1024,
    'entanglement_depth': 4,
    'correlation_strength': 0.9,
    'decoherence_rate': 0.05
}

scientific_embedder = EntangledEmbedding(**scientific_config)

# Sample scientific text
scientific_text = """
The quantum entanglement between photons demonstrates non-locality 
in quantum mechanics. Bell's theorem shows that no physical theory 
can simultaneously satisfy locality and realism while reproducing 
all predictions of quantum mechanics.
"""

# Tokenize and analyze
inputs = tokenizer(scientific_text, return_tensors='pt')
embeddings, correlations = scientific_embedder(inputs['input_ids'], return_correlations=True)

# Identify key scientific concepts
token_importance = correlations.mean(dim=-1).squeeze()
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())

# Sort by importance
token_scores = list(zip(tokens, token_importance.tolist()))
token_scores.sort(key=lambda x: x[1], reverse=True)

print("Most entangled scientific concepts:")
for token, score in token_scores[:10]:
    if not token.startswith('[') and len(token) > 2:
        print(f"  {token}: {score:.4f}")
```

### Sentiment Analysis with Quantum Enhancement

```python
from sklearn.metrics import accuracy_score, classification_report

class QuantumSentimentAnalyzer(nn.Module):
    def __init__(self, quantum_embedder, hidden_dim=768):
        super().__init__()
        self.quantum_embedder = quantum_embedder
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)  # Positive, Negative, Neutral
        )
    
    def forward(self, input_ids):
        embeddings, _ = self.quantum_embedder(input_ids)
        # Global average pooling
        pooled = embeddings.mean(dim=1)
        return self.sentiment_head(pooled)

# Initialize sentiment analyzer
sentiment_analyzer = QuantumSentimentAnalyzer(embedder)

# Example training data
train_texts = [
    "I love this quantum-enhanced approach!",
    "This method is terrible and confusing.",
    "The results are okay, nothing special.",
    "Absolutely fantastic breakthrough in NLP!",
    "Disappointing performance on our dataset."
]
train_labels = torch.tensor([0, 1, 2, 0, 1])  # 0: Positive, 1: Negative, 2: Neutral

# Simple training loop
optimizer = AdamW(sentiment_analyzer.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    optimizer.zero_grad()
    
    # Tokenize batch
    batch_inputs = tokenizer(train_texts, return_tensors='pt', padding=True, truncation=True)
    
    # Forward pass
    outputs = sentiment_analyzer(batch_inputs['input_ids'])
    loss = criterion(outputs, train_labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluation
sentiment_analyzer.eval()
with torch.no_grad():
    test_inputs = tokenizer(train_texts, return_tensors='pt', padding=True, truncation=True)
    predictions = sentiment_analyzer(test_inputs['input_ids'])
    predicted_labels = torch.argmax(predictions, dim=1)
    
    accuracy = accuracy_score(train_labels.cpu(), predicted_labels.cpu())
    print(f"Training Accuracy: {accuracy:.4f}")
```

## 🧪 Advanced Quantum Operations

### Custom Quantum Gates

```python
from entanglement_enhanced_nlp.utils.quantum_simulator import QuantumSimulator

# Create custom quantum gates
def create_custom_entangling_gate(theta):
    """Create parameterized entangling gate."""
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    gate = torch.tensor([
        [cos_theta, 0, 0, sin_theta],
        [0, cos_theta, -sin_theta, 0],
        [0, sin_theta, cos_theta, 0],
        [-sin_theta, 0, 0, cos_theta]
    ], dtype=torch.complex64)
    
    return gate

# Apply custom gates to embeddings
simulator = QuantumSimulator()
theta = torch.tensor(0.5)  # Gate parameter
custom_gate = create_custom_entangling_gate(theta)

# Convert embeddings to quantum states
quantum_states = simulator.embedding_to_quantum_state(embeddings)

# Apply custom gate
evolved_states = simulator.apply_quantum_gate(
    quantum_states, 
    custom_gate, 
    qubit_indices=[0, 1]
)

# Convert back to embeddings
evolved_embeddings = simulator.quantum_state_to_embedding(evolved_states)
```

### Quantum Error Correction

```python
class QuantumErrorCorrection(nn.Module):
    def __init__(self, embedding_dim, correction_strength=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.correction_strength = correction_strength
        
        # Error syndrome detection
        self.syndrome_detector = nn.Linear(embedding_dim, embedding_dim // 4)
        self.error_corrector = nn.Linear(embedding_dim // 4, embedding_dim)
    
    def forward(self, embeddings):
        # Detect error syndromes
        syndromes = torch.tanh(self.syndrome_detector(embeddings))
        
        # Generate corrections
        corrections = self.error_corrector(syndromes)
        
        # Apply corrections
        corrected_embeddings = embeddings + self.correction_strength * corrections
        
        return corrected_embeddings

# Apply error correction
error_corrector = QuantumErrorCorrection(768)
corrected_embeddings = error_corrector(quantum_embeddings)

print(f"Error correction applied. Shape: {corrected_embeddings.shape}")
```

## 🚀 Performance Optimization

### Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class OptimizedQuantumModel(nn.Module):
    def __init__(self, embedder, contextualizer):
        super().__init__()
        self.embedder = embedder
        self.contextualizer = contextualizer
        
        # Enable gradient checkpointing
        self.embedder.gradient_checkpointing_enable()
        
    def forward(self, input_ids):
        # Use gradient checkpointing for memory efficiency
        embeddings = checkpoint(self.embedder, input_ids, use_reentrant=False)
        quantum_embeddings = checkpoint(self.contextualizer, embeddings, use_reentrant=False)
        return quantum_embeddings

# Usage
optimized_model = OptimizedQuantumModel(embedder, contextualizer)
output = optimized_model(text_tokens)
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize mixed precision scaler
scaler = GradScaler()

# Training with mixed precision
for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        embeddings, correlations = embedder(batch['input_ids'], return_correlations=True)
        quantum_embeddings = contextualizer(embeddings)
        loss = compute_loss(quantum_embeddings, batch['targets'])
    
    # Scale loss and backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## 📝 Custom Extensions

### Domain-Specific Embeddings

```python
class DomainSpecificEntanglement(nn.Module):
    def __init__(self, base_embedder, domain_vocab, domain_correlation_strength=0.9):
        super().__init__()
        self.base_embedder = base_embedder
        self.domain_vocab = set(domain_vocab)
        self.domain_correlation_strength = domain_correlation_strength
        
        # Domain-specific entanglement parameters
        self.domain_entanglement_matrix = nn.Parameter(
            torch.randn(len(domain_vocab), len(domain_vocab)) * 0.1
        )
        self.vocab_to_idx = {word: idx for idx, word in enumerate(domain_vocab)}
    
    def forward(self, input_ids, token_strings=None):
        # Get base embeddings
        base_embeddings, correlations = self.base_embedder(input_ids, return_correlations=True)
        
        if token_strings is not None:
            # Apply domain-specific enhancements
            for i, token in enumerate(token_strings):
                if token in self.domain_vocab:
                    domain_idx = self.vocab_to_idx[token]
                    # Enhance with domain-specific correlations
                    domain_correlation = self.domain_entanglement_matrix[domain_idx]
                    base_embeddings[:, i, :] += 0.1 * domain_correlation.mean() * base_embeddings[:, i, :]
        
        return base_embeddings, correlations

# Usage for medical domain
medical_vocab = ["diagnosis", "treatment", "patient", "symptom", "therapy", "medication"]
medical_embedder = DomainSpecificEntanglement(embedder, medical_vocab)

medical_text = ["patient", "shows", "symptoms", "requiring", "immediate", "treatment"]
token_ids = torch.tensor([tokenizer.convert_tokens_to_ids(medical_text)])
enhanced_embeddings, _ = medical_embedder(token_ids, medical_text)
```

This comprehensive set of examples demonstrates the versatility and power of the Entanglement-Enhanced NLP framework across various applications, from basic usage to advanced quantum operations and domain-specific customizations.
