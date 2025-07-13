# API Reference

Complete reference documentation for all classes, functions, and modules in the Entanglement-Enhanced NLP framework.

## 🔗 Core Modules

### EntangledEmbedding

The primary embedding layer that implements quantum entanglement-inspired mechanisms.

```python
class EntangledEmbedding(nn.Module):
    """
    Quantum-inspired embedding layer with entanglement capabilities.
    
    This layer creates embeddings that exhibit non-local correlations
    similar to quantum entanglement, enabling better semantic relationship
    modeling in natural language processing tasks.
    """
```

#### Constructor

```python
EntangledEmbedding(
    vocab_size: int,
    embedding_dim: int,
    entanglement_depth: int = 2,
    correlation_strength: float = 0.7,
    decoherence_rate: float = 0.1,
    max_position_embeddings: int = 512,
    dropout: float = 0.1,
    device: Optional[torch.device] = None
)
```

**Parameters:**

- **`vocab_size`** (*int*): Size of the vocabulary
- **`embedding_dim`** (*int*): Dimension of the embeddings
- **`entanglement_depth`** (*int, optional*): Number of entanglement layers (default: 2)
- **`correlation_strength`** (*float, optional*): Strength of quantum correlations (0.0-1.0, default: 0.7)
- **`decoherence_rate`** (*float, optional*): Rate of quantum decoherence (0.0-1.0, default: 0.1)
- **`max_position_embeddings`** (*int, optional*): Maximum sequence length (default: 512)
- **`dropout`** (*float, optional*): Dropout probability (default: 0.1)
- **`device`** (*torch.device, optional*): Device to place the module on

#### Methods

##### `forward(input_ids, return_correlations=False)`

Forward pass through the entangled embedding layer.

```python
def forward(
    self,
    input_ids: torch.Tensor,
    return_correlations: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Args:
        input_ids: Token IDs tensor of shape (batch_size, sequence_length)
        return_correlations: Whether to return correlation matrix
        
    Returns:
        Tuple of (embeddings, correlations):
        - embeddings: Shape (batch_size, sequence_length, embedding_dim)
        - correlations: Shape (batch_size, sequence_length, sequence_length) if requested
    """
```

**Example:**

```python

embedder = EntangledEmbedding(vocab_size=1000, embedding_dim=768)
input_ids = torch.randint(0, 1000, (2, 10))
embeddings, correlations = embedder(input_ids, return_correlations=True)
```

##### `get_entanglement_statistics(input_ids)`

Compute comprehensive entanglement statistics.

```python
def get_entanglement_statistics(
    self,
    input_ids: torch.Tensor
) -> Dict[str, float]:
    """
    Compute detailed statistics about entanglement patterns.
    
    Args:
        input_ids: Input token tensor
        
    Returns:
        Dictionary containing:
        - average_correlation: Mean correlation strength
        - maximum_correlation: Peak correlation value
        - entanglement_entropy: Quantum entanglement entropy
        - quantum_coherence: Measure of quantum coherence
        - decoherence_rate: Current decoherence rate
        - correlation_strength: Current correlation strength
    """
```

**Example:**

```python
stats = embedder.get_entanglement_statistics(input_ids)
print(f"Average correlation: {stats['average_correlation']:.4f}")
```

### QuantumContextualizer

Applies quantum state evolution to enhance contextual understanding.

```python
class QuantumContextualizer(nn.Module):
    """
    Quantum-inspired contextualizer using state evolution principles.
    
    This module applies quantum-like evolution operators to embeddings,
    simulating quantum state evolution to enhance contextual representations.
    """
```

#### Constructor

```python
QuantumContextualizer(
    hidden_dim: int,
    num_qubits: int = 8,
    evolution_steps: int = 5,
    decoherence_rate: float = 0.1,
    measurement_basis: str = 'computational',
    device: Optional[torch.device] = None
)
```

**Parameters:**

- **`hidden_dim`** (*int*): Dimension of input embeddings
- **`num_qubits`** (*int, optional*): Number of simulated qubits (default: 8)
- **`evolution_steps`** (*int, optional*): Number of evolution steps (default: 5)
- **`decoherence_rate`** (*float, optional*): Decoherence rate per step (default: 0.1)
- **`measurement_basis`** (*str, optional*): Measurement basis ('computational' or 'hadamard', default: 'computational')

#### Methods

##### `forward(embeddings)`

Apply quantum contextualization to embeddings.

```python
def forward(
    self,
    embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Args:
        embeddings: Input embeddings of shape (batch_size, seq_len, hidden_dim)
        attention_mask: Optional attention mask
        
    Returns:
        Quantum-enhanced embeddings of same shape as input
    """
```

##### `measure_quantum_states(embeddings)`

Perform quantum measurements on evolved states.

```python
def measure_quantum_states(
    self,
    embeddings: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Simulate quantum measurements on the evolved states.
    
    Returns:
        Dictionary containing:
        - probabilities: Measurement probabilities
        - collapsed_states: Post-measurement states
        - entropy: Measurement entropy
    """
```

### EntangledAttention

Quantum-inspired attention mechanism with entanglement correlations.

```python
class EntangledAttention(nn.Module):
    """
    Multi-head attention with quantum entanglement-inspired correlations.
    """
```

#### Constructor

```python
EntangledAttention(
    hidden_dim: int,
    num_heads: int = 8,
    entanglement_strength: float = 0.5,
    dropout: float = 0.1
)
```

#### Methods

##### `forward(hidden_states, attention_mask=None)`

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        hidden_states: Shape (batch_size, seq_len, hidden_dim)
        attention_mask: Optional attention mask
        
    Returns:
        Tuple of (attended_states, attention_weights)
    """
```

## 🔄 Transformer Integration

### EntangledTransformer

Wrapper for integrating entanglement with HuggingFace transformers.

```python
class EntangledTransformer(nn.Module):
    """
    Wrapper that adds entanglement capabilities to existing transformer models.
    """
```

#### Constructor

```python
EntangledTransformer(
    base_model: nn.Module,
    entanglement_layers: List[int],
    correlation_strength: float = 0.7,
    enable_quantum_attention: bool = True
)
```

**Parameters:**

- **`base_model`** (*nn.Module*): Base transformer model (e.g., BERT, GPT)
- **`entanglement_layers`** (*List[int]*): Layer indices to add entanglement to
- **`correlation_strength`** (*float*): Strength of entanglement correlations
- **`enable_quantum_attention`** (*bool*): Whether to use quantum attention

## 🛠 Utility Modules

### QuantumSimulator

Backend simulator for quantum-like operations.

```python
class QuantumSimulator:
    """
    Classical simulator for quantum-inspired operations.
    """
```

#### Methods

##### `create_entangled_state(num_qubits, entanglement_type='bell')`

```python
@staticmethod
def create_entangled_state(
    num_qubits: int,
    entanglement_type: str = 'bell'
) -> torch.Tensor:
    """
    Create an entangled quantum state.
    
    Args:
        num_qubits: Number of qubits in the system
        entanglement_type: Type of entanglement ('bell', 'ghz', 'w')
        
    Returns:
        Quantum state vector representing the entangled state
    """
```

##### `apply_quantum_gate(state, gate, qubit_indices)`

```python
@staticmethod
def apply_quantum_gate(
    state: torch.Tensor,
    gate: torch.Tensor,
    qubit_indices: List[int]
) -> torch.Tensor:
    """
    Apply a quantum gate to specific qubits.
    
    Args:
        state: Current quantum state
        gate: Gate matrix to apply
        qubit_indices: Indices of qubits to apply gate to
        
    Returns:
        New quantum state after gate application
    """
```

##### `measure_state(state, basis='computational')`

```python
@staticmethod
def measure_state(
    state: torch.Tensor,
    basis: str = 'computational'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform quantum measurement on state.
    
    Args:
        state: Quantum state to measure
        basis: Measurement basis
        
    Returns:
        Tuple of (measurement_probabilities, collapsed_state)
    """
```

## 📊 Analysis and Visualization

### CorrelationAnalyzer

Tools for analyzing entanglement patterns and correlations.

```python
class CorrelationAnalyzer:
    """
    Analyzer for quantum correlations and entanglement patterns.
    """
```

#### Methods

##### `compute_mutual_information(embeddings)`

```python
def compute_mutual_information(
    self,
    embeddings: torch.Tensor
) -> torch.Tensor:
    """
    Compute mutual information between token embeddings.
    
    Args:
        embeddings: Input embeddings
        
    Returns:
        Mutual information matrix
    """
```

##### `analyze_entanglement_decay(correlations, distances)`

```python
def analyze_entanglement_decay(
    self,
    correlations: torch.Tensor,
    distances: torch.Tensor
) -> Dict[str, float]:
    """
    Analyze how entanglement correlations decay with distance.
    
    Args:
        correlations: Correlation matrix
        distances: Distance matrix between tokens
        
    Returns:
        Dictionary with decay analysis results
    """
```

### EntanglementVisualizer

Visualization tools for entanglement patterns and quantum states.

```python
class EntanglementVisualizer:
    """
    Visualization tools for quantum entanglement patterns.
    """
```

#### Methods

##### `plot_correlation_heatmap(correlations, token_labels=None)`

```python
def plot_correlation_heatmap(
    self,
    correlations: torch.Tensor,
    token_labels: Optional[List[str]] = None,
    title: str = "Entanglement Correlations",
    **kwargs
) -> plt.Figure:
    """
    Create heatmap visualization of correlation matrix.
    
    Args:
        correlations: Correlation matrix to visualize
        token_labels: Optional labels for tokens
        title: Plot title
        
    Returns:
        Matplotlib figure object
    """
```

##### `create_entanglement_network(correlations, threshold=0.3)`

```python
def create_entanglement_network(
    self,
    correlations: torch.Tensor,
    token_labels: Optional[List[str]] = None,
    threshold: float = 0.3,
    layout: str = 'spring'
) -> plt.Figure:
    """
    Create network graph of entanglement connections.
    
    Args:
        correlations: Correlation matrix
        token_labels: Optional token labels
        threshold: Minimum correlation to show edge
        layout: Network layout algorithm
        
    Returns:
        Network graph figure
    """
```

##### `plot_quantum_evolution(evolution_data)`

```python
def plot_quantum_evolution(
    self,
    evolution_data: Dict[str, torch.Tensor],
    metrics: List[str] = ['entropy', 'coherence']
) -> plt.Figure:
    """
    Visualize quantum state evolution over time.
    
    Args:
        evolution_data: Evolution trajectory data
        metrics: Metrics to plot
        
    Returns:
        Evolution plot figure
    """
```

## 🎛 Configuration Classes

### EntanglementConfig

Configuration class for entanglement parameters.

```python
@dataclass
class EntanglementConfig:
    """Configuration for entanglement parameters."""
    
    entanglement_depth: int = 2
    correlation_strength: float = 0.7
    decoherence_rate: float = 0.1
    evolution_steps: int = 5
    measurement_basis: str = 'computational'
    enable_error_correction: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EntanglementConfig':
        """Create config from dictionary."""
```

## 🔧 Utility Functions

### Quantum State Utilities

```python
def create_bell_state(device: torch.device = None) -> torch.Tensor:
    """Create a Bell entangled state."""

def create_ghz_state(n_qubits: int, device: torch.device = None) -> torch.Tensor:
    """Create a GHZ entangled state for n qubits."""

def compute_entanglement_entropy(state: torch.Tensor) -> float:
    """Compute von Neumann entropy of quantum state."""

def apply_decoherence(
    state: torch.Tensor,
    decoherence_rate: float
) -> torch.Tensor:
    """Apply decoherence effects to quantum state."""
```

### Mathematical Utilities

```python
def quantum_fidelity(state1: torch.Tensor, state2: torch.Tensor) -> float:
    """Compute quantum fidelity between two states."""

def trace_distance(state1: torch.Tensor, state2: torch.Tensor) -> float:
    """Compute trace distance between quantum states."""

def compute_concurrence(state: torch.Tensor) -> float:
    """Compute concurrence measure of entanglement."""
```

## 📝 Type Definitions

```python
from typing import TypeVar, Union, Optional, Dict, List, Tuple, Any

# Tensor types
TensorType = torch.Tensor
DeviceType = Union[str, torch.device]

# Quantum state types
QuantumState = torch.Tensor
CorrelationMatrix = torch.Tensor

# Configuration types
ConfigDict = Dict[str, Any]
ParameterDict = Dict[str, Union[int, float, str, bool]]

# Return types
EmbeddingOutput = Tuple[torch.Tensor, Optional[torch.Tensor]]
QuantumMeasurement = Dict[str, torch.Tensor]
StatisticsOutput = Dict[str, float]
```

## 🚨 Exceptions

### Custom Exception Classes

```python
class QuantumProcessingError(Exception):
    """Raised when quantum processing operations fail."""
    pass

class EntanglementError(Exception):
    """Raised when entanglement operations encounter issues."""
    pass

class DimensionMismatchError(Exception):
    """Raised when tensor dimensions don't match expected values."""
    pass

class ConfigurationError(Exception):
    """Raised when configuration parameters are invalid."""
    pass
```

## 📊 Performance Considerations

### Memory Usage

- **EntangledEmbedding**: O(vocab_size × embedding_dim + seq_len²) for correlations
- **QuantumContextualizer**: O(seq_len × hidden_dim × evolution_steps)
- **EntangledAttention**: O(seq_len² × num_heads) additional memory

### Computational Complexity

- **Entanglement computation**: O(seq_len² × embedding_dim)
- **Quantum evolution**: O(evolution_steps × seq_len × hidden_dim)
- **Correlation analysis**: O(seq_len² × embedding_dim)

### Optimization Tips

```python
# Use gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Disable correlations when not needed
embeddings, _ = embedder(input_ids, return_correlations=False)

# Use lower precision for larger models
with torch.cuda.amp.autocast():
    outputs = model(inputs)
```

This API reference provides comprehensive documentation for all components of the Entanglement-Enhanced NLP framework. For practical usage examples, see the [Usage Guide](usage.md).
