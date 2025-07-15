"""
Core entangled embedding implementation.

This module implements quantum-inspired entangled embeddings that model
non-local correlations between semantically related tokens using shared
vector states and quantum entanglement principles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import numpy as np
import math

# Import licensing
from ..licensing import validate_class_license, requires_license


class EntangledEmbedding(nn.Module):
    """
    Quantum-inspired entangled embedding layer.
    
    This layer creates embeddings where semantically related tokens share
    entangled quantum states, enabling non-local correlations that capture
    deeper semantic relationships beyond traditional word embeddings.
    
    Mathematical Foundation:
    - Entangled State: |ψ⟩ = α|00⟩ + β|11⟩ + γ|01⟩ + δ|10⟩
    - Correlation Measure: C(i,j) = ⟨ψᵢ|ψⱼ⟩ · exp(-λ·d(i,j))
    - Non-local Attention: A(i,j) = softmax(Q(i)·K(j)ᵀ + C(i,j))
    
    LICENSE REQUIRED: This class requires a valid license to operate.
    Contact bajpaikrishna715@gmail.com for licensing information.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of embedding vectors
        entanglement_depth: Number of entanglement layers (default: 3)
        correlation_strength: Strength of quantum correlations (0-1, default: 0.8)
        decoherence_rate: Rate of quantum decoherence (default: 0.1)
        max_position_embeddings: Maximum sequence length for positional encoding
        dropout: Dropout probability for regularization
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        entanglement_depth: int = 3,
        correlation_strength: float = 0.8,
        decoherence_rate: float = 0.1,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
    ):
        # Validate license before allowing class instantiation
        validate_class_license(["basic_embedding"])
        
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.entanglement_depth = entanglement_depth
        self.correlation_strength = correlation_strength
        self.decoherence_rate = decoherence_rate
        
        # Traditional embedding layer as base
        self.base_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Quantum state parameters
        # Each token has amplitude and phase components for quantum superposition
        self.quantum_amplitudes = nn.Parameter(
            torch.randn(vocab_size, embedding_dim, 2)  # Real and imaginary parts
        )
        
        # Entanglement correlation matrix
        self.entanglement_matrix = nn.Parameter(
            torch.eye(vocab_size) * correlation_strength
        )
        
        # Position-dependent quantum evolution operators
        self.evolution_operators = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) 
            for _ in range(entanglement_depth)
        ])
        
        # Decoherence modeling layers
        self.decoherence_gates = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim)
            for _ in range(entanglement_depth)
        ])
        
        # Positional encoding for quantum state evolution
        self.positional_encoding = self._create_positional_encoding(
            max_position_embeddings, embedding_dim
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
        self._initialize_weights()
    
    def _create_positional_encoding(
        self, max_len: int, d_model: int
    ) -> torch.Tensor:
        """
        Create sinusoidal positional encoding with quantum phase modulation.
        
        The encoding includes quantum phase terms that evolve with position,
        mimicking quantum state evolution over time/space.
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        # Traditional sinusoidal encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add quantum phase modulation
        quantum_phase = torch.sin(position * math.pi / max_len)
        pe = pe * (1 + 0.1 * quantum_phase)
        
        return pe.unsqueeze(0)
    
    def _initialize_weights(self) -> None:
        """Initialize weights with quantum-inspired initialization."""
        # Initialize base embeddings with Xavier uniform
        nn.init.xavier_uniform_(self.base_embedding.weight)
        
        # Initialize quantum amplitudes with complex normal distribution
        nn.init.normal_(self.quantum_amplitudes, mean=0, std=0.02)
        
        # Initialize entanglement matrix with small random correlations
        with torch.no_grad():
            self.entanglement_matrix.fill_diagonal_(self.correlation_strength)
            # Add small random off-diagonal correlations
            noise = torch.randn_like(self.entanglement_matrix) * 0.01
            self.entanglement_matrix.data += (noise + noise.T) / 2
    
    def _compute_quantum_correlations(
        self, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute quantum entanglement correlations between tokens.
        
        This implements the quantum correlation function:
        C(i,j) = ⟨ψᵢ|ψⱼ⟩ · exp(-λ·|i-j|)
        
        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_len)
            
        Returns:
            Correlation matrix of shape (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len = input_ids.shape
        
        # Get entanglement correlations for input tokens
        token_correlations = self.entanglement_matrix[input_ids]  # (B, L, V)
        token_correlations = torch.gather(
            token_correlations, dim=2, 
            index=input_ids.unsqueeze(-1).expand(-1, -1, seq_len)
        )  # (B, L, L)
        
        # Apply distance-based decay
        positions = torch.arange(seq_len, device=input_ids.device)
        distance_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        decay_factor = torch.exp(-self.decoherence_rate * distance_matrix.float())
        
        correlations = token_correlations * decay_factor.unsqueeze(0)
        
        return correlations
    
    def _apply_quantum_evolution(
        self, embeddings: torch.Tensor, step: int
    ) -> torch.Tensor:
        """
        Apply quantum state evolution operator.
        
        This implements unitary evolution: |ψ(t+1)⟩ = U(θ)|ψ(t)⟩
        where U(θ) = exp(-iH·θ) is approximated by the evolution operators.
        
        Args:
            embeddings: Input embeddings tensor
            step: Evolution step index
            
        Returns:
            Evolved embeddings tensor
        """
        # Apply evolution operator (unitary transformation)
        evolved = self.evolution_operators[step](embeddings)
        
        # Apply decoherence (non-unitary effects)
        decoherence_factor = torch.sigmoid(self.decoherence_gates[step](embeddings))
        evolved = evolved * decoherence_factor + embeddings * (1 - decoherence_factor)
        
        return evolved
    
    def _create_superposition_states(
        self, input_ids: torch.Tensor, base_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Create quantum superposition states from base embeddings.
        
        Combines base embeddings with quantum amplitude components to create
        superposition states that can represent multiple semantic interpretations.
        
        Args:
            input_ids: Token IDs tensor
            base_embeddings: Base embedding vectors
            
        Returns:
            Superposition state embeddings
        """
        batch_size, seq_len = input_ids.shape
        
        # Get quantum amplitudes for tokens
        quantum_amps = self.quantum_amplitudes[input_ids]  # (B, L, D, 2)
        
        # Create complex embeddings (real + i*imaginary)
        real_part = base_embeddings + quantum_amps[..., 0]
        imag_part = quantum_amps[..., 1]
        
        # Compute magnitude for superposition weighting
        magnitude = torch.sqrt(real_part**2 + imag_part**2 + 1e-8)
        
        # Normalize and combine with base embeddings
        superposition = base_embeddings + 0.1 * magnitude * torch.tanh(real_part)
        
        return superposition
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        return_correlations: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through entangled embedding layer.
        
        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_len)
            position_ids: Position IDs tensor (optional)
            return_correlations: Whether to return correlation matrix
            
        Returns:
            Tuple of (entangled_embeddings, correlations)
            - entangled_embeddings: Enhanced embeddings with quantum effects
            - correlations: Entanglement correlation matrix (if requested)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get base embeddings
        base_embeddings = self.base_embedding(input_ids)  # (B, L, D)
        
        # Add positional encoding
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        pos_encoding = self.positional_encoding[:, :seq_len, :].to(device)
        embeddings = base_embeddings + pos_encoding
        
        # Create quantum superposition states
        embeddings = self._create_superposition_states(input_ids, embeddings)
        
        # Compute quantum correlations
        correlations = self._compute_quantum_correlations(input_ids)
        
        # Apply entanglement-aware attention
        for i in range(seq_len):
            for j in range(seq_len):
                if i != j:
                    correlation_strength = correlations[:, i, j].unsqueeze(-1)
                    embeddings[:, i] += correlation_strength * embeddings[:, j] * 0.1
        
        # Apply quantum evolution steps
        for step in range(self.entanglement_depth):
            embeddings = self._apply_quantum_evolution(embeddings, step)
            embeddings = self.layer_norm(embeddings)
        
        # Apply dropout for regularization
        embeddings = self.dropout(embeddings)
        
        if return_correlations:
            return embeddings, correlations
        return embeddings, None
    
    def get_entanglement_statistics(self, input_ids: torch.Tensor) -> Dict[str, float]:
        """
        Compute entanglement statistics for analysis.
        
        Args:
            input_ids: Token IDs tensor
            
        Returns:
            Dictionary with entanglement metrics
        """
        with torch.no_grad():
            correlations = self._compute_quantum_correlations(input_ids)
            
            # Compute various entanglement measures
            avg_correlation = correlations.mean().item()
            max_correlation = correlations.max().item()
            entanglement_entropy = -(correlations * torch.log(correlations + 1e-8)).sum(-1).mean().item()
            
            # Quantum coherence measure
            off_diagonal = correlations - torch.diag_embed(torch.diagonal(correlations, dim1=-2, dim2=-1))
            coherence = torch.norm(off_diagonal, dim=(-2, -1)).mean().item()
            
            return {
                "average_correlation": avg_correlation,
                "maximum_correlation": max_correlation,
                "entanglement_entropy": entanglement_entropy,
                "quantum_coherence": coherence,
                "decoherence_rate": self.decoherence_rate,
                "correlation_strength": self.correlation_strength,
            }
