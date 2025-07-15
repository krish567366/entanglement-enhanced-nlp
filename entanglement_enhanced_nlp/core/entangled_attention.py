"""
Entangled Attention mechanism implementation.

This module implements quantum-inspired attention mechanisms that incorporate
entanglement correlations and non-local interactions for enhanced semantic
understanding in transformer architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
import numpy as np

# Import licensing
from ..licensing import validate_class_license, requires_license


class EntangledAttention(nn.Module):
    """
    Quantum-inspired entangled attention mechanism.
    
    This attention layer incorporates quantum entanglement principles to model
    non-local correlations between tokens, enabling more sophisticated semantic
    relationship modeling beyond traditional attention mechanisms.
    
    Mathematical Foundation:
    - Entangled Attention: A(i,j) = softmax(Q(i)·K(j)ᵀ/√d + E(i,j))
    - Entanglement Term: E(i,j) = ⟨ψᵢ|ψⱼ⟩ · exp(-λ·|i-j|)
    - Non-local Correlation: C(i,j) = Σₖ A(i,k)·A(j,k)·entangle(k)
    - Quantum Superposition: |out⟩ = Σᵢ αᵢ|vᵢ⟩ with quantum amplitudes
    
    LICENSE REQUIRED: This class requires a valid license to operate.
    Contact bajpaikrishna715@gmail.com for licensing information.
    
    Args:
        hidden_dim: Dimension of hidden states
        num_heads: Number of attention heads
        entanglement_strength: Strength of quantum entanglement effects (0-1)
        correlation_decay: Decay rate for distance-based correlations
        superposition_weight: Weight for quantum superposition effects
        dropout: Dropout probability for attention weights
        bias: Whether to use bias in linear layers
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        entanglement_strength: float = 0.5,
        correlation_decay: float = 0.1,
        superposition_weight: float = 0.3,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        # Validate license before allowing class instantiation
        validate_class_license(["entangled_attention"])
        
        super().__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.entanglement_strength = entanglement_strength
        self.correlation_decay = correlation_decay
        self.superposition_weight = superposition_weight
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Standard attention projections
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        # Quantum entanglement projections
        self.entangle_query = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.entangle_key = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.entangle_value = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        # Quantum amplitude and phase generators
        self.amplitude_generator = nn.Linear(hidden_dim, num_heads)
        self.phase_generator = nn.Linear(hidden_dim, num_heads)
        
        # Non-local correlation modeling
        self.correlation_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Superposition mixing layers
        self.superposition_mixer = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim) for _ in range(num_heads)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights with quantum-inspired initialization."""
        # Standard Xavier initialization for most layers
        for module in [self.query_proj, self.key_proj, self.value_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        # Special initialization for entanglement layers (smaller scale)
        for module in [self.entangle_query, self.entangle_key, self.entangle_value]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        # Initialize amplitude and phase generators
        nn.init.normal_(self.amplitude_generator.weight, mean=0, std=0.02)
        nn.init.normal_(self.phase_generator.weight, mean=0, std=0.02)
    
    def _compute_entanglement_correlations(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor,
        position_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute quantum entanglement correlations between query and key tokens.
        
        Args:
            query: Query tensor (batch_size, seq_len, hidden_dim)
            key: Key tensor (batch_size, seq_len, hidden_dim)
            position_bias: Optional positional bias tensor
            
        Returns:
            Entanglement correlation matrix (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Project to entanglement space
        entangle_q = self.entangle_query(query)  # (B, Lq, H)
        entangle_k = self.entangle_key(key)      # (B, Lk, H)
        
        # Reshape for multi-head attention
        entangle_q = entangle_q.view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        entangle_k = entangle_k.view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        entangle_q = entangle_q.transpose(1, 2)  # (B, NH, Lq, HD)
        entangle_k = entangle_k.transpose(1, 2)  # (B, NH, Lk, HD)
        
        # Compute entanglement correlations using quantum inner product
        # ⟨ψᵢ|ψⱼ⟩ = Σₖ qᵢₖ* · kⱼₖ
        entangle_scores = torch.matmul(entangle_q, entangle_k.transpose(-2, -1))
        
        # Apply quantum correlation normalization
        entangle_scores = entangle_scores / math.sqrt(self.head_dim)
        
        # Add distance-based decay for non-local correlations
        if position_bias is None:
            positions_q = torch.arange(seq_len_q, device=query.device)
            positions_k = torch.arange(seq_len_k, device=key.device)
            distance_matrix = torch.abs(positions_q.unsqueeze(1) - positions_k.unsqueeze(0))
            distance_decay = torch.exp(-self.correlation_decay * distance_matrix.float())
            position_bias = distance_decay.unsqueeze(0).unsqueeze(0)  # (1, 1, Lq, Lk)
        
        entangle_scores = entangle_scores * position_bias
        
        return entangle_scores
    
    def _compute_quantum_amplitudes(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute quantum amplitudes and phases for superposition states.
        
        Args:
            hidden_states: Input hidden states
            
        Returns:
            Tuple of (amplitudes, phases) for quantum superposition
        """
        # Generate quantum amplitudes (probabilities)
        amplitudes = torch.sigmoid(self.amplitude_generator(hidden_states))  # (B, L, NH)
        
        # Generate quantum phases
        phases = torch.tanh(self.phase_generator(hidden_states)) * math.pi  # (B, L, NH)
        
        return amplitudes, phases
    
    def _apply_superposition_mixing(
        self, 
        values: torch.Tensor, 
        amplitudes: torch.Tensor, 
        phases: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply quantum superposition mixing to value vectors.
        
        Args:
            values: Value vectors (batch_size, num_heads, seq_len, head_dim)
            amplitudes: Quantum amplitudes (batch_size, seq_len, num_heads)
            phases: Quantum phases (batch_size, seq_len, num_heads)
            
        Returns:
            Superposition-mixed values
        """
        batch_size, num_heads, seq_len, head_dim = values.shape
        
        # Transpose amplitudes and phases to match values shape
        amplitudes = amplitudes.transpose(1, 2).unsqueeze(-1)  # (B, NH, L, 1)
        phases = phases.transpose(1, 2).unsqueeze(-1)          # (B, NH, L, 1)
        
        # Create complex representation
        real_part = amplitudes * torch.cos(phases)
        imag_part = amplitudes * torch.sin(phases)
        
        # Apply superposition mixing to each head
        mixed_values = values.clone()
        for head_idx in range(num_heads):
            head_values = values[:, head_idx]  # (B, L, HD)
            head_real = real_part[:, head_idx]  # (B, L, 1)
            head_imag = imag_part[:, head_idx]  # (B, L, 1)
            
            # Apply superposition transformation
            superposed = self.superposition_mixer[head_idx](head_values)
            mixed_values[:, head_idx] = (
                head_values + 
                self.superposition_weight * head_real * superposed +
                self.superposition_weight * head_imag * torch.roll(superposed, 1, dim=-1)
            )
        
        return mixed_values
    
    def _compute_non_local_correlations(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute non-local correlations for enhanced context modeling.
        
        Args:
            query: Query tensor
            key: Key tensor  
            attention_weights: Standard attention weights
            
        Returns:
            Non-local correlation enhancement
        """
        batch_size, seq_len_q, hidden_dim = query.shape
        seq_len_k = key.shape[1]
        
        # Create pairwise feature combinations
        query_expanded = query.unsqueeze(2).expand(-1, -1, seq_len_k, -1)  # (B, Lq, Lk, H)
        key_expanded = key.unsqueeze(1).expand(-1, seq_len_q, -1, -1)      # (B, Lq, Lk, H)
        
        # Concatenate for correlation modeling
        pairwise_features = torch.cat([query_expanded, key_expanded], dim=-1)  # (B, Lq, Lk, 2H)
        
        # Compute non-local correlation scores
        correlation_scores = self.correlation_encoder(pairwise_features).squeeze(-1)  # (B, Lq, Lk)
        
        # Normalize correlation scores
        correlation_scores = torch.tanh(correlation_scores)
        
        # Weight by existing attention and sum over heads
        weighted_correlations = torch.mean(attention_weights, dim=1) * correlation_scores  # (B, Lq, Lk)
        
        return weighted_correlations.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
        return_correlations: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through entangled attention layer.
        
        Args:
            hidden_states: Input hidden states (batch_size, seq_len, hidden_dim)
            attention_mask: Attention mask for padding tokens
            position_bias: Optional positional bias tensor
            return_attention_weights: Whether to return attention weights
            return_correlations: Whether to return entanglement correlations
            
        Returns:
            Tuple of (output, attention_weights, correlations)
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Standard attention projections
        query = self.query_proj(hidden_states)  # (B, L, H)
        key = self.key_proj(hidden_states)      # (B, L, H)
        value = self.value_proj(hidden_states)  # (B, L, H)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute standard attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Compute entanglement correlations
        entangle_scores = self._compute_entanglement_correlations(
            hidden_states, hidden_states, position_bias
        )
        
        # Combine standard attention with entanglement
        combined_scores = attention_scores + self.entanglement_strength * entangle_scores
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for multi-head attention
            mask_expanded = attention_mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
            combined_scores = combined_scores.masked_fill(~mask_expanded, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(combined_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Compute non-local correlations
        non_local_corr = self._compute_non_local_correlations(
            hidden_states, hidden_states, attention_weights
        )
        
        # Enhance attention weights with non-local correlations
        enhanced_weights = attention_weights + 0.1 * non_local_corr
        enhanced_weights = F.softmax(enhanced_weights, dim=-1)
        
        # Generate quantum amplitudes and phases
        amplitudes, phases = self._compute_quantum_amplitudes(hidden_states)
        
        # Apply superposition mixing to values
        value_superposed = self._apply_superposition_mixing(value, amplitudes, phases)
        
        # Apply attention to superposed values
        attended_output = torch.matmul(enhanced_weights, value_superposed)
        
        # Reshape and project output
        attended_output = attended_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_dim
        )
        output = self.output_proj(attended_output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(output + hidden_states)
        
        # Prepare return values
        return_weights = enhanced_weights if return_attention_weights else None
        return_corr = entangle_scores if return_correlations else None
        
        return output, return_weights, return_corr
    
    def get_entanglement_statistics(
        self, hidden_states: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute entanglement statistics for analysis.
        
        Args:
            hidden_states: Input hidden states
            
        Returns:
            Dictionary with entanglement metrics
        """
        with torch.no_grad():
            # Compute entanglement correlations
            entangle_scores = self._compute_entanglement_correlations(
                hidden_states, hidden_states
            )
            
            # Compute statistics
            avg_entanglement = entangle_scores.mean().item()
            max_entanglement = entangle_scores.max().item()
            min_entanglement = entangle_scores.min().item()
            
            # Compute quantum amplitudes and phases
            amplitudes, phases = self._compute_quantum_amplitudes(hidden_states)
            avg_amplitude = amplitudes.mean().item()
            phase_variance = phases.var().item()
            
            # Entanglement entropy
            entangle_probs = F.softmax(entangle_scores, dim=-1)
            entropy = -(entangle_probs * torch.log(entangle_probs + 1e-8)).sum(-1).mean().item()
            
            return {
                "average_entanglement": avg_entanglement,
                "maximum_entanglement": max_entanglement,
                "minimum_entanglement": min_entanglement,
                "entanglement_entropy": entropy,
                "average_amplitude": avg_amplitude,
                "phase_variance": phase_variance,
                "entanglement_strength": self.entanglement_strength,
                "correlation_decay": self.correlation_decay,
                "superposition_weight": self.superposition_weight,
            }
