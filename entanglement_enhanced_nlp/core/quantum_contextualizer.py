"""
Quantum Contextualizer implementation.

This module implements a quantum-inspired contextualizer that enhances token
embeddings using quantum state evolution principles, enabling more sophisticated
context modeling through quantum superposition and entanglement effects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np
import math


class QuantumContextualizer(nn.Module):
    """
    Quantum-inspired contextualizer for enhancing token embeddings.
    
    This layer applies quantum state evolution principles to enhance embeddings
    with contextual information, using quantum superposition to model multiple
    possible interpretations and quantum entanglement for long-range dependencies.
    
    Mathematical Foundation:
    - Quantum State: |ψ⟩ = Σᵢ αᵢ|φᵢ⟩ (superposition of context states)
    - Evolution Operator: U(t) = exp(-iH·t) (Hamiltonian evolution)
    - Measurement: ⟨O⟩ = ⟨ψ|O|ψ⟩ (expectation value of observable)
    - Decoherence: ρ(t) = Σₖ EₖρE†ₖ (Kraus operator formalism)
    
    Args:
        hidden_dim: Dimension of hidden states
        num_qubits: Number of quantum bits for simulation (default: 8)
        num_layers: Number of quantum evolution layers (default: 3)
        decoherence_rate: Rate of quantum decoherence (default: 0.1)
        gate_types: Types of quantum gates to use (default: ['rx', 'ry', 'rz', 'cnot'])
        measurement_basis: Measurement basis for quantum states (default: 'computational')
        dropout: Dropout probability for regularization
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_qubits: int = 8,
        num_layers: int = 3,
        decoherence_rate: float = 0.1,
        gate_types: List[str] = None,
        measurement_basis: str = 'computational',
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.decoherence_rate = decoherence_rate
        self.measurement_basis = measurement_basis
        
        if gate_types is None:
            gate_types = ['rx', 'ry', 'rz', 'cnot']
        self.gate_types = gate_types
        
        # Quantum state initialization layers
        self.state_encoder = nn.Linear(hidden_dim, num_qubits * 2)  # Amplitude + Phase
        self.state_decoder = nn.Linear(num_qubits * 2, hidden_dim)
        
        # Quantum gate parameter generators
        self.gate_generators = nn.ModuleDict({
            gate_type: nn.Linear(hidden_dim, num_qubits if gate_type != 'cnot' else num_qubits//2)
            for gate_type in gate_types
        })
        
        # Hamiltonian generators for time evolution
        self.hamiltonian_generators = nn.ModuleList([
            nn.Linear(hidden_dim, num_qubits * num_qubits)
            for _ in range(num_layers)
        ])
        
        # Contextual attention mechanisms
        self.context_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Quantum measurement operators
        self.measurement_operators = nn.ModuleList([
            nn.Linear(num_qubits, hidden_dim // num_layers)
            for _ in range(num_layers)
        ])
        
        # Layer normalization and dropout
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers + 1)
        ])
        self.dropout = nn.Dropout(dropout)
        
        # Initialize quantum circuit parameters
        self._initialize_quantum_parameters()
    
    def _initialize_quantum_parameters(self) -> None:
        """Initialize quantum circuit parameters."""
        # Initialize gate generators with small random values
        for gate_gen in self.gate_generators.values():
            nn.init.normal_(gate_gen.weight, mean=0, std=0.02)
            nn.init.zeros_(gate_gen.bias)
        
        # Initialize Hamiltonian generators
        for ham_gen in self.hamiltonian_generators:
            nn.init.orthogonal_(ham_gen.weight, gain=0.1)
            nn.init.zeros_(ham_gen.bias)
    
    def _encode_to_quantum_state(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Encode classical embeddings to quantum state representation.
        
        Args:
            embeddings: Input embeddings of shape (batch_size, seq_len, hidden_dim)
            
        Returns:
            Quantum state tensor of shape (batch_size, seq_len, num_qubits, 2)
        """
        batch_size, seq_len, _ = embeddings.shape
        
        # Encode to amplitude and phase components
        state_params = self.state_encoder(embeddings)  # (B, L, 2*Q)
        state_params = state_params.view(batch_size, seq_len, self.num_qubits, 2)
        
        # Normalize to valid quantum state amplitudes
        amplitudes = torch.tanh(state_params[..., 0])  # Real part
        phases = torch.sigmoid(state_params[..., 1]) * 2 * math.pi  # Phase [0, 2π]
        
        # Create complex amplitudes
        real_part = amplitudes * torch.cos(phases)
        imag_part = amplitudes * torch.sin(phases)
        
        quantum_state = torch.stack([real_part, imag_part], dim=-1)
        
        # Normalize quantum state (ensure |ψ|² = 1)
        norm = torch.sqrt(torch.sum(quantum_state**2, dim=(-2, -1), keepdim=True) + 1e-8)
        quantum_state = quantum_state / norm
        
        return quantum_state
    
    def _apply_quantum_gate(
        self, 
        quantum_state: torch.Tensor, 
        gate_type: str, 
        parameters: torch.Tensor,
        qubit_indices: Optional[Tuple[int, ...]] = None
    ) -> torch.Tensor:
        """
        Apply quantum gate to quantum state.
        
        Args:
            quantum_state: Quantum state tensor
            gate_type: Type of quantum gate ('rx', 'ry', 'rz', 'cnot')
            parameters: Gate parameters
            qubit_indices: Indices of qubits to apply gate to
            
        Returns:
            Updated quantum state
        """
        if gate_type in ['rx', 'ry', 'rz']:
            return self._apply_rotation_gate(quantum_state, gate_type, parameters)
        elif gate_type == 'cnot':
            return self._apply_cnot_gate(quantum_state, parameters)
        else:
            raise ValueError(f"Unsupported gate type: {gate_type}")
    
    def _apply_rotation_gate(
        self, quantum_state: torch.Tensor, gate_type: str, angles: torch.Tensor
    ) -> torch.Tensor:
        """Apply single-qubit rotation gates (RX, RY, RZ)."""
        batch_size, seq_len, num_qubits, _ = quantum_state.shape
        
        # Extract real and imaginary parts
        real = quantum_state[..., 0]  # (B, L, Q)
        imag = quantum_state[..., 1]  # (B, L, Q)
        
        if gate_type == 'rx':
            # RX(θ) = cos(θ/2)I - i*sin(θ/2)σx
            cos_half = torch.cos(angles / 2)
            sin_half = torch.sin(angles / 2)
            new_real = cos_half * real
            new_imag = cos_half * imag - sin_half * real
        
        elif gate_type == 'ry':
            # RY(θ) = cos(θ/2)I - i*sin(θ/2)σy
            cos_half = torch.cos(angles / 2)
            sin_half = torch.sin(angles / 2)
            new_real = cos_half * real - sin_half * imag
            new_imag = cos_half * imag + sin_half * real
        
        elif gate_type == 'rz':
            # RZ(θ) = exp(-iθ/2)
            cos_half = torch.cos(angles / 2)
            sin_half = torch.sin(angles / 2)
            new_real = cos_half * real + sin_half * imag
            new_imag = cos_half * imag - sin_half * real
        
        return torch.stack([new_real, new_imag], dim=-1)
    
    def _apply_cnot_gate(
        self, quantum_state: torch.Tensor, control_params: torch.Tensor
    ) -> torch.Tensor:
        """Apply CNOT gates between qubit pairs."""
        batch_size, seq_len, num_qubits, _ = quantum_state.shape
        
        # Create CNOT connections based on parameters
        cnot_strength = torch.sigmoid(control_params)  # (B, L, Q//2)
        
        new_state = quantum_state.clone()
        
        # Apply CNOT operations between adjacent qubit pairs
        for i in range(0, num_qubits - 1, 2):
            if i // 2 < cnot_strength.shape[-1]:
                strength = cnot_strength[..., i // 2].unsqueeze(-1)  # (B, L, 1)
                
                # Simplified CNOT operation (entanglement between qubits)
                control_real = new_state[..., i, 0]  # (B, L)
                control_imag = new_state[..., i, 1]  # (B, L)
                target_real = new_state[..., i + 1, 0]  # (B, L)
                target_imag = new_state[..., i + 1, 1]  # (B, L)
                
                # Create entanglement
                entangled_real = strength * (control_real * target_real - control_imag * target_imag)
                entangled_imag = strength * (control_real * target_imag + control_imag * target_real)
                
                new_state[..., i + 1, 0] = target_real + 0.1 * entangled_real
                new_state[..., i + 1, 1] = target_imag + 0.1 * entangled_imag
        
        return new_state
    
    def _apply_hamiltonian_evolution(
        self, quantum_state: torch.Tensor, hamiltonian_params: torch.Tensor, dt: float = 0.1
    ) -> torch.Tensor:
        """
        Apply Hamiltonian evolution: |ψ(t+dt)⟩ = exp(-iH·dt)|ψ(t)⟩
        
        Args:
            quantum_state: Current quantum state
            hamiltonian_params: Parameters defining the Hamiltonian
            dt: Time step for evolution
            
        Returns:
            Evolved quantum state
        """
        batch_size, seq_len, num_qubits, _ = quantum_state.shape
        
        # Reshape Hamiltonian parameters to matrix form
        H_params = hamiltonian_params.view(batch_size, seq_len, num_qubits, num_qubits)
        
        # Make Hamiltonian Hermitian
        H = (H_params + H_params.transpose(-2, -1)) / 2
        
        # Apply simplified evolution (first-order approximation)
        real = quantum_state[..., 0]  # (B, L, Q)
        imag = quantum_state[..., 1]  # (B, L, Q)
        
        # H|ψ⟩ for real and imaginary parts
        H_real = torch.matmul(H, real.unsqueeze(-1)).squeeze(-1)
        H_imag = torch.matmul(H, imag.unsqueeze(-1)).squeeze(-1)
        
        # exp(-iH·dt)|ψ⟩ ≈ (I - iH·dt)|ψ⟩
        new_real = real + dt * H_imag  # Real part of -iH|ψ⟩
        new_imag = imag - dt * H_real  # Imaginary part of -iH|ψ⟩
        
        return torch.stack([new_real, new_imag], dim=-1)
    
    def _apply_decoherence(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """
        Apply quantum decoherence effects.
        
        Decoherence gradually destroys quantum superposition and entanglement,
        modeling the transition from quantum to classical behavior.
        """
        if self.decoherence_rate <= 0:
            return quantum_state
        
        # Add noise to quantum state (decoherence)
        noise_real = torch.randn_like(quantum_state[..., 0]) * self.decoherence_rate
        noise_imag = torch.randn_like(quantum_state[..., 1]) * self.decoherence_rate
        
        decoherent_state = quantum_state.clone()
        decoherent_state[..., 0] += noise_real
        decoherent_state[..., 1] += noise_imag
        
        # Renormalize
        norm = torch.sqrt(torch.sum(decoherent_state**2, dim=(-2, -1), keepdim=True) + 1e-8)
        decoherent_state = decoherent_state / norm
        
        return decoherent_state
    
    def _measure_quantum_state(
        self, quantum_state: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """
        Perform quantum measurement and extract classical information.
        
        Args:
            quantum_state: Quantum state tensor
            layer_idx: Index of measurement layer
            
        Returns:
            Classical feature vector from quantum measurement
        """
        # Compute probability amplitudes |ψ|²
        probabilities = torch.sum(quantum_state**2, dim=-1)  # (B, L, Q)
        
        # Apply measurement operator
        measured_features = self.measurement_operators[layer_idx](probabilities)
        
        return measured_features
    
    def forward(
        self, 
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_quantum_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through quantum contextualizer.
        
        Args:
            embeddings: Input embeddings (batch_size, seq_len, hidden_dim)
            attention_mask: Attention mask for padding tokens
            return_quantum_states: Whether to return intermediate quantum states
            
        Returns:
            Tuple of (enhanced_embeddings, quantum_states)
        """
        batch_size, seq_len, hidden_dim = embeddings.shape
        
        # Apply initial layer normalization
        x = self.layer_norms[0](embeddings)
        
        # Encode to quantum state representation
        quantum_state = self._encode_to_quantum_state(x)
        quantum_states_history = [quantum_state] if return_quantum_states else None
        
        # Store measured features from each layer
        measured_features = []
        
        # Apply quantum evolution layers
        for layer_idx in range(self.num_layers):
            # Generate gate parameters from current embeddings
            gate_params = {}
            for gate_type in self.gate_types:
                gate_params[gate_type] = self.gate_generators[gate_type](x)
            
            # Apply quantum gates
            for gate_type in self.gate_types:
                quantum_state = self._apply_quantum_gate(
                    quantum_state, gate_type, gate_params[gate_type]
                )
            
            # Apply Hamiltonian evolution
            hamiltonian_params = self.hamiltonian_generators[layer_idx](x)
            quantum_state = self._apply_hamiltonian_evolution(quantum_state, hamiltonian_params)
            
            # Apply decoherence
            quantum_state = self._apply_decoherence(quantum_state)
            
            # Perform quantum measurement
            measured = self._measure_quantum_state(quantum_state, layer_idx)
            measured_features.append(measured)
            
            if return_quantum_states:
                quantum_states_history.append(quantum_state)
        
        # Combine measured features
        combined_features = torch.cat(measured_features, dim=-1)  # (B, L, hidden_dim)
        
        # Apply contextual attention
        attended_features, _ = self.context_attention(
            combined_features, combined_features, combined_features,
            key_padding_mask=~attention_mask if attention_mask is not None else None
        )
        
        # Residual connection and layer normalization
        enhanced_embeddings = self.layer_norms[-1](embeddings + attended_features)
        enhanced_embeddings = self.dropout(enhanced_embeddings)
        
        return enhanced_embeddings, quantum_states_history
    
    def get_quantum_statistics(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """
        Compute quantum statistics for analysis.
        
        Args:
            embeddings: Input embeddings tensor
            
        Returns:
            Dictionary with quantum metrics
        """
        with torch.no_grad():
            quantum_state = self._encode_to_quantum_state(embeddings)
            
            # Compute quantum metrics
            probabilities = torch.sum(quantum_state**2, dim=-1)
            
            # Von Neumann entropy (quantum entropy)
            entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum(-1).mean().item()
            
            # Quantum coherence (off-diagonal elements)
            real_part = quantum_state[..., 0]
            imag_part = quantum_state[..., 1]
            coherence = torch.sqrt(real_part**2 + imag_part**2).mean().item()
            
            # Entanglement measure (simplified)
            batch_size, seq_len, num_qubits, _ = quantum_state.shape
            entanglement = 0.0
            for i in range(num_qubits - 1):
                for j in range(i + 1, num_qubits):
                    corr = torch.corrcoef(torch.stack([
                        probabilities[..., i].flatten(),
                        probabilities[..., j].flatten()
                    ]))[0, 1]
                    entanglement += abs(corr.item()) if not torch.isnan(corr) else 0.0
            
            entanglement /= (num_qubits * (num_qubits - 1) / 2)
            
            return {
                "quantum_entropy": entropy,
                "quantum_coherence": coherence,
                "entanglement_measure": entanglement,
                "decoherence_rate": self.decoherence_rate,
                "num_qubits": self.num_qubits,
                "num_layers": self.num_layers,
            }
