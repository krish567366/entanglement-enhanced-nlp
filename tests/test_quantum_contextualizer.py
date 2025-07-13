"""
Tests for QuantumContextualizer module.
"""

import pytest
import torch
import numpy as np
from entanglement_enhanced_nlp.core.quantum_contextualizer import QuantumContextualizer


class TestQuantumContextualizer:
    """Test cases for QuantumContextualizer class."""
    
    @pytest.fixture
    def contextualizer(self):
        """Create a test QuantumContextualizer instance."""
        return QuantumContextualizer(
            hidden_dim=128,
            num_qubits=6,
            num_layers=3,
            decoherence_rate=0.1,
            dropout=0.1,
        )
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embedding tensor."""
        return torch.randn(2, 10, 128)  # (batch_size=2, seq_len=10, hidden_dim=128)
    
    def test_initialization(self, contextualizer):
        """Test proper initialization of QuantumContextualizer."""
        assert contextualizer.hidden_dim == 128
        assert contextualizer.num_qubits == 6
        assert contextualizer.num_layers == 3
        assert contextualizer.decoherence_rate == 0.1
        
        # Check component initialization
        assert hasattr(contextualizer, 'state_encoder')
        assert hasattr(contextualizer, 'state_decoder')
        assert hasattr(contextualizer, 'gate_generators')
        assert hasattr(contextualizer, 'hamiltonian_generators')
        assert len(contextualizer.hamiltonian_generators) == 3
    
    def test_forward_pass(self, contextualizer, sample_embeddings):
        """Test forward pass through QuantumContextualizer."""
        # Test without quantum states return
        enhanced_embeddings, quantum_states = contextualizer(sample_embeddings)
        
        assert enhanced_embeddings.shape == sample_embeddings.shape
        assert quantum_states is None
        
        # Test with quantum states return
        enhanced_embeddings, quantum_states = contextualizer(
            sample_embeddings, return_quantum_states=True
        )
        
        assert enhanced_embeddings.shape == sample_embeddings.shape
        assert quantum_states is not None
        assert len(quantum_states) == contextualizer.num_layers + 1  # Initial + after each layer
    
    def test_quantum_state_encoding(self, contextualizer, sample_embeddings):
        """Test encoding to quantum state representation."""
        quantum_state = contextualizer._encode_to_quantum_state(sample_embeddings)
        
        batch_size, seq_len, _ = sample_embeddings.shape
        expected_shape = (batch_size, seq_len, contextualizer.num_qubits, 2)
        assert quantum_state.shape == expected_shape
        
        # Check normalization (quantum states should be normalized)
        norms = torch.sum(quantum_state**2, dim=(-2, -1))
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    def test_quantum_gates(self, contextualizer):
        """Test quantum gate operations."""
        batch_size, seq_len, num_qubits = 2, 5, 6
        quantum_state = torch.randn(batch_size, seq_len, num_qubits, 2)
        quantum_state = quantum_state / torch.norm(quantum_state, dim=(-2, -1), keepdim=True)
        
        # Test rotation gates
        for gate_type in ['rx', 'ry', 'rz']:
            angles = torch.randn(batch_size, seq_len, num_qubits)
            rotated_state = contextualizer._apply_rotation_gate(quantum_state, gate_type, angles)
            assert rotated_state.shape == quantum_state.shape
            # State should be different after rotation
            assert not torch.allclose(rotated_state, quantum_state, atol=1e-6)
        
        # Test CNOT gate
        control_params = torch.randn(batch_size, seq_len, num_qubits // 2)
        cnot_state = contextualizer._apply_cnot_gate(quantum_state, control_params)
        assert cnot_state.shape == quantum_state.shape
    
    def test_hamiltonian_evolution(self, contextualizer):
        """Test Hamiltonian evolution operation."""
        batch_size, seq_len, num_qubits = 2, 5, 6
        quantum_state = torch.randn(batch_size, seq_len, num_qubits, 2)
        quantum_state = quantum_state / torch.norm(quantum_state, dim=(-2, -1), keepdim=True)
        
        hamiltonian_params = torch.randn(batch_size, seq_len, num_qubits * num_qubits)
        evolved_state = contextualizer._apply_hamiltonian_evolution(
            quantum_state, hamiltonian_params, dt=0.1
        )
        
        assert evolved_state.shape == quantum_state.shape
        # Evolution should change the state
        assert not torch.allclose(evolved_state, quantum_state, atol=1e-6)
    
    def test_decoherence(self, contextualizer):
        """Test quantum decoherence application."""
        batch_size, seq_len, num_qubits = 2, 5, 6
        quantum_state = torch.randn(batch_size, seq_len, num_qubits, 2)
        quantum_state = quantum_state / torch.norm(quantum_state, dim=(-2, -1), keepdim=True)
        
        # Test with decoherence
        decoherent_state = contextualizer._apply_decoherence(quantum_state)
        assert decoherent_state.shape == quantum_state.shape
        
        # Check normalization is preserved
        norms = torch.sum(decoherent_state**2, dim=(-2, -1))
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        
        # With zero decoherence rate, state should be unchanged
        contextualizer.decoherence_rate = 0.0
        unchanged_state = contextualizer._apply_decoherence(quantum_state)
        assert torch.allclose(unchanged_state, quantum_state, atol=1e-6)
    
    def test_quantum_measurement(self, contextualizer):
        """Test quantum measurement operation."""
        batch_size, seq_len, num_qubits = 2, 5, 6
        quantum_state = torch.randn(batch_size, seq_len, num_qubits, 2)
        quantum_state = quantum_state / torch.norm(quantum_state, dim=(-2, -1), keepdim=True)
        
        for layer_idx in range(contextualizer.num_layers):
            measured_features = contextualizer._measure_quantum_state(quantum_state, layer_idx)
            expected_dim = contextualizer.hidden_dim // contextualizer.num_layers
            assert measured_features.shape == (batch_size, seq_len, expected_dim)
    
    def test_quantum_statistics(self, contextualizer, sample_embeddings):
        """Test quantum statistics computation."""
        stats = contextualizer.get_quantum_statistics(sample_embeddings)
        
        required_keys = [
            "quantum_entropy", "quantum_coherence", "entanglement_measure",
            "decoherence_rate", "num_qubits", "num_layers"
        ]
        
        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], (float, int))
        
        # Check reasonable value ranges
        assert stats["quantum_entropy"] >= 0.0
        assert stats["quantum_coherence"] >= 0.0
        assert stats["entanglement_measure"] >= 0.0
        assert stats["num_qubits"] == contextualizer.num_qubits
        assert stats["num_layers"] == contextualizer.num_layers
    
    def test_attention_mask(self, contextualizer, sample_embeddings):
        """Test attention mask handling."""
        batch_size, seq_len = sample_embeddings.shape[:2]
        
        # Create attention mask (mask out last 3 tokens)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        attention_mask[:, -3:] = False
        
        enhanced_embeddings, _ = contextualizer(sample_embeddings, attention_mask=attention_mask)
        assert enhanced_embeddings.shape == sample_embeddings.shape
    
    def test_gradient_flow(self, contextualizer, sample_embeddings):
        """Test that gradients flow properly through the model."""
        contextualizer.train()
        enhanced_embeddings, quantum_states = contextualizer(
            sample_embeddings, return_quantum_states=True
        )
        
        # Create a simple loss
        loss = enhanced_embeddings.mean()
        loss.backward()
        
        # Check that gradients exist for key parameters
        assert contextualizer.state_encoder.weight.grad is not None
        assert contextualizer.state_decoder.weight.grad is not None
        
        for gate_gen in contextualizer.gate_generators.values():
            assert gate_gen.weight.grad is not None
    
    @pytest.mark.parametrize("num_qubits", [4, 6, 8])
    def test_different_qubit_numbers(self, num_qubits):
        """Test with different numbers of qubits."""
        contextualizer = QuantumContextualizer(
            hidden_dim=128,
            num_qubits=num_qubits,
            num_layers=2,
        )
        
        embeddings = torch.randn(1, 5, 128)
        enhanced_embeddings, _ = contextualizer(embeddings)
        assert enhanced_embeddings.shape == embeddings.shape
    
    @pytest.mark.parametrize("num_layers", [1, 2, 4])
    def test_different_layer_numbers(self, num_layers):
        """Test with different numbers of quantum layers."""
        contextualizer = QuantumContextualizer(
            hidden_dim=128,
            num_qubits=6,
            num_layers=num_layers,
        )
        
        embeddings = torch.randn(1, 5, 128)
        enhanced_embeddings, quantum_states = contextualizer(
            embeddings, return_quantum_states=True
        )
        
        assert enhanced_embeddings.shape == embeddings.shape
        assert len(quantum_states) == num_layers + 1
    
    def test_gate_types_configuration(self):
        """Test different gate type configurations."""
        gate_types = ['rx', 'ry']
        contextualizer = QuantumContextualizer(
            hidden_dim=64,
            num_qubits=4,
            gate_types=gate_types,
        )
        
        assert set(contextualizer.gate_types) == set(gate_types)
        assert len(contextualizer.gate_generators) == len(gate_types)
        
        embeddings = torch.randn(1, 3, 64)
        enhanced_embeddings, _ = contextualizer(embeddings)
        assert enhanced_embeddings.shape == embeddings.shape
    
    def test_device_compatibility(self, contextualizer, sample_embeddings):
        """Test CUDA compatibility if available."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            contextualizer = contextualizer.to(device)
            sample_embeddings = sample_embeddings.to(device)
            
            enhanced_embeddings, quantum_states = contextualizer(
                sample_embeddings, return_quantum_states=True
            )
            
            assert enhanced_embeddings.device == device
            if quantum_states:
                for q_state in quantum_states:
                    assert q_state.device == device
