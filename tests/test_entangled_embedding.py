"""
Tests for EntangledEmbedding module.
"""

import pytest
import torch
import numpy as np
from entanglement_enhanced_nlp.core.entangled_embedding import EntangledEmbedding


class TestEntangledEmbedding:
    """Test cases for EntangledEmbedding class."""
    
    @pytest.fixture
    def embedding_layer(self):
        """Create a test EntangledEmbedding instance."""
        return EntangledEmbedding(
            vocab_size=1000,
            embedding_dim=128,
            entanglement_depth=2,
            correlation_strength=0.7,
            decoherence_rate=0.1,
            max_position_embeddings=256,
        )
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randint(0, 1000, (2, 10))  # (batch_size=2, seq_len=10)
    
    def test_initialization(self, embedding_layer):
        """Test proper initialization of EntangledEmbedding."""
        assert embedding_layer.vocab_size == 1000
        assert embedding_layer.embedding_dim == 128
        assert embedding_layer.entanglement_depth == 2
        assert embedding_layer.correlation_strength == 0.7
        assert embedding_layer.decoherence_rate == 0.1
        
        # Check parameter initialization
        assert hasattr(embedding_layer, 'base_embedding')
        assert hasattr(embedding_layer, 'quantum_amplitudes')
        assert hasattr(embedding_layer, 'entanglement_matrix')
        assert hasattr(embedding_layer, 'evolution_operators')
    
    def test_forward_pass(self, embedding_layer, sample_input):
        """Test forward pass through EntangledEmbedding."""
        # Test without correlations
        embeddings, correlations = embedding_layer(sample_input)
        
        assert embeddings.shape == (2, 10, 128)  # (batch, seq_len, hidden_dim)
        assert correlations is None
        
        # Test with correlations
        embeddings, correlations = embedding_layer(sample_input, return_correlations=True)
        
        assert embeddings.shape == (2, 10, 128)
        assert correlations is not None
        assert correlations.shape == (2, 10, 10)  # (batch, seq_len, seq_len)
    
    def test_quantum_correlations(self, embedding_layer, sample_input):
        """Test quantum correlation computation."""
        correlations = embedding_layer._compute_quantum_correlations(sample_input)
        
        assert correlations.shape == (2, 10, 10)
        assert torch.allclose(correlations, correlations.transpose(-2, -1))  # Should be symmetric
        
        # Check correlation values are in reasonable range
        assert torch.all(correlations >= -1.0)
        assert torch.all(correlations <= 1.0)
    
    def test_quantum_evolution(self, embedding_layer):
        """Test quantum state evolution."""
        batch_size, seq_len, hidden_dim = 2, 10, 128
        embeddings = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Test each evolution step
        for step in range(embedding_layer.entanglement_depth):
            evolved = embedding_layer._apply_quantum_evolution(embeddings, step)
            assert evolved.shape == embeddings.shape
            
            # Check that evolution changes the embeddings
            assert not torch.allclose(evolved, embeddings, atol=1e-6)
            embeddings = evolved
    
    def test_superposition_states(self, embedding_layer, sample_input):
        """Test creation of quantum superposition states."""
        base_embeddings = embedding_layer.base_embedding(sample_input)
        superposition = embedding_layer._create_superposition_states(sample_input, base_embeddings)
        
        assert superposition.shape == base_embeddings.shape
        # Superposition should be different from base embeddings
        assert not torch.allclose(superposition, base_embeddings, atol=1e-6)
    
    def test_entanglement_statistics(self, embedding_layer, sample_input):
        """Test entanglement statistics computation."""
        stats = embedding_layer.get_entanglement_statistics(sample_input)
        
        required_keys = [
            "average_correlation", "maximum_correlation", "entanglement_entropy",
            "quantum_coherence", "decoherence_rate", "correlation_strength"
        ]
        
        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], float)
        
        # Check reasonable value ranges
        assert -1.0 <= stats["average_correlation"] <= 1.0
        assert -1.0 <= stats["maximum_correlation"] <= 1.0
        assert stats["entanglement_entropy"] >= 0.0
        assert stats["quantum_coherence"] >= 0.0
    
    def test_position_encoding(self, embedding_layer):
        """Test positional encoding creation."""
        max_len, d_model = 100, 128
        pos_encoding = embedding_layer._create_positional_encoding(max_len, d_model)
        
        assert pos_encoding.shape == (1, max_len, d_model)
        
        # Check that positions are different
        pos_0 = pos_encoding[0, 0, :]
        pos_1 = pos_encoding[0, 1, :]
        assert not torch.allclose(pos_0, pos_1)
    
    def test_gradient_flow(self, embedding_layer, sample_input):
        """Test that gradients flow properly through the model."""
        embedding_layer.train()
        embeddings, correlations = embedding_layer(sample_input, return_correlations=True)
        
        # Create a simple loss
        loss = embeddings.mean() + (correlations.mean() if correlations is not None else 0)
        loss.backward()
        
        # Check that gradients exist for key parameters
        assert embedding_layer.base_embedding.weight.grad is not None
        assert embedding_layer.quantum_amplitudes.grad is not None
        assert embedding_layer.entanglement_matrix.grad is not None
    
    def test_different_sequence_lengths(self, embedding_layer):
        """Test handling of different sequence lengths."""
        for seq_len in [5, 15, 50]:
            input_ids = torch.randint(0, 1000, (1, seq_len))
            embeddings, _ = embedding_layer(input_ids)
            assert embeddings.shape == (1, seq_len, 128)
    
    def test_batch_consistency(self, embedding_layer):
        """Test that processing maintains consistency across batch sizes."""
        single_input = torch.randint(0, 1000, (1, 10))
        batch_input = single_input.repeat(4, 1)
        
        single_embeddings, _ = embedding_layer(single_input)
        batch_embeddings, _ = embedding_layer(batch_input)
        
        # First item in batch should match single processing
        assert torch.allclose(single_embeddings[0], batch_embeddings[0], atol=1e-6)
    
    @pytest.mark.parametrize("correlation_strength", [0.1, 0.5, 0.9])
    def test_correlation_strength_effect(self, correlation_strength):
        """Test effect of different correlation strengths."""
        embedding_layer = EntangledEmbedding(
            vocab_size=100,
            embedding_dim=64,
            correlation_strength=correlation_strength,
        )
        
        input_ids = torch.randint(0, 100, (1, 5))
        stats = embedding_layer.get_entanglement_statistics(input_ids)
        
        # Higher correlation strength should generally lead to higher correlations
        assert abs(stats["correlation_strength"] - correlation_strength) < 1e-6
    
    @pytest.mark.parametrize("decoherence_rate", [0.0, 0.1, 0.5])
    def test_decoherence_effect(self, decoherence_rate):
        """Test effect of different decoherence rates."""
        embedding_layer = EntangledEmbedding(
            vocab_size=100,
            embedding_dim=64,
            decoherence_rate=decoherence_rate,
        )
        
        input_ids = torch.randint(0, 100, (1, 5))
        stats = embedding_layer.get_entanglement_statistics(input_ids)
        
        assert abs(stats["decoherence_rate"] - decoherence_rate) < 1e-6
    
    def test_device_compatibility(self, embedding_layer, sample_input):
        """Test CUDA compatibility if available."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            embedding_layer = embedding_layer.to(device)
            sample_input = sample_input.to(device)
            
            embeddings, correlations = embedding_layer(sample_input, return_correlations=True)
            
            assert embeddings.device == device
            if correlations is not None:
                assert correlations.device == device
