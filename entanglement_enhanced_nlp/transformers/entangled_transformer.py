"""
Entangled Transformer implementation.

This module extends HuggingFace transformers with quantum entanglement-aware
attention mechanisms, enabling enhanced semantic understanding through
quantum-inspired processing pipelines.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List, Union, Any
from transformers import (
    PreTrainedModel, 
    PretrainedConfig,
    AutoModel,
    AutoTokenizer,
    BertConfig,
    BertModel,
)
from transformers.modeling_outputs import BaseModelOutput
import warnings

from ..core.entangled_attention import EntangledAttention
from ..core.quantum_contextualizer import QuantumContextualizer
from ..core.entangled_embedding import EntangledEmbedding

# Import licensing
from ..licensing import validate_class_license, requires_license


class EntangledTransformerConfig(PretrainedConfig):
    """
    Configuration class for EntangledTransformer.
    
    This configuration extends the standard transformer configuration with
    quantum entanglement-specific parameters.
    
    LICENSE REQUIRED: This class requires a valid license to operate.
    Contact bajpaikrishna715@gmail.com for licensing information.
    """
    
    model_type = "entangled_transformer"
    
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        position_embedding_type: str = "absolute",
        # Quantum entanglement parameters
        entanglement_depth: int = 3,
        correlation_strength: float = 0.8,
        decoherence_rate: float = 0.1,
        entanglement_strength: float = 0.5,
        correlation_decay: float = 0.1,
        superposition_weight: float = 0.3,
        num_qubits: int = 8,
        quantum_layers: List[int] = None,
        use_quantum_contextualizer: bool = True,
        use_entangled_embeddings: bool = True,
        gate_types: List[str] = None,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        
        # Quantum parameters
        self.entanglement_depth = entanglement_depth
        self.correlation_strength = correlation_strength
        self.decoherence_rate = decoherence_rate
        self.entanglement_strength = entanglement_strength
        self.correlation_decay = correlation_decay
        self.superposition_weight = superposition_weight
        self.num_qubits = num_qubits
        self.quantum_layers = quantum_layers or list(range(num_hidden_layers))
        self.use_quantum_contextualizer = use_quantum_contextualizer
        self.use_entangled_embeddings = use_entangled_embeddings
        self.gate_types = gate_types or ['rx', 'ry', 'rz', 'cnot']


class EntangledTransformerLayer(nn.Module):
    """
    Single transformer layer with quantum entanglement enhancement.
    
    This layer integrates quantum-inspired attention and contextualizer
    into a standard transformer architecture.
    
    LICENSE REQUIRED: This class requires a valid license to operate.
    Contact bajpaikrishna715@gmail.com for licensing information.
    """
    
    def __init__(self, config: EntangledTransformerConfig):
        # Validate license before allowing class instantiation
        validate_class_license(["entangled_transformer"])
        
        super().__init__()
        
        self.config = config
        
        # Standard transformer components
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Entangled attention mechanism
        self.entangled_attention = EntangledAttention(
            hidden_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            entanglement_strength=config.entanglement_strength,
            correlation_decay=config.correlation_decay,
            superposition_weight=config.superposition_weight,
            dropout=config.attention_probs_dropout_prob,
        )
        
        # Feed-forward network
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Optional quantum contextualizer
        if config.use_quantum_contextualizer:
            self.quantum_contextualizer = QuantumContextualizer(
                hidden_dim=config.hidden_size,
                num_qubits=config.num_qubits,
                num_layers=3,
                decoherence_rate=config.decoherence_rate,
                gate_types=config.gate_types,
                dropout=config.hidden_dropout_prob,
            )
        else:
            self.quantum_contextualizer = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
        return_quantum_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through entangled transformer layer.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask for padding tokens
            position_bias: Optional positional bias
            return_attention_weights: Whether to return attention weights
            return_quantum_states: Whether to return quantum states
            
        Returns:
            Tuple of (hidden_states, attention_weights, quantum_states)
        """
        # Layer normalization before attention
        normed_hidden_states = self.layer_norm_1(hidden_states)
        
        # Apply entangled attention
        attention_output, attention_weights, _ = self.entangled_attention(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            return_attention_weights=return_attention_weights,
        )
        
        # Residual connection
        hidden_states = hidden_states + attention_output
        
        # Apply quantum contextualizer if enabled
        quantum_states = None
        if self.quantum_contextualizer is not None:
            normed_states = self.layer_norm_2(hidden_states)
            contextualized_states, quantum_states = self.quantum_contextualizer(
                normed_states,
                attention_mask=attention_mask,
                return_quantum_states=return_quantum_states,
            )
            hidden_states = hidden_states + contextualized_states
        
        # Feed-forward network with residual connection
        normed_states = self.layer_norm_2(hidden_states)
        intermediate_output = torch.relu(self.intermediate(normed_states))
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        hidden_states = hidden_states + layer_output
        
        return hidden_states, attention_weights, quantum_states


class EntangledTransformer(PreTrainedModel):
    """
    Transformer model with quantum entanglement enhancements.
    
    This model extends standard transformer architectures with quantum-inspired
    mechanisms for enhanced semantic understanding and context modeling.
    
    LICENSE REQUIRED: This class requires a valid license to operate.
    Contact bajpaikrishna715@gmail.com for licensing information.
    """
    
    config_class = EntangledTransformerConfig
    base_model_prefix = "entangled_transformer"
    
    def __init__(self, config: EntangledTransformerConfig):
        # Validate license before allowing class instantiation
        validate_class_license(["entangled_transformer"])
        
        super().__init__(config)
        
        self.config = config
        
        # Embeddings
        if config.use_entangled_embeddings:
            self.embeddings = EntangledEmbedding(
                vocab_size=config.vocab_size,
                embedding_dim=config.hidden_size,
                entanglement_depth=config.entanglement_depth,
                correlation_strength=config.correlation_strength,
                decoherence_rate=config.decoherence_rate,
                max_position_embeddings=config.max_position_embeddings,
                dropout=config.hidden_dropout_prob,
            )
        else:
            # Use standard embeddings as fallback
            self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
        
        # Token type embeddings
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            EntangledTransformerLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize weights
        self.init_weights()
    
    def get_input_embeddings(self) -> nn.Module:
        """Get input embeddings."""
        if isinstance(self.embeddings, EntangledEmbedding):
            return self.embeddings.base_embedding
        return self.embeddings
    
    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set input embeddings."""
        if isinstance(self.embeddings, EntangledEmbedding):
            self.embeddings.base_embedding = value
        else:
            self.embeddings = value
    
    def _prepare_embeddings(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Prepare input embeddings with quantum enhancements."""
        seq_len = input_ids.shape[1]
        device = input_ids.device
        
        if isinstance(self.embeddings, EntangledEmbedding):
            # Use entangled embeddings
            embeddings, _ = self.embeddings(input_ids, position_ids)
        else:
            # Use standard embeddings
            embeddings = self.embeddings(input_ids)
            
            # Add positional embeddings
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        
        # Add token type embeddings
        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings = embeddings + token_type_embeddings
        
        return self.dropout(embeddings)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_quantum_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutput]:
        """
        Forward pass through entangled transformer.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask for padding tokens
            token_type_ids: Token type IDs for segment embedding
            position_ids: Position IDs for positional embedding
            inputs_embeds: Pre-computed embeddings (alternative to input_ids)
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states from all layers
            output_quantum_states: Whether to output quantum states
            return_dict: Whether to return ModelOutput object
            
        Returns:
            Model outputs with optional attention weights and quantum states
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_len = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_len = inputs_embeds.shape[:2]
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Prepare embeddings
        if inputs_embeds is None:
            hidden_states = self._prepare_embeddings(input_ids, token_type_ids, position_ids)
        else:
            hidden_states = inputs_embeds
        
        # Apply layer normalization to embeddings
        hidden_states = self.layer_norm(hidden_states)
        
        # Store outputs
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_quantum_states = () if output_quantum_states else None
        
        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Apply entangled transformer layer
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                return_attention_weights=output_attentions,
                return_quantum_states=output_quantum_states,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions and layer_outputs[1] is not None:
                all_attentions = all_attentions + (layer_outputs[1],)
            
            if output_quantum_states and layer_outputs[2] is not None:
                all_quantum_states = all_quantum_states + (layer_outputs[2],)
        
        # Final layer normalization
        hidden_states = self.layer_norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [
                hidden_states,
                all_hidden_states,
                all_attentions,
                all_quantum_states,
            ] if v is not None)
        
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
    
    def get_entanglement_statistics(self, input_ids: torch.Tensor) -> Dict[str, Any]:
        """
        Compute comprehensive entanglement statistics.
        
        Args:
            input_ids: Input token IDs
            
        Returns:
            Dictionary with entanglement metrics from all components
        """
        stats = {}
        
        # Get embedding statistics
        if isinstance(self.embeddings, EntangledEmbedding):
            stats["embedding_stats"] = self.embeddings.get_entanglement_statistics(input_ids)
        
        # Get layer statistics
        with torch.no_grad():
            hidden_states = self._prepare_embeddings(input_ids)
            
            layer_stats = []
            for i, layer in enumerate(self.layers):
                layer_stat = {}
                
                # Attention statistics
                attention_stats = layer.entangled_attention.get_entanglement_statistics(hidden_states)
                layer_stat["attention"] = attention_stats
                
                # Quantum contextualizer statistics
                if layer.quantum_contextualizer is not None:
                    quantum_stats = layer.quantum_contextualizer.get_quantum_statistics(hidden_states)
                    layer_stat["quantum_contextualizer"] = quantum_stats
                
                layer_stats.append(layer_stat)
                
                # Update hidden states for next layer
                hidden_states, _, _ = layer(hidden_states)
            
            stats["layer_stats"] = layer_stats
        
        return stats


def create_entangled_transformer_from_pretrained(
    model_name_or_path: str,
    entanglement_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> EntangledTransformer:
    """
    Create an EntangledTransformer from a pre-trained model.
    
    This function loads a pre-trained transformer and enhances it with
    quantum entanglement capabilities.
    
    Args:
        model_name_or_path: Pre-trained model name or path
        entanglement_config: Configuration for entanglement parameters
        **kwargs: Additional arguments for model loading
        
    Returns:
        EntangledTransformer model with quantum enhancements
    """
    # Load base configuration
    base_config = AutoModel.from_pretrained(model_name_or_path, config=True).config
    
    # Create entangled configuration
    config_dict = base_config.to_dict()
    
    # Add entanglement parameters
    if entanglement_config:
        config_dict.update(entanglement_config)
    
    entangled_config = EntangledTransformerConfig(**config_dict)
    
    # Create entangled model
    model = EntangledTransformer(entangled_config)
    
    # Load pre-trained weights where possible
    try:
        base_model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        
        # Transfer compatible weights
        model_dict = model.state_dict()
        pretrained_dict = base_model.state_dict()
        
        # Filter out incompatible keys
        compatible_dict = {
            k: v for k, v in pretrained_dict.items() 
            if k in model_dict and v.shape == model_dict[k].shape
        }
        
        model_dict.update(compatible_dict)
        model.load_state_dict(model_dict)
        
        print(f"Loaded {len(compatible_dict)} compatible parameters from {model_name_or_path}")
        
    except Exception as e:
        warnings.warn(f"Could not load pre-trained weights: {e}")
    
    return model
