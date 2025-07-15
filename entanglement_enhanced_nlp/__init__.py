"""
Entanglement Enhanced NLP

A groundbreaking framework that integrates quantum entanglement concepts into 
Natural Language Processing models, enabling more nuanced understanding of 
semantic relationships and superior context awareness.

Author: Krishna Bajpai (bajpaikrishna715@gmail.com)

This software is protected by licensing. Contact bajpaikrishna715@gmail.com for licensing information.
"""

__version__ = "1.0.1"
__author__ = "Krishna Bajpai"
__email__ = "bajpaikrishna715@gmail.com"
__description__ = "Quantum entanglement-inspired Natural Language Processing framework"

# Import licensing system first
from .licensing import get_license_manager, show_license_status, validate_class_license

# Validate basic license on package import
try:
    license_manager = get_license_manager()
    license_manager.validate_license(["core"])
    print("✅ Entanglement-Enhanced NLP licensed - quantum NLP capabilities activated!")
except Exception as e:
    print(f"⚠️ License validation failed: {e}")
    # Do not raise here to allow error inspection, but classes will enforce licensing

# Import core components (each will validate its own licensing requirements)
from .core.entangled_embedding import EntangledEmbedding
from .core.quantum_contextualizer import QuantumContextualizer
from .core.entangled_attention import EntangledAttention
from .transformers.entangled_transformer import EntangledTransformer
from .utils.quantum_simulator import QuantumSimulator
from .analysis.correlation_analyzer import CorrelationAnalyzer
from .visualization.entanglement_visualizer import EntanglementVisualizer

__all__ = [
    "EntangledEmbedding",
    "QuantumContextualizer", 
    "EntangledAttention",
    "EntangledTransformer",
    "QuantumSimulator",
    "CorrelationAnalyzer",
    "EntanglementVisualizer",
    "show_license_status",
    "get_license_manager"
]
