"""
Entanglement Enhanced NLP

A groundbreaking framework that integrates quantum entanglement concepts into 
Natural Language Processing models, enabling more nuanced understanding of 
semantic relationships and superior context awareness.

Author: Krishna Bajpai (bajpaikrishna715@gmail.com)
"""

__version__ = "0.1.0"
__author__ = "Krishna Bajpai"
__email__ = "bajpaikrishna715@gmail.com"
__description__ = "Quantum entanglement-inspired Natural Language Processing framework"

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
]
