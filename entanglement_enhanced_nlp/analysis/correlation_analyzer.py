"""
Correlation Analyzer for entanglement-enhanced NLP.

This module provides comprehensive analysis tools for examining quantum
correlations, entanglement measures, and semantic relationships in
entanglement-enhanced NLP models.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx
from collections import defaultdict
import json

# Import licensing
from ..licensing import validate_class_license, requires_license


class CorrelationAnalyzer:
    """
    Comprehensive analyzer for quantum correlations and semantic relationships.
    
    This class provides tools for analyzing:
    - Quantum entanglement correlations
    - Semantic similarity patterns
    - Token interaction networks
    - Temporal correlation evolution
    - Cross-linguistic entanglement patterns
    
    LICENSE REQUIRED: This class requires a valid license to operate.
    Contact bajpaikrishna715@gmail.com for licensing information.
    
    Args:
        correlation_threshold: Minimum correlation strength to consider significant
        entanglement_threshold: Minimum entanglement measure to consider significant
        max_tokens: Maximum number of tokens to analyze in detail
        analysis_mode: Type of analysis ('comprehensive', 'fast', 'detailed')
    """
    
    def __init__(
        self,
        correlation_threshold: float = 0.1,
        entanglement_threshold: float = 0.05,
        max_tokens: int = 1000,
        analysis_mode: str = "comprehensive",
    ):
        # Validate license before allowing class instantiation
        validate_class_license(["correlation_analyzer"])
        
        self.correlation_threshold = correlation_threshold
        self.entanglement_threshold = entanglement_threshold
        self.max_tokens = max_tokens
        self.analysis_mode = analysis_mode
        
        # Storage for analysis results
        self.correlation_history = []
        self.entanglement_history = []
        self.semantic_networks = []
        self.temporal_patterns = []
        
        # Analysis cache
        self._cache = {}
    
    def analyze_token_correlations(
        self,
        embeddings: torch.Tensor,
        token_ids: torch.Tensor,
        tokenizer: Optional[Any] = None,
        correlation_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Analyze correlations between tokens in embeddings.
        
        Args:
            embeddings: Token embeddings (batch_size, seq_len, hidden_dim)
            token_ids: Token IDs (batch_size, seq_len)
            tokenizer: Tokenizer for converting IDs to text
            correlation_matrix: Pre-computed correlation matrix
            
        Returns:
            Dictionary with correlation analysis results
        """
        batch_size, seq_len, hidden_dim = embeddings.shape
        
        results = {
            "correlation_statistics": {},
            "significant_correlations": [],
            "correlation_matrix": None,
            "token_similarities": {},
            "cluster_analysis": {},
        }
        
        # Convert to numpy for analysis
        embeddings_np = embeddings.detach().cpu().numpy()
        token_ids_np = token_ids.detach().cpu().numpy()
        
        # Compute correlation matrix if not provided
        if correlation_matrix is None:
            correlation_matrix = self._compute_correlation_matrix(embeddings)
        
        correlation_np = correlation_matrix.detach().cpu().numpy()
        results["correlation_matrix"] = correlation_np
        
        # Analyze correlation statistics
        corr_stats = self._compute_correlation_statistics(correlation_np)
        results["correlation_statistics"] = corr_stats
        
        # Find significant correlations
        significant_corrs = self._find_significant_correlations(
            correlation_np, token_ids_np, tokenizer
        )
        results["significant_correlations"] = significant_corrs
        
        # Compute token similarities
        token_sims = self._compute_token_similarities(embeddings_np, token_ids_np, tokenizer)
        results["token_similarities"] = token_sims
        
        # Perform cluster analysis
        if self.analysis_mode in ["comprehensive", "detailed"]:
            cluster_analysis = self._perform_cluster_analysis(embeddings_np, token_ids_np)
            results["cluster_analysis"] = cluster_analysis
        
        # Store in history
        self.correlation_history.append(results)
        
        return results
    
    def analyze_quantum_entanglement(
        self,
        quantum_states: List[torch.Tensor],
        entanglement_matrices: List[torch.Tensor],
        token_ids: torch.Tensor,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Analyze quantum entanglement patterns in the model.
        
        Args:
            quantum_states: List of quantum state tensors from different layers
            entanglement_matrices: List of entanglement correlation matrices
            token_ids: Token IDs tensor
            tokenizer: Tokenizer for text conversion
            
        Returns:
            Dictionary with entanglement analysis results
        """
        results = {
            "entanglement_statistics": {},
            "layer_analysis": [],
            "entanglement_evolution": {},
            "quantum_measures": {},
            "entangled_pairs": [],
        }
        
        # Analyze each layer's quantum states
        for layer_idx, (q_state, ent_matrix) in enumerate(zip(quantum_states, entanglement_matrices)):
            layer_analysis = self._analyze_layer_entanglement(
                q_state, ent_matrix, layer_idx, token_ids, tokenizer
            )
            results["layer_analysis"].append(layer_analysis)
        
        # Analyze entanglement evolution across layers
        if len(quantum_states) > 1:
            evolution = self._analyze_entanglement_evolution(quantum_states, entanglement_matrices)
            results["entanglement_evolution"] = evolution
        
        # Compute quantum measures
        quantum_measures = self._compute_quantum_measures(quantum_states[-1])  # Use final layer
        results["quantum_measures"] = quantum_measures
        
        # Find strongly entangled token pairs
        entangled_pairs = self._find_entangled_pairs(
            entanglement_matrices[-1], token_ids, tokenizer
        )
        results["entangled_pairs"] = entangled_pairs
        
        # Store in history
        self.entanglement_history.append(results)
        
        return results
    
    def analyze_semantic_networks(
        self,
        embeddings: torch.Tensor,
        token_ids: torch.Tensor,
        tokenizer: Optional[Any] = None,
        correlation_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Analyze semantic networks formed by token relationships.
        
        Args:
            embeddings: Token embeddings
            token_ids: Token IDs
            tokenizer: Tokenizer for text conversion
            correlation_matrix: Pre-computed correlation matrix
            
        Returns:
            Dictionary with network analysis results
        """
        results = {
            "network_properties": {},
            "central_tokens": [],
            "communities": [],
            "semantic_clusters": {},
            "network_graph": None,
        }
        
        # Create semantic network graph
        network = self._create_semantic_network(embeddings, token_ids, tokenizer, correlation_matrix)
        results["network_graph"] = network
        
        # Analyze network properties
        network_props = self._analyze_network_properties(network)
        results["network_properties"] = network_props
        
        # Find central tokens (high degree, betweenness, etc.)
        central_tokens = self._find_central_tokens(network, tokenizer)
        results["central_tokens"] = central_tokens
        
        # Detect communities
        if self.analysis_mode in ["comprehensive", "detailed"]:
            communities = self._detect_communities(network, tokenizer)
            results["communities"] = communities
        
        # Perform semantic clustering
        semantic_clusters = self._perform_semantic_clustering(embeddings, token_ids, tokenizer)
        results["semantic_clusters"] = semantic_clusters
        
        # Store in network history
        self.semantic_networks.append(results)
        
        return results
    
    def _compute_correlation_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute correlation matrix between token embeddings."""
        batch_size, seq_len, hidden_dim = embeddings.shape
        
        # Flatten batch dimension
        flat_embeddings = embeddings.view(-1, hidden_dim)  # (B*L, H)
        
        # Compute correlation matrix
        correlation_matrix = torch.corrcoef(flat_embeddings)
        
        # Reshape to separate sequences
        correlation_matrix = correlation_matrix.view(batch_size, seq_len, batch_size, seq_len)
        
        # Average across batch dimension for analysis
        avg_correlation = correlation_matrix.mean(dim=(0, 2))  # (L, L)
        
        return avg_correlation
    
    def _compute_correlation_statistics(self, correlation_matrix: np.ndarray) -> Dict[str, float]:
        """Compute statistical measures of the correlation matrix."""
        # Remove diagonal (self-correlations)
        off_diagonal = correlation_matrix[~np.eye(correlation_matrix.shape[0], dtype=bool)]
        
        stats = {
            "mean_correlation": float(np.mean(off_diagonal)),
            "std_correlation": float(np.std(off_diagonal)),
            "max_correlation": float(np.max(off_diagonal)),
            "min_correlation": float(np.min(off_diagonal)),
            "median_correlation": float(np.median(off_diagonal)),
            "correlation_entropy": float(-np.sum(off_diagonal * np.log(np.abs(off_diagonal) + 1e-8))),
            "significant_correlations": int(np.sum(np.abs(off_diagonal) > self.correlation_threshold)),
            "sparsity": float(np.sum(np.abs(off_diagonal) < 0.01) / len(off_diagonal)),
        }
        
        return stats
    
    def _find_significant_correlations(
        self,
        correlation_matrix: np.ndarray,
        token_ids: np.ndarray,
        tokenizer: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Find pairs of tokens with significant correlations."""
        significant_pairs = []
        seq_len = correlation_matrix.shape[0]
        
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                correlation = correlation_matrix[i, j]
                
                if abs(correlation) > self.correlation_threshold:
                    pair_info = {
                        "position_i": i,
                        "position_j": j,
                        "correlation": float(correlation),
                        "distance": j - i,
                    }
                    
                    if tokenizer is not None and len(token_ids) > 0:
                        # Get token text (use first sequence in batch)
                        token_i = tokenizer.decode([token_ids[0, i]]) if i < token_ids.shape[1] else "UNK"
                        token_j = tokenizer.decode([token_ids[0, j]]) if j < token_ids.shape[1] else "UNK"
                        pair_info["token_i"] = token_i
                        pair_info["token_j"] = token_j
                    
                    significant_pairs.append(pair_info)
        
        # Sort by correlation strength
        significant_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return significant_pairs[:100]  # Return top 100 pairs
    
    def _compute_token_similarities(
        self,
        embeddings: np.ndarray,
        token_ids: np.ndarray,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Compute various similarity measures between tokens."""
        batch_size, seq_len, hidden_dim = embeddings.shape
        
        # Average embeddings across batch
        avg_embeddings = np.mean(embeddings, axis=0)  # (L, H)
        
        # Compute different similarity measures
        cosine_sim = cosine_similarity(avg_embeddings)
        
        similarities = {
            "cosine_similarity": cosine_sim.tolist(),
            "euclidean_distances": [],
            "manhattan_distances": [],
        }
        
        # Compute distance measures
        for i in range(seq_len):
            euclidean_row = []
            manhattan_row = []
            for j in range(seq_len):
                euclidean_dist = np.linalg.norm(avg_embeddings[i] - avg_embeddings[j])
                manhattan_dist = np.sum(np.abs(avg_embeddings[i] - avg_embeddings[j]))
                euclidean_row.append(float(euclidean_dist))
                manhattan_row.append(float(manhattan_dist))
            
            similarities["euclidean_distances"].append(euclidean_row)
            similarities["manhattan_distances"].append(manhattan_row)
        
        return similarities
    
    def _perform_cluster_analysis(
        self,
        embeddings: np.ndarray,
        token_ids: np.ndarray,
    ) -> Dict[str, Any]:
        """Perform clustering analysis on token embeddings."""
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.metrics import silhouette_score
        
        batch_size, seq_len, hidden_dim = embeddings.shape
        
        # Flatten embeddings for clustering
        flat_embeddings = embeddings.reshape(-1, hidden_dim)
        
        # Reduce dimensionality for visualization
        pca = PCA(n_components=min(50, hidden_dim))
        reduced_embeddings = pca.fit_transform(flat_embeddings)
        
        # K-means clustering
        optimal_k = min(10, len(flat_embeddings) // 2)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans_labels = kmeans.fit_predict(reduced_embeddings)
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        dbscan_labels = dbscan.fit_predict(reduced_embeddings)
        
        # Compute silhouette scores
        kmeans_silhouette = silhouette_score(reduced_embeddings, kmeans_labels)
        if len(set(dbscan_labels)) > 1:
            dbscan_silhouette = silhouette_score(reduced_embeddings, dbscan_labels)
        else:
            dbscan_silhouette = -1.0
        
        cluster_analysis = {
            "kmeans_clusters": kmeans_labels.tolist(),
            "dbscan_clusters": dbscan_labels.tolist(),
            "kmeans_silhouette": float(kmeans_silhouette),
            "dbscan_silhouette": float(dbscan_silhouette),
            "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
            "optimal_k": optimal_k,
        }
        
        return cluster_analysis
    
    def _analyze_layer_entanglement(
        self,
        quantum_state: torch.Tensor,
        entanglement_matrix: torch.Tensor,
        layer_idx: int,
        token_ids: torch.Tensor,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Analyze entanglement for a specific layer."""
        # Convert to numpy
        q_state_np = quantum_state.detach().cpu().numpy()
        ent_matrix_np = entanglement_matrix.detach().cpu().numpy()
        
        # Compute entanglement measures
        entanglement_stats = {
            "layer_index": layer_idx,
            "mean_entanglement": float(np.mean(ent_matrix_np)),
            "max_entanglement": float(np.max(ent_matrix_np)),
            "entanglement_variance": float(np.var(ent_matrix_np)),
            "quantum_coherence": self._compute_quantum_coherence(q_state_np),
        }
        
        # Find strongest entangled pairs for this layer
        strong_pairs = self._find_layer_entangled_pairs(ent_matrix_np, token_ids, tokenizer)
        entanglement_stats["strong_pairs"] = strong_pairs
        
        return entanglement_stats
    
    def _compute_quantum_coherence(self, quantum_state: np.ndarray) -> float:
        """Compute quantum coherence measure."""
        if quantum_state.ndim == 4:  # (B, L, Q, 2) - complex representation
            # Compute coherence as off-diagonal elements of density matrix
            batch_size, seq_len, num_qubits, _ = quantum_state.shape
            
            coherence_sum = 0.0
            for b in range(batch_size):
                for l in range(seq_len):
                    # Convert to complex numbers
                    complex_state = quantum_state[b, l, :, 0] + 1j * quantum_state[b, l, :, 1]
                    
                    # Compute density matrix
                    density_matrix = np.outer(complex_state, np.conj(complex_state))
                    
                    # Compute coherence (sum of off-diagonal magnitudes)
                    off_diagonal_sum = np.sum(np.abs(density_matrix)) - np.sum(np.abs(np.diag(density_matrix)))
                    coherence_sum += off_diagonal_sum
            
            return float(coherence_sum / (batch_size * seq_len))
        
        return 0.0
    
    def _find_layer_entangled_pairs(
        self,
        entanglement_matrix: np.ndarray,
        token_ids: torch.Tensor,
        tokenizer: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Find strongly entangled pairs in a specific layer."""
        pairs = []
        seq_len = entanglement_matrix.shape[0]
        
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                entanglement = entanglement_matrix[i, j]
                
                if abs(entanglement) > self.entanglement_threshold:
                    pair = {
                        "position_i": i,
                        "position_j": j,
                        "entanglement": float(entanglement),
                        "distance": j - i,
                    }
                    
                    if tokenizer is not None:
                        try:
                            token_i = tokenizer.decode([token_ids[0, i].item()])
                            token_j = tokenizer.decode([token_ids[0, j].item()])
                            pair["token_i"] = token_i
                            pair["token_j"] = token_j
                        except:
                            pair["token_i"] = "UNK"
                            pair["token_j"] = "UNK"
                    
                    pairs.append(pair)
        
        return sorted(pairs, key=lambda x: abs(x["entanglement"]), reverse=True)[:20]
    
    def _create_semantic_network(
        self,
        embeddings: torch.Tensor,
        token_ids: torch.Tensor,
        tokenizer: Optional[Any] = None,
        correlation_matrix: Optional[torch.Tensor] = None,
    ) -> nx.Graph:
        """Create a semantic network graph from token relationships."""
        if correlation_matrix is None:
            correlation_matrix = self._compute_correlation_matrix(embeddings)
        
        correlation_np = correlation_matrix.detach().cpu().numpy()
        seq_len = correlation_np.shape[0]
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes (tokens)
        for i in range(seq_len):
            node_attrs = {"position": i}
            if tokenizer is not None:
                try:
                    token_text = tokenizer.decode([token_ids[0, i].item()])
                    node_attrs["token"] = token_text
                except:
                    node_attrs["token"] = f"token_{i}"
            else:
                node_attrs["token"] = f"token_{i}"
            
            G.add_node(i, **node_attrs)
        
        # Add edges based on correlations
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                correlation = correlation_np[i, j]
                if abs(correlation) > self.correlation_threshold:
                    G.add_edge(i, j, weight=abs(correlation), correlation=correlation)
        
        return G
    
    def _analyze_network_properties(self, network: nx.Graph) -> Dict[str, Any]:
        """Analyze properties of the semantic network."""
        properties = {
            "num_nodes": network.number_of_nodes(),
            "num_edges": network.number_of_edges(),
            "density": nx.density(network),
            "is_connected": nx.is_connected(network),
        }
        
        if network.number_of_edges() > 0:
            properties["average_clustering"] = nx.average_clustering(network)
            properties["transitivity"] = nx.transitivity(network)
            
            if nx.is_connected(network):
                properties["diameter"] = nx.diameter(network)
                properties["average_shortest_path"] = nx.average_shortest_path_length(network)
            else:
                # Analyze largest connected component
                largest_cc = max(nx.connected_components(network), key=len)
                subgraph = network.subgraph(largest_cc)
                properties["largest_component_size"] = len(largest_cc)
                properties["diameter"] = nx.diameter(subgraph)
                properties["average_shortest_path"] = nx.average_shortest_path_length(subgraph)
        
        return properties
    
    def _find_central_tokens(self, network: nx.Graph, tokenizer: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Find central tokens in the semantic network."""
        central_tokens = []
        
        if network.number_of_edges() == 0:
            return central_tokens
        
        # Compute centrality measures
        degree_centrality = nx.degree_centrality(network)
        betweenness_centrality = nx.betweenness_centrality(network)
        closeness_centrality = nx.closeness_centrality(network)
        eigenvector_centrality = nx.eigenvector_centrality(network)
        
        # Combine centrality measures
        for node in network.nodes():
            node_data = network.nodes[node]
            central_tokens.append({
                "node_id": node,
                "token": node_data.get("token", f"token_{node}"),
                "degree_centrality": degree_centrality[node],
                "betweenness_centrality": betweenness_centrality[node],
                "closeness_centrality": closeness_centrality[node],
                "eigenvector_centrality": eigenvector_centrality[node],
                "combined_centrality": (
                    degree_centrality[node] + betweenness_centrality[node] + 
                    closeness_centrality[node] + eigenvector_centrality[node]
                ) / 4,
            })
        
        # Sort by combined centrality
        central_tokens.sort(key=lambda x: x["combined_centrality"], reverse=True)
        
        return central_tokens[:20]  # Return top 20 central tokens
    
    def export_analysis_results(self, filepath: str, format: str = "json") -> None:
        """
        Export analysis results to file.
        
        Args:
            filepath: Path to save results
            format: Export format ('json', 'csv', 'pickle')
        """
        results = {
            "correlation_history": self.correlation_history,
            "entanglement_history": self.entanglement_history,
            "semantic_networks": [
                {
                    "network_properties": net["network_properties"],
                    "central_tokens": net["central_tokens"],
                    "communities": net.get("communities", []),
                }
                for net in self.semantic_networks
            ],
            "analysis_parameters": {
                "correlation_threshold": self.correlation_threshold,
                "entanglement_threshold": self.entanglement_threshold,
                "max_tokens": self.max_tokens,
                "analysis_mode": self.analysis_mode,
            },
        }
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        elif format == "pickle":
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
        
        elif format == "csv":
            # Export flattened results to CSV
            import pandas as pd
            
            # Flatten correlation statistics
            corr_data = []
            for i, corr_hist in enumerate(self.correlation_history):
                stats = corr_hist["correlation_statistics"]
                stats["analysis_id"] = i
                corr_data.append(stats)
            
            df_corr = pd.DataFrame(corr_data)
            df_corr.to_csv(filepath.replace('.csv', '_correlations.csv'), index=False)
            
            print(f"Results exported to {filepath}")
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of all analysis results."""
        summary = {
            "total_analyses": len(self.correlation_history),
            "total_networks": len(self.semantic_networks),
            "average_correlations": [],
            "entanglement_trends": [],
            "network_evolution": [],
        }
        
        # Summarize correlation trends
        for corr_hist in self.correlation_history:
            stats = corr_hist["correlation_statistics"]
            summary["average_correlations"].append(stats.get("mean_correlation", 0.0))
        
        # Summarize entanglement trends
        for ent_hist in self.entanglement_history:
            if ent_hist["layer_analysis"]:
                avg_entanglement = np.mean([
                    layer["mean_entanglement"] for layer in ent_hist["layer_analysis"]
                ])
                summary["entanglement_trends"].append(float(avg_entanglement))
        
        # Summarize network evolution
        for net in self.semantic_networks:
            props = net["network_properties"]
            summary["network_evolution"].append({
                "num_nodes": props.get("num_nodes", 0),
                "num_edges": props.get("num_edges", 0),
                "density": props.get("density", 0.0),
                "clustering": props.get("average_clustering", 0.0),
            })
        
        return summary
