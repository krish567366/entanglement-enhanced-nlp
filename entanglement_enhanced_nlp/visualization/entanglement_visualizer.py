"""
Entanglement Visualizer for quantum-enhanced NLP.

This module provides comprehensive visualization tools for analyzing and 
displaying quantum entanglement patterns, semantic correlations, and 
network structures in entanglement-enhanced NLP models.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import io
import base64


class EntanglementVisualizer:
    """
    Comprehensive visualizer for quantum entanglement and semantic analysis.
    
    This class provides various visualization methods for:
    - Quantum entanglement correlation heatmaps
    - Semantic network graphs
    - Token interaction patterns
    - Temporal evolution of quantum states
    - Dimensionality reduction plots
    - Interactive dashboards
    
    Args:
        style: Matplotlib/seaborn style ('default', 'dark', 'scientific')
        color_palette: Color palette for visualizations
        figure_size: Default figure size (width, height)
        dpi: Resolution for saved figures
        interactive: Whether to use interactive plotly visualizations
    """
    
    def __init__(
        self,
        style: str = "default",
        color_palette: str = "viridis",
        figure_size: Tuple[int, int] = (12, 8),
        dpi: int = 300,
        interactive: bool = True,
    ):
        self.style = style
        self.color_palette = color_palette
        self.figure_size = figure_size
        self.dpi = dpi
        self.interactive = interactive
        
        # Set matplotlib style
        self._setup_matplotlib_style()
        
        # Color schemes for different types of plots
        self.color_schemes = {
            "entanglement": "plasma",
            "correlation": "RdBu_r", 
            "network": "Set3",
            "quantum": "cool",
            "semantic": "tab10",
        }
    
    def _setup_matplotlib_style(self) -> None:
        """Setup matplotlib and seaborn styles."""
        if self.style == "dark":
            plt.style.use("dark_background")
            sns.set_theme(style="darkgrid")
        elif self.style == "scientific":
            plt.style.use("seaborn-v0_8-paper")
            sns.set_theme(style="whitegrid")
        else:
            sns.set_theme(style="whitegrid")
        
        # Set default parameters
        plt.rcParams['figure.figsize'] = self.figure_size
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
    
    def plot_entanglement_heatmap(
        self,
        entanglement_matrix: Union[torch.Tensor, np.ndarray],
        token_labels: Optional[List[str]] = None,
        title: str = "Quantum Entanglement Correlation Matrix",
        save_path: Optional[str] = None,
        interactive: Optional[bool] = None,
    ) -> Union[plt.Figure, go.Figure]:
        """
        Create a heatmap visualization of quantum entanglement correlations.
        
        Args:
            entanglement_matrix: Matrix of entanglement correlations
            token_labels: Labels for tokens (optional)
            title: Title for the plot
            save_path: Path to save the figure
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib figure or Plotly figure object
        """
        # Convert to numpy if needed
        if isinstance(entanglement_matrix, torch.Tensor):
            matrix = entanglement_matrix.detach().cpu().numpy()
        else:
            matrix = entanglement_matrix
        
        use_interactive = interactive if interactive is not None else self.interactive
        
        if use_interactive:
            return self._create_interactive_heatmap(matrix, token_labels, title, save_path)
        else:
            return self._create_static_heatmap(matrix, token_labels, title, save_path)
    
    def _create_static_heatmap(
        self,
        matrix: np.ndarray,
        token_labels: Optional[List[str]],
        title: str,
        save_path: Optional[str],
    ) -> plt.Figure:
        """Create static heatmap using matplotlib/seaborn."""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create heatmap
        sns.heatmap(
            matrix,
            annot=False,
            cmap=self.color_schemes["entanglement"],
            center=0,
            square=True,
            linewidths=0.1,
            cbar_kws={"shrink": 0.8, "label": "Entanglement Strength"},
            xticklabels=token_labels if token_labels else False,
            yticklabels=token_labels if token_labels else False,
            ax=ax
        )
        
        ax.set_title(title, fontsize=16, pad=20)
        ax.set_xlabel("Token Position", fontsize=12)
        ax.set_ylabel("Token Position", fontsize=12)
        
        # Rotate labels if provided
        if token_labels:
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _create_interactive_heatmap(
        self,
        matrix: np.ndarray,
        token_labels: Optional[List[str]],
        title: str,
        save_path: Optional[str],
    ) -> go.Figure:
        """Create interactive heatmap using plotly."""
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=token_labels if token_labels else list(range(matrix.shape[1])),
            y=token_labels if token_labels else list(range(matrix.shape[0])),
            colorscale=self.color_schemes["entanglement"],
            colorbar=dict(title="Entanglement Strength"),
            hovertemplate="Token X: %{x}<br>Token Y: %{y}<br>Entanglement: %{z:.4f}<extra></extra>",
        ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis_title="Token Position",
            yaxis_title="Token Position",
            width=800,
            height=600,
        )
        
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, width=800, height=600)
        
        return fig
    
    def plot_semantic_network(
        self,
        network: nx.Graph,
        layout_algorithm: str = "spring",
        node_size_attr: str = "degree_centrality",
        edge_weight_attr: str = "weight",
        title: str = "Semantic Token Network",
        save_path: Optional[str] = None,
        interactive: Optional[bool] = None,
    ) -> Union[plt.Figure, go.Figure]:
        """
        Visualize semantic network of token relationships.
        
        Args:
            network: NetworkX graph object
            layout_algorithm: Layout algorithm ('spring', 'circular', 'kamada_kawai')
            node_size_attr: Node attribute for sizing
            edge_weight_attr: Edge attribute for line thickness
            title: Title for the plot
            save_path: Path to save the figure
            interactive: Whether to create interactive plot
            
        Returns:
            Matplotlib figure or Plotly figure object
        """
        use_interactive = interactive if interactive is not None else self.interactive
        
        if use_interactive:
            return self._create_interactive_network(
                network, layout_algorithm, node_size_attr, edge_weight_attr, title, save_path
            )
        else:
            return self._create_static_network(
                network, layout_algorithm, node_size_attr, edge_weight_attr, title, save_path
            )
    
    def _create_static_network(
        self,
        network: nx.Graph,
        layout_algorithm: str,
        node_size_attr: str,
        edge_weight_attr: str,
        title: str,
        save_path: Optional[str],
    ) -> plt.Figure:
        """Create static network visualization using matplotlib."""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Compute layout
        if layout_algorithm == "spring":
            pos = nx.spring_layout(network, k=1, iterations=50)
        elif layout_algorithm == "circular":
            pos = nx.circular_layout(network)
        elif layout_algorithm == "kamada_kawai":
            pos = nx.kamada_kawai_layout(network)
        else:
            pos = nx.spring_layout(network)
        
        # Get node sizes based on attribute
        if node_size_attr in next(iter(network.nodes(data=True)))[1]:
            node_sizes = [network.nodes[node].get(node_size_attr, 0.1) * 1000 for node in network.nodes()]
        else:
            node_sizes = [300] * len(network.nodes())
        
        # Get edge widths based on attribute
        if edge_weight_attr and network.edges():
            edge_weights = [network.edges[edge].get(edge_weight_attr, 0.1) * 5 for edge in network.edges()]
        else:
            edge_weights = [1.0] * len(network.edges())
        
        # Draw network
        nx.draw_networkx_edges(
            network, pos, width=edge_weights, alpha=0.6, edge_color='gray', ax=ax
        )
        
        nodes = nx.draw_networkx_nodes(
            network, pos, node_size=node_sizes, 
            node_color=range(len(network.nodes())),
            cmap=plt.cm.Set3, alpha=0.8, ax=ax
        )
        
        # Add labels for important nodes
        important_nodes = sorted(
            network.nodes(data=True), 
            key=lambda x: x[1].get(node_size_attr, 0), 
            reverse=True
        )[:10]
        
        labels = {node: data.get('token', f'N{node}') for node, data in important_nodes}
        nx.draw_networkx_labels(network, pos, labels, font_size=8, ax=ax)
        
        ax.set_title(title, fontsize=16, pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def _create_interactive_network(
        self,
        network: nx.Graph,
        layout_algorithm: str,
        node_size_attr: str,
        edge_weight_attr: str,
        title: str,
        save_path: Optional[str],
    ) -> go.Figure:
        """Create interactive network visualization using plotly."""
        # Compute layout
        if layout_algorithm == "spring":
            pos = nx.spring_layout(network, k=1, iterations=50)
        elif layout_algorithm == "circular":
            pos = nx.circular_layout(network)
        elif layout_algorithm == "kamada_kawai":
            pos = nx.kamada_kawai_layout(network)
        else:
            pos = nx.spring_layout(network)
        
        # Extract node and edge information
        node_x = [pos[node][0] for node in network.nodes()]
        node_y = [pos[node][1] for node in network.nodes()]
        node_text = [network.nodes[node].get('token', f'Node {node}') for node in network.nodes()]
        node_sizes = [network.nodes[node].get(node_size_attr, 0.1) * 50 for node in network.nodes()]
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in network.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=[f"Token: {text}<br>Size: {size:.3f}" for text, size in zip(node_text, node_sizes)],
            marker=dict(
                size=node_sizes,
                color=list(range(len(network.nodes()))),
                colorscale='Set3',
                showscale=False,
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=dict(text=title, x=0.5, font=dict(size=16)),
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Hover over nodes for details",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           width=900,
                           height=700,
                       ))
        
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, width=900, height=700)
        
        return fig
    
    def plot_correlation_evolution(
        self,
        correlation_history: List[Dict[str, Any]],
        title: str = "Correlation Evolution Over Time",
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Plot evolution of correlations over time/layers.
        
        Args:
            correlation_history: List of correlation analysis results
            title: Title for the plot
            save_path: Path to save the figure
            
        Returns:
            Plotly figure object
        """
        # Extract correlation statistics over time
        time_points = list(range(len(correlation_history)))
        metrics = ["mean_correlation", "max_correlation", "correlation_entropy", "significant_correlations"]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Mean Correlation", "Max Correlation", "Correlation Entropy", "Significant Correlations"),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, metric in enumerate(metrics):
            values = [hist["correlation_statistics"].get(metric, 0) for hist in correlation_history]
            
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=values,
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=6),
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            height=600,
            showlegend=False,
        )
        
        fig.update_xaxes(title_text="Time/Layer", row=2, col=1)
        fig.update_xaxes(title_text="Time/Layer", row=2, col=2)
        
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, width=1000, height=600)
        
        return fig
    
    def plot_quantum_state_evolution(
        self,
        quantum_states: List[torch.Tensor],
        title: str = "Quantum State Evolution",
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Visualize evolution of quantum states across layers.
        
        Args:
            quantum_states: List of quantum state tensors
            title: Title for the plot
            save_path: Path to save the figure
            
        Returns:
            Plotly figure object
        """
        # Compute quantum measures for each state
        measures = []
        layer_indices = []
        
        for i, state in enumerate(quantum_states):
            if state is not None:
                state_np = state.detach().cpu().numpy()
                
                # Compute various quantum measures
                if state_np.ndim == 4:  # (B, L, Q, 2)
                    batch_size, seq_len, num_qubits, _ = state_np.shape
                    
                    # Average over batch and sequence
                    avg_state = np.mean(state_np, axis=(0, 1))  # (Q, 2)
                    
                    # Compute measures
                    amplitudes = np.sqrt(avg_state[:, 0]**2 + avg_state[:, 1]**2)
                    phases = np.arctan2(avg_state[:, 1], avg_state[:, 0])
                    
                    coherence = np.std(phases)
                    entanglement = np.mean(amplitudes)
                    purity = np.sum(amplitudes**4)
                    
                    measures.append({
                        "layer": i,
                        "coherence": coherence,
                        "entanglement": entanglement,
                        "purity": purity,
                        "amplitude_variance": np.var(amplitudes),
                    })
                    layer_indices.append(i)
        
        if not measures:
            # Create empty figure if no valid quantum states
            fig = go.Figure()
            fig.update_layout(title="No valid quantum states found")
            return fig
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Quantum Coherence", "Entanglement Measure", "State Purity", "Amplitude Variance"),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        metrics = ["coherence", "entanglement", "purity", "amplitude_variance"]
        colors = px.colors.qualitative.Set1
        
        for i, metric in enumerate(metrics):
            values = [m[metric] for m in measures]
            
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Scatter(
                    x=layer_indices,
                    y=values,
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=8),
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            height=600,
            showlegend=False,
        )
        
        fig.update_xaxes(title_text="Layer", row=2, col=1)
        fig.update_xaxes(title_text="Layer", row=2, col=2)
        
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, width=1000, height=600)
        
        return fig
    
    def plot_embedding_space(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
        token_labels: Optional[List[str]] = None,
        method: str = "tsne",
        title: str = "Token Embedding Space",
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Visualize token embeddings in reduced dimensional space.
        
        Args:
            embeddings: Token embeddings (n_tokens, embedding_dim)
            token_labels: Labels for tokens
            method: Dimensionality reduction method ('tsne', 'pca')
            title: Title for the plot
            save_path: Path to save the figure
            
        Returns:
            Plotly figure object
        """
        # Convert to numpy if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.detach().cpu().numpy()
        else:
            embeddings_np = embeddings
        
        # Flatten if needed (remove batch/sequence dimensions)
        if embeddings_np.ndim > 2:
            embeddings_np = embeddings_np.reshape(-1, embeddings_np.shape[-1])
        
        # Apply dimensionality reduction
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_np)-1))
            reduced_embeddings = reducer.fit_transform(embeddings_np)
        elif method == "pca":
            reducer = PCA(n_components=2, random_state=42)
            reduced_embeddings = reducer.fit_transform(embeddings_np)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Create scatter plot
        if token_labels is not None and len(token_labels) == len(reduced_embeddings):
            hover_text = token_labels
            text_labels = [label[:10] + "..." if len(label) > 10 else label for label in token_labels]
        else:
            hover_text = [f"Token {i}" for i in range(len(reduced_embeddings))]
            text_labels = None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            mode='markers+text' if text_labels else 'markers',
            text=text_labels,
            textposition="top center",
            hovertext=hover_text,
            hoverinfo='text',
            marker=dict(
                size=8,
                color=list(range(len(reduced_embeddings))),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Token Index"),
                line=dict(width=1, color='white')
            ),
            textfont=dict(size=8)
        ))
        
        method_name = method.upper()
        fig.update_layout(
            title=dict(text=f"{title} ({method_name})", x=0.5, font=dict(size=16)),
            xaxis_title=f"{method_name} Component 1",
            yaxis_title=f"{method_name} Component 2",
            width=800,
            height=600,
            hovermode='closest'
        )
        
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, width=800, height=600)
        
        return fig
    
    def create_analysis_dashboard(
        self,
        analysis_results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> go.Figure:
        """
        Create comprehensive analysis dashboard.
        
        Args:
            analysis_results: Dictionary containing all analysis results
            save_path: Path to save the dashboard
            
        Returns:
            Plotly figure object with dashboard
        """
        # Extract data from analysis results
        correlation_data = analysis_results.get("correlation_history", [])
        entanglement_data = analysis_results.get("entanglement_history", [])
        network_data = analysis_results.get("semantic_networks", [])
        
        # Create subplot structure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Correlation Statistics", "Entanglement Evolution",
                "Network Properties", "Quantum Measures",
                "Central Tokens", "Analysis Summary"
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.12,
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # Add traces for each subplot
        if correlation_data:
            self._add_correlation_traces(fig, correlation_data, row=1, col=1)
        
        if entanglement_data:
            self._add_entanglement_traces(fig, entanglement_data, row=1, col=2)
        
        if network_data:
            self._add_network_traces(fig, network_data, row=2, col=1)
            self._add_centrality_traces(fig, network_data, row=3, col=1)
        
        # Add summary table
        if analysis_results:
            self._add_summary_table(fig, analysis_results, row=3, col=2)
        
        fig.update_layout(
            title=dict(text="Entanglement-Enhanced NLP Analysis Dashboard", x=0.5, font=dict(size=20)),
            height=1200,
            showlegend=True,
        )
        
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, width=1400, height=1200)
        
        return fig
    
    def _add_correlation_traces(self, fig: go.Figure, data: List[Dict], row: int, col: int) -> None:
        """Add correlation statistics traces to subplot."""
        if not data:
            return
        
        time_points = list(range(len(data)))
        mean_corrs = [d["correlation_statistics"].get("mean_correlation", 0) for d in data]
        max_corrs = [d["correlation_statistics"].get("max_correlation", 0) for d in data]
        
        fig.add_trace(
            go.Scatter(x=time_points, y=mean_corrs, name="Mean Correlation", 
                      line=dict(color='blue')), row=row, col=col
        )
        fig.add_trace(
            go.Scatter(x=time_points, y=max_corrs, name="Max Correlation", 
                      line=dict(color='red')), row=row, col=col
        )
    
    def _add_entanglement_traces(self, fig: go.Figure, data: List[Dict], row: int, col: int) -> None:
        """Add entanglement evolution traces to subplot."""
        if not data:
            return
        
        time_points = list(range(len(data)))
        entanglement_values = []
        
        for d in data:
            layer_analysis = d.get("layer_analysis", [])
            if layer_analysis:
                avg_entanglement = np.mean([layer.get("mean_entanglement", 0) for layer in layer_analysis])
                entanglement_values.append(avg_entanglement)
            else:
                entanglement_values.append(0)
        
        fig.add_trace(
            go.Scatter(x=time_points, y=entanglement_values, name="Avg Entanglement",
                      line=dict(color='purple')), row=row, col=col
        )
    
    def _add_network_traces(self, fig: go.Figure, data: List[Dict], row: int, col: int) -> None:
        """Add network properties traces to subplot."""
        if not data:
            return
        
        properties = ["num_nodes", "num_edges", "density"]
        colors = ['green', 'orange', 'brown']
        
        for i, prop in enumerate(properties):
            values = [d["network_properties"].get(prop, 0) for d in data]
            time_points = list(range(len(values)))
            
            fig.add_trace(
                go.Bar(x=time_points, y=values, name=prop.replace('_', ' ').title(),
                      marker_color=colors[i]), row=row, col=col
            )
    
    def _add_centrality_traces(self, fig: go.Figure, data: List[Dict], row: int, col: int) -> None:
        """Add centrality measures traces to subplot."""
        if not data or not data[0].get("central_tokens"):
            return
        
        # Use latest network data
        central_tokens = data[-1]["central_tokens"][:10]  # Top 10
        tokens = [token["token"][:10] for token in central_tokens]  # Truncate long tokens
        centralities = [token["combined_centrality"] for token in central_tokens]
        
        fig.add_trace(
            go.Bar(x=tokens, y=centralities, name="Token Centrality",
                  marker_color='teal'), row=row, col=col
        )
    
    def _add_summary_table(self, fig: go.Figure, data: Dict[str, Any], row: int, col: int) -> None:
        """Add analysis summary table to subplot."""
        # Create summary statistics
        summary_data = []
        
        if data.get("correlation_history"):
            corr_count = len(data["correlation_history"])
            summary_data.append(["Correlation Analyses", str(corr_count)])
        
        if data.get("entanglement_history"):
            ent_count = len(data["entanglement_history"])
            summary_data.append(["Entanglement Analyses", str(ent_count)])
        
        if data.get("semantic_networks"):
            net_count = len(data["semantic_networks"])
            summary_data.append(["Semantic Networks", str(net_count)])
        
        if summary_data:
            fig.add_trace(
                go.Table(
                    header=dict(values=["Metric", "Value"], fill_color='lightblue'),
                    cells=dict(values=list(zip(*summary_data)), fill_color='white')
                ), row=row, col=col
            )
    
    def save_all_plots(
        self,
        analysis_results: Dict[str, Any],
        output_dir: str,
        formats: List[str] = ["png", "html"],
    ) -> Dict[str, str]:
        """
        Save all analysis plots to specified directory.
        
        Args:
            analysis_results: Complete analysis results
            output_dir: Output directory path
            formats: List of formats to save ('png', 'pdf', 'html', 'svg')
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # Generate and save various plots
        plot_configs = [
            ("correlation_heatmap", self.plot_entanglement_heatmap),
            ("semantic_network", self.plot_semantic_network),
            ("correlation_evolution", self.plot_correlation_evolution),
            ("analysis_dashboard", self.create_analysis_dashboard),
        ]
        
        for plot_name, plot_function in plot_configs:
            try:
                for fmt in formats:
                    filename = f"{plot_name}.{fmt}"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Call appropriate plot function based on available data
                    if plot_name == "correlation_heatmap" and analysis_results.get("correlation_history"):
                        # Use latest correlation matrix
                        latest_corr = analysis_results["correlation_history"][-1]
                        if "correlation_matrix" in latest_corr:
                            fig = plot_function(
                                latest_corr["correlation_matrix"],
                                save_path=filepath,
                                interactive=fmt == "html"
                            )
                    
                    elif plot_name == "correlation_evolution" and analysis_results.get("correlation_history"):
                        fig = plot_function(
                            analysis_results["correlation_history"],
                            save_path=filepath
                        )
                    
                    elif plot_name == "analysis_dashboard":
                        fig = plot_function(analysis_results, save_path=filepath)
                    
                    saved_files[f"{plot_name}_{fmt}"] = filepath
                    
            except Exception as e:
                print(f"Warning: Could not generate {plot_name}: {e}")
        
        return saved_files
