"""
Command Line Interface for Entanglement Enhanced NLP.

This module provides a comprehensive CLI tool for running entanglement analysis
on text files, datasets, and performing various quantum-inspired NLP tasks.
"""

import click
import torch
import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from .core.entangled_embedding import EntangledEmbedding
from .core.quantum_contextualizer import QuantumContextualizer
from .transformers.entangled_transformer import EntangledTransformer, EntangledTransformerConfig
from .analysis.correlation_analyzer import CorrelationAnalyzer
from .visualization.entanglement_visualizer import EntanglementVisualizer
from .utils.quantum_simulator import QuantumSimulator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.1")
def main():
    """
    Entanglement Enhanced NLP CLI Tool
    
    A quantum-inspired Natural Language Processing framework that integrates
    quantum entanglement concepts for enhanced semantic understanding.
    
    Author: Krishna Bajpai (bajpaikrishna715@gmail.com)
    """
    pass


@main.command()
@click.option("--input", "-i", required=True, help="Input text file or dataset path")
@click.option("--output", "-o", required=True, help="Output file path for results")
@click.option("--model-config", "-c", help="Model configuration file (YAML/JSON)")
@click.option("--vocab-size", default=10000, help="Vocabulary size for embeddings")
@click.option("--embedding-dim", default=768, help="Embedding dimension")
@click.option("--entanglement-depth", default=3, help="Entanglement depth layers")
@click.option("--correlation-strength", default=0.8, help="Correlation strength (0-1)")
@click.option("--decoherence-rate", default=0.1, help="Decoherence rate (0-1)")
@click.option("--max-length", default=512, help="Maximum sequence length")
@click.option("--batch-size", default=8, help="Batch size for processing")
@click.option("--device", default="auto", help="Device to use (cpu/cuda/auto)")
@click.option("--visualize", is_flag=True, help="Generate visualization plots")
@click.option("--export-format", default="json", help="Export format (json/csv/pickle)")
def analyze(
    input: str,
    output: str,
    model_config: Optional[str],
    vocab_size: int,
    embedding_dim: int,
    entanglement_depth: int,
    correlation_strength: float,
    decoherence_rate: float,
    max_length: int,
    batch_size: int,
    device: str,
    visualize: bool,
    export_format: str,
):
    """
    Analyze text with quantum entanglement-enhanced embeddings.
    
    This command processes text files or datasets using quantum-inspired
    NLP models and provides comprehensive analysis of entanglement patterns,
    correlations, and semantic relationships.
    """
    click.echo("🌌 Starting Entanglement Enhanced NLP Analysis...")
    
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    click.echo(f"Using device: {device}")
    
    # Load configuration if provided
    config = {}
    if model_config:
        with open(model_config, 'r') as f:
            if model_config.endswith('.yaml') or model_config.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
    
    # Create model configuration
    model_config_dict = {
        "vocab_size": vocab_size,
        "hidden_size": embedding_dim,
        "entanglement_depth": entanglement_depth,
        "correlation_strength": correlation_strength,
        "decoherence_rate": decoherence_rate,
        "max_position_embeddings": max_length,
        **config
    }
    
    # Initialize components
    click.echo("🔬 Initializing quantum components...")
    
    entangled_embedding = EntangledEmbedding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        entanglement_depth=entanglement_depth,
        correlation_strength=correlation_strength,
        decoherence_rate=decoherence_rate,
        max_position_embeddings=max_length,
    ).to(device)
    
    quantum_contextualizer = QuantumContextualizer(
        hidden_dim=embedding_dim,
        num_qubits=8,
        decoherence_rate=decoherence_rate,
    ).to(device)
    
    analyzer = CorrelationAnalyzer()
    
    # Process input data
    click.echo("📖 Processing input data...")
    input_path = Path(input)
    
    if input_path.suffix == '.txt':
        # Process text file
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        results = _process_text(
            text, entangled_embedding, quantum_contextualizer, 
            analyzer, max_length, batch_size, device
        )
    
    elif input_path.suffix == '.csv':
        # Process CSV dataset
        df = pd.read_csv(input_path)
        results = _process_dataset(
            df, entangled_embedding, quantum_contextualizer,
            analyzer, max_length, batch_size, device
        )
    
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    # Generate visualizations if requested
    if visualize:
        click.echo("🎨 Generating visualizations...")
        visualizer = EntanglementVisualizer()
        
        output_dir = Path(output).parent / "visualizations"
        output_dir.mkdir(exist_ok=True)
        
        saved_plots = visualizer.save_all_plots(
            results, str(output_dir), formats=["png", "html"]
        )
        
        results["visualization_files"] = saved_plots
    
    # Export results
    click.echo(f"💾 Exporting results to {output}...")
    analyzer.export_analysis_results(output, format=export_format)
    
    # Save additional results
    if export_format == "json":
        with open(output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    click.echo("✅ Analysis complete!")
    
    # Print summary
    summary = analyzer.get_analysis_summary()
    click.echo("\n📊 Analysis Summary:")
    click.echo(f"  Total analyses: {summary['total_analyses']}")
    click.echo(f"  Networks analyzed: {summary['total_networks']}")
    if summary['average_correlations']:
        avg_corr = sum(summary['average_correlations']) / len(summary['average_correlations'])
        click.echo(f"  Average correlation: {avg_corr:.4f}")


@main.command()
@click.option("--dataset", "-d", required=True, help="Dataset file path (CSV/JSON)")
@click.option("--text-column", default="text", help="Name of text column in dataset")
@click.option("--output-dir", "-o", required=True, help="Output directory for results")
@click.option("--model-config", "-c", help="Model configuration file")
@click.option("--batch-size", default=16, help="Batch size for processing")
@click.option("--max-samples", default=1000, help="Maximum samples to process")
@click.option("--save-embeddings", is_flag=True, help="Save entangled embeddings")
@click.option("--analysis-mode", default="comprehensive", help="Analysis mode (fast/comprehensive/detailed)")
def process(
    dataset: str,
    text_column: str,
    output_dir: str,
    model_config: Optional[str],
    batch_size: int,
    max_samples: int,
    save_embeddings: bool,
    analysis_mode: str,
):
    """
    Process large datasets with quantum-enhanced NLP models.
    
    This command processes datasets in batch mode, providing scalable
    analysis of quantum entanglement patterns across large text corpora.
    """
    click.echo("🚀 Starting dataset processing...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    if dataset.endswith('.csv'):
        df = pd.read_csv(dataset)
    elif dataset.endswith('.json'):
        df = pd.read_json(dataset)
    else:
        raise ValueError("Unsupported dataset format")
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataset")
    
    # Limit samples if specified
    if max_samples > 0:
        df = df.head(max_samples)
    
    click.echo(f"Processing {len(df)} samples...")
    
    # Load configuration
    config = {}
    if model_config:
        with open(model_config, 'r') as f:
            if model_config.endswith('.yaml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
    
    # Initialize model
    transformer_config = EntangledTransformerConfig(**config)
    model = EntangledTransformer(transformer_config)
    
    # Process in batches
    all_results = []
    analyzer = CorrelationAnalyzer(analysis_mode=analysis_mode)
    
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_df = df.iloc[i:i+batch_size]
        batch_texts = batch_df[text_column].tolist()
        
        # Process batch
        batch_results = _process_text_batch(batch_texts, model, analyzer)
        all_results.extend(batch_results)
    
    # Save results
    results_file = output_path / "processing_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Generate summary report
    _generate_processing_report(all_results, output_path)
    
    click.echo(f"✅ Processing complete! Results saved to {output_dir}")


@main.command()
@click.option("--input", "-i", required=True, help="Analysis results file (JSON)")
@click.option("--output-dir", "-o", required=True, help="Output directory for visualizations")
@click.option("--plot-types", default="all", help="Plot types (all/heatmap/network/evolution)")
@click.option("--format", default="html", help="Output format (html/png/pdf/svg)")
@click.option("--interactive", is_flag=True, help="Generate interactive plots")
@click.option("--style", default="default", help="Plot style (default/dark/scientific)")
def visualize(
    input: str,
    output_dir: str,
    plot_types: str,
    format: str,
    interactive: bool,
    style: str,
):
    """
    Generate visualizations from analysis results.
    
    Create comprehensive visualizations of quantum entanglement patterns,
    semantic networks, and correlation evolution from analysis results.
    """
    click.echo("🎨 Generating quantum entanglement visualizations...")
    
    # Load analysis results
    with open(input, 'r') as f:
        results = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    visualizer = EntanglementVisualizer(
        style=style,
        interactive=interactive,
    )
    
    # Generate requested plots
    plot_list = plot_types.split(',') if plot_types != "all" else [
        "heatmap", "network", "evolution", "dashboard"
    ]
    
    saved_files = {}
    
    for plot_type in plot_list:
        click.echo(f"Generating {plot_type} visualization...")
        
        try:
            if plot_type == "heatmap" and "correlation_history" in results:
                latest_corr = results["correlation_history"][-1]
                if "correlation_matrix" in latest_corr:
                    fig = visualizer.plot_entanglement_heatmap(
                        latest_corr["correlation_matrix"],
                        title="Quantum Entanglement Correlation Matrix",
                        save_path=str(output_path / f"entanglement_heatmap.{format}"),
                        interactive=interactive,
                    )
                    saved_files["heatmap"] = str(output_path / f"entanglement_heatmap.{format}")
            
            elif plot_type == "evolution" and "correlation_history" in results:
                fig = visualizer.plot_correlation_evolution(
                    results["correlation_history"],
                    save_path=str(output_path / f"correlation_evolution.{format}"),
                )
                saved_files["evolution"] = str(output_path / f"correlation_evolution.{format}")
            
            elif plot_type == "dashboard":
                fig = visualizer.create_analysis_dashboard(
                    results,
                    save_path=str(output_path / f"analysis_dashboard.{format}"),
                )
                saved_files["dashboard"] = str(output_path / f"analysis_dashboard.{format}")
            
        except Exception as e:
            click.echo(f"Warning: Could not generate {plot_type}: {e}")
    
    # Save visualization summary
    summary_file = output_path / "visualization_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "generated_plots": saved_files,
            "plot_types": plot_list,
            "format": format,
            "style": style,
            "interactive": interactive,
        }, f, indent=2)
    
    click.echo(f"✅ Visualizations saved to {output_dir}")
    for plot_name, file_path in saved_files.items():
        click.echo(f"  {plot_name}: {file_path}")


@main.command()
@click.option("--text", "-t", help="Input text to analyze")
@click.option("--file", "-f", help="Input text file")
@click.option("--interactive", is_flag=True, help="Interactive mode")
def demo(text: Optional[str], file: Optional[str], interactive: bool):
    """
    Run a quick demo of quantum entanglement analysis.
    
    Demonstrates the capabilities of the entanglement-enhanced NLP framework
    with a simple example or interactive session.
    """
    click.echo("🌟 Quantum Entanglement NLP Demo")
    click.echo("=" * 40)
    
    # Get input text
    if file:
        with open(file, 'r') as f:
            input_text = f.read()
    elif text:
        input_text = text
    else:
        input_text = "The quantum entanglement between semantic tokens enables deeper understanding of natural language through non-local correlations."
    
    click.echo(f"📝 Analyzing text: {input_text[:100]}...")
    
    # Initialize small model for demo
    embedding_dim = 256
    vocab_size = 1000
    
    entangled_embedding = EntangledEmbedding(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        entanglement_depth=2,
        correlation_strength=0.7,
    )
    
    quantum_contextualizer = QuantumContextualizer(
        hidden_dim=embedding_dim,
        num_qubits=6,
        num_layers=2,
    )
    
    # Simulate processing (simple tokenization)
    words = input_text.split()[:20]  # Limit for demo
    token_ids = torch.randint(0, vocab_size, (1, len(words)))
    
    click.echo("🔬 Computing quantum entangled embeddings...")
    with torch.no_grad():
        embeddings, correlations = entangled_embedding(token_ids, return_correlations=True)
        
        click.echo("⚛️  Applying quantum contextualizer...")
        enhanced_embeddings, quantum_states = quantum_contextualizer(
            embeddings, return_quantum_states=True
        )
        
        # Get statistics
        embed_stats = entangled_embedding.get_entanglement_statistics(token_ids)
        quantum_stats = quantum_contextualizer.get_quantum_statistics(embeddings)
    
    # Display results
    click.echo("\n📊 Entanglement Statistics:")
    for key, value in embed_stats.items():
        click.echo(f"  {key}: {value:.4f}")
    
    click.echo("\n⚛️  Quantum Statistics:")
    for key, value in quantum_stats.items():
        click.echo(f"  {key}: {value:.4f}")
    
    if correlations is not None:
        max_correlation = torch.max(correlations).item()
        avg_correlation = torch.mean(correlations).item()
        click.echo(f"\n🔗 Correlation Analysis:")
        click.echo(f"  Maximum correlation: {max_correlation:.4f}")
        click.echo(f"  Average correlation: {avg_correlation:.4f}")
    
    click.echo("\n✨ Demo complete! This is just a glimpse of the full framework's capabilities.")
    click.echo("💡 For comprehensive analysis, use the 'analyze' command with real text data.")


def _process_text(
    text: str,
    entangled_embedding: EntangledEmbedding,
    quantum_contextualizer: QuantumContextualizer,
    analyzer: CorrelationAnalyzer,
    max_length: int,
    batch_size: int,
    device: str,
) -> Dict[str, Any]:
    """Process a single text with quantum analysis."""
    # Simple tokenization (in real implementation, use proper tokenizer)
    words = text.split()
    
    results = {
        "text_length": len(text),
        "num_words": len(words),
        "correlation_analysis": [],
        "entanglement_analysis": [],
        "quantum_statistics": {},
    }
    
    # Process in chunks
    for i in range(0, len(words), max_length):
        chunk_words = words[i:i+max_length]
        
        # Simulate token IDs (in real implementation, use proper tokenizer)
        token_ids = torch.randint(0, entangled_embedding.vocab_size, (1, len(chunk_words))).to(device)
        
        with torch.no_grad():
            # Get entangled embeddings
            embeddings, correlations = entangled_embedding(token_ids, return_correlations=True)
            
            # Apply quantum contextualizer
            enhanced_embeddings, quantum_states = quantum_contextualizer(
                embeddings, return_quantum_states=True
            )
            
            # Analyze correlations
            if correlations is not None:
                corr_analysis = analyzer.analyze_token_correlations(
                    embeddings, token_ids
                )
                results["correlation_analysis"].append(corr_analysis)
            
            # Analyze quantum entanglement
            if quantum_states:
                ent_analysis = analyzer.analyze_quantum_entanglement(
                    quantum_states, [correlations] if correlations is not None else [],
                    token_ids
                )
                results["entanglement_analysis"].append(ent_analysis)
    
    return results


def _process_dataset(
    df: pd.DataFrame,
    entangled_embedding: EntangledEmbedding,
    quantum_contextualizer: QuantumContextualizer,
    analyzer: CorrelationAnalyzer,
    max_length: int,
    batch_size: int,
    device: str,
) -> Dict[str, Any]:
    """Process a dataset with quantum analysis."""
    # This is a simplified implementation
    # In practice, you'd implement proper batching and text column detection
    
    results = {
        "dataset_info": {
            "num_samples": len(df),
            "columns": df.columns.tolist(),
        },
        "analysis_results": [],
    }
    
    # For demo, process first text column
    text_columns = [col for col in df.columns if df[col].dtype == 'object']
    if text_columns:
        text_col = text_columns[0]
        sample_texts = df[text_col].head(10).tolist()  # Limit for demo
        
        for text in sample_texts:
            if isinstance(text, str):
                text_results = _process_text(
                    text, entangled_embedding, quantum_contextualizer,
                    analyzer, max_length, batch_size, device
                )
                results["analysis_results"].append(text_results)
    
    return results


def _process_text_batch(
    texts: List[str],
    model: EntangledTransformer,
    analyzer: CorrelationAnalyzer,
) -> List[Dict[str, Any]]:
    """Process a batch of texts."""
    results = []
    
    for text in texts:
        # Simplified processing
        words = text.split()[:100]  # Limit for demo
        token_ids = torch.randint(0, model.config.vocab_size, (1, len(words)))
        
        with torch.no_grad():
            outputs = model(
                input_ids=token_ids,
                output_attentions=True,
                output_quantum_states=True,
            )
            
            # Analyze results
            text_results = {
                "text": text[:200],  # Truncate for storage
                "num_tokens": len(words),
                "model_output_shape": list(outputs.last_hidden_state.shape),
            }
            
            results.append(text_results)
    
    return results


def _generate_processing_report(results: List[Dict], output_path: Path) -> None:
    """Generate processing summary report."""
    report = {
        "total_samples": len(results),
        "average_tokens": sum(r.get("num_tokens", 0) for r in results) / len(results) if results else 0,
        "processing_summary": {
            "successful": len([r for r in results if "model_output_shape" in r]),
            "failed": len([r for r in results if "model_output_shape" not in r]),
        }
    }
    
    report_file = output_path / "processing_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
