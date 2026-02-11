#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.settings import Settings, load_settings
from src.core.query_engine.hybrid_search import HybridSearch
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.evaluation.eval_runner import EvalRunner


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on golden test set.")
    parser.add_argument(
        "--test-set",
        type=str,
        default="tests/fixtures/golden_test_set.json",
        help="Path to the golden test set JSON file.",
    )
    args = parser.parse_args()

    console = Console()
    console.print("[bold blue]Starting Evaluation...[/bold blue]")

    # 1. Initialize Settings
    try:
        settings = load_settings()  # Loads from environment/yaml
    except Exception as e:
        console.print(f"[bold red]Failed to load settings:[/bold red] {e}")
        return

    # 2. Initialize Components
    try:
        console.print("Initializing HybridSearch...")
        hybrid_search = HybridSearch(settings)
        
        console.print("Initializing Evaluator...")
        evaluator = EvaluatorFactory.create(settings)
    except Exception as e:
        console.print(f"[bold red]Failed to initialize components:[/bold red] {e}")
        return

    # 3. Run Evaluation
    runner = EvalRunner(settings, hybrid_search, evaluator)
    test_set_path = args.test_set
    
    if not Path(test_set_path).exists():
        console.print(f"[bold red]Test set not found:[/bold red] {test_set_path}")
        return

    console.print(f"Running evaluation on [cyan]{test_set_path}[/cyan]...")
    try:
        report = runner.run(test_set_path)
    except Exception as e:
        console.print(f"[bold red]Evaluation failed:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Print Results
    console.print("\n[bold green]Evaluation Completed![/bold green]")
    
    # Aggregate Metrics Table
    table = Table(title="Aggregate Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    for k, v in report.aggregate_metrics.items():
        table.add_row(k, f"{v:.4f}")
    
    console.print(table)

    # Detailed Results (Summary)
    console.print(f"\nTotal Cases: {report.total_cases}")
    
    # Optional: Print failed cases or details if needed
    # for res in report.case_results:
    #     console.print(f"Query: {res.query} -> {res.metrics}")

if __name__ == "__main__":
    main()
