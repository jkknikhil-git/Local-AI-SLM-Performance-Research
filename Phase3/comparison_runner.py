"""
Phase3/comparison_runner.py

Phase 3 — Multi-Model Comparison Study

Runs all models in the comparison set against the standardized prompt bank,
collects performance metrics, saves per-model JSON results, and prints
a cross-model summary table.

Run one model at a time (results accumulate in Phase3/results/):
    python Phase3/comparison_runner.py --model llama3.1:8b
    python Phase3/comparison_runner.py --model gemma4:e4b
    python Phase3/comparison_runner.py --model qwen3.5:4b
    python Phase3/comparison_runner.py --model qwen3.5:9b

Run all models in one go (takes a long time on CPU):
    python Phase3/comparison_runner.py --all

Generate the comparison report from saved results:
    python Phase3/comparison_runner.py --report
"""

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Phase1"))

from ollama_client import OllamaClient, BenchmarkMetrics
from Phase3.test_prompts import COMPARISON_PROMPTS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"
REPORT_DIR  = Path(__file__).parent / "report"

COMPARISON_MODELS = [
    "llama3.1:8b",
    "gemma4:e4b",
    "qwen3.5:4b",
]

HARDWARE_META = {
    "cpu": "11th Gen Intel Core i5-11320H @ 3.20GHz",
    "ram_gb": 16,
    "gpu": "Intel Iris Xe (integrated, 128MB dedicated)",
    "os": "Windows 11 64-bit",
    "inference_backend": "Ollama (CPU-only, no GPU offload)",
}

console = Console()


# ---------------------------------------------------------------------------
# Run single model
# ---------------------------------------------------------------------------

def run_model(model: str, runs_per_prompt: int = 2, num_ctx: int = 4096) -> dict:
    """
    Benchmark a single model across all comparison prompts.
    Returns a result dict ready for JSON serialisation.
    """
    client = OllamaClient()

    # Preflight
    console.print(f"\n[bold]Checking model '[cyan]{model}[/cyan]'...[/bold] ", end="")
    local = client.list_local_models()
    if not any(model in m for m in local):
        console.print(f"[red]✗ Not pulled locally.[/red]")
        console.print(f"  Run:  [cyan]ollama pull {model}[/cyan]")
        return {}
    console.print("[green]✓ Available[/green]")

    all_metrics: list[BenchmarkMetrics] = []
    total = len(COMPARISON_PROMPTS) * (1 + runs_per_prompt)  # 1 cold + N warm

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Running {model}...", total=total)

        for prompt in COMPARISON_PROMPTS:

            # Cold run (discarded from averages but recorded)
            progress.update(task, description=f"[yellow]COLD[/yellow]  {prompt.id}")
            cold = client.benchmark_generate(
               model=model,
               prompt=prompt.text,
               prompt_id=prompt.id,
               run_index=0,
               run_type="cold",
               extra_options={"think": False} if "qwen" in model.lower() else {},
            )
            all_metrics.append(cold)
            progress.advance(task)

            # Warm runs
            for i in range(runs_per_prompt):
                progress.update(
                    task,
                    description=f"[green]WARM {i+1}/{runs_per_prompt}[/green]  {prompt.id}",
                )
                warm = client.benchmark_generate(
                    model=model,
                    prompt=prompt.text,
                    prompt_id=prompt.id,
                    run_index=i + 1,
                    run_type="warm",
                    extra_options={"think": False} if "qwen" in model.lower() else {},
                )
                all_metrics.append(warm)
                progress.advance(task)

    # Aggregate warm-run stats per category
    warm = [m for m in all_metrics if m.run_type == "warm"]

    category_stats = {}
    categories = sorted(set(p.category for p in COMPARISON_PROMPTS))
    for cat in categories:
        cat_runs = [m for m in warm if m.prompt_id.startswith(cat[:4])]
        if cat_runs:
            category_stats[cat] = {
                "tps_mean":      round(mean(r.tokens_per_second for r in cat_runs), 2),
                "ttft_mean_s":   round(mean(r.time_to_first_token_s for r in cat_runs), 3),
                "latency_mean_s":round(mean(r.total_latency_s for r in cat_runs), 3),
                "tokens_mean":   round(mean(r.completion_tokens for r in cat_runs), 1),
                "cpu_mean_pct":  round(mean(r.cpu_percent_avg for r in cat_runs), 1),
                "ram_delta_mean_mb": round(mean(r.ram_delta_mb for r in cat_runs), 1),
            }

    overall = {
        "tps_mean":      round(mean(r.tokens_per_second for r in warm), 2),
        "ttft_mean_s":   round(mean(r.time_to_first_token_s for r in warm), 3),
        "latency_mean_s":round(mean(r.total_latency_s for r in warm), 3),
        "tokens_mean":   round(mean(r.completion_tokens for r in warm), 1),
        "cpu_mean_pct":  round(mean(r.cpu_percent_avg for r in warm), 1),
        "ram_delta_mean_mb": round(mean(r.ram_delta_mb for r in warm), 1),
    }

    return {
        "model": model,
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": HARDWARE_META,
        "runs_per_prompt": runs_per_prompt,
        "prompt_count": len(COMPARISON_PROMPTS),
        "overall": overall,
        "by_category": category_stats,
        "runs": [m.to_dict() for m in all_metrics],
    }


def save_model_result(model: str, result: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    tag = model.replace(":", "_").replace("/", "_")
    out_path = RESULTS_DIR / f"{tag}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    console.print(f"[bold green]✓[/bold green] Saved → [cyan]{out_path}[/cyan]")
    return out_path


# ---------------------------------------------------------------------------
# Load saved results
# ---------------------------------------------------------------------------

def load_all_results() -> dict[str, dict]:
    """Load all per-model JSON files from RESULTS_DIR."""
    results = {}
    if not RESULTS_DIR.exists():
        return results
    for f in RESULTS_DIR.glob("*.json"):
        # Skip temperature experiment files from Phase2 if accidentally placed here
        if "temp_exp" in f.name:
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if "model" in data and "overall" in data:
                results[data["model"]] = data
        except Exception:
            pass
    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_overall_comparison(results: dict[str, dict]):
    """Print top-level cross-model comparison table."""
    console.print()
    console.rule("[bold cyan]Phase 3 — Multi-Model Comparison[/bold cyan]")
    console.print()

    table = Table(
        title="Overall Performance (warm runs avg · CPU-only · i5-11320H · 16GB RAM)",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold magenta",
    )

    table.add_column("Model",           style="cyan", no_wrap=True)
    table.add_column("TPS",             justify="right", style="green")
    table.add_column("TTFT (s)",        justify="right")
    table.add_column("Latency (s)",     justify="right")
    table.add_column("Avg Tokens",      justify="right")
    table.add_column("CPU %",           justify="right", style="red")
    table.add_column("RAM Δ (MB)",      justify="right", style="yellow")

    for model, data in sorted(results.items()):
        o = data["overall"]
        table.add_row(
            model,
            str(o["tps_mean"]),
            str(o["ttft_mean_s"]),
            str(o["latency_mean_s"]),
            str(o["tokens_mean"]),
            f"{o['cpu_mean_pct']}%",
            str(o["ram_delta_mean_mb"]),
        )

    console.print(table)


def print_category_comparison(results: dict[str, dict], category: str):
    """Print per-category comparison for one category."""
    console.print()
    table = Table(
        title=f"Category: {category}",
        box=box.SIMPLE_HEAVY,
        header_style="bold",
    )
    table.add_column("Model",       style="cyan")
    table.add_column("TPS",         justify="right", style="green")
    table.add_column("TTFT (s)",    justify="right")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Avg Tokens",  justify="right")

    for model, data in sorted(results.items()):
        cat_stats = data.get("by_category", {}).get(category)
        if not cat_stats:
            continue
        table.add_row(
            model,
            str(cat_stats["tps_mean"]),
            str(cat_stats["ttft_mean_s"]),
            str(cat_stats["latency_mean_s"]),
            str(cat_stats["tokens_mean"]),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def generate_report(results: dict[str, dict]):
    """Generate a markdown technical report from accumulated results."""
    if not results:
        console.print("[red]No results found in Phase3/results/. Run some models first.[/red]")
        return

    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / "technical_report.md"

    categories = sorted(set(
        cat
        for data in results.values()
        for cat in data.get("by_category", {}).keys()
    ))

    lines = []
    lines.append("# Phase 3 — Multi-Model Comparison: Technical Report\n")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  \n")
    lines.append(f"**Models compared:** {', '.join(sorted(results.keys()))}  \n")
    lines.append(f"**Prompts:** {list(results.values())[0].get('prompt_count', 30)} (30 across 6 categories)  \n")
    lines.append(f"**Runs per prompt:** {list(results.values())[0].get('runs_per_prompt', 2)} warm  \n")
    lines.append("\n---\n")

    # Hardware
    lines.append("## Hardware\n")
    hw = list(results.values())[0]["hardware"]
    lines.append(f"| Component | Spec |\n|-----------|------|\n")
    lines.append(f"| CPU | {hw['cpu']} |\n")
    lines.append(f"| RAM | {hw['ram_gb']}GB |\n")
    lines.append(f"| GPU | {hw['gpu']} |\n")
    lines.append(f"| OS | {hw['os']} |\n")
    lines.append(f"| Backend | {hw['inference_backend']} |\n")
    lines.append("\n---\n")

    # Overall comparison table
    lines.append("## Overall Performance\n")
    lines.append("| Model | TPS | TTFT (s) | Latency (s) | Avg Tokens | CPU % | RAM Δ (MB) |\n")
    lines.append("|-------|-----|----------|-------------|------------|-------|------------|\n")
    for model in sorted(results.keys()):
        o = results[model]["overall"]
        lines.append(
            f"| {model} | **{o['tps_mean']}** | {o['ttft_mean_s']} | "
            f"{o['latency_mean_s']} | {o['tokens_mean']} | "
            f"{o['cpu_mean_pct']}% | {o['ram_delta_mean_mb']} |\n"
        )
    lines.append("\n")

    # Per-category tables
    lines.append("## Results by Category\n")
    for cat in categories:
        lines.append(f"### {cat.capitalize()}\n")
        lines.append("| Model | TPS | TTFT (s) | Latency (s) | Avg Tokens |\n")
        lines.append("|-------|-----|----------|-------------|------------|\n")
        for model in sorted(results.keys()):
            cat_stats = results[model].get("by_category", {}).get(cat)
            if cat_stats:
                lines.append(
                    f"| {model} | {cat_stats['tps_mean']} | {cat_stats['ttft_mean_s']} | "
                    f"{cat_stats['latency_mean_s']} | {cat_stats['tokens_mean']} |\n"
                )
        lines.append("\n")

    # Analysis placeholder
    lines.append("---\n\n## Analysis\n\n")
    lines.append("> **TODO:** Fill in observations after reviewing the numbers above.\n\n")
    lines.append("### Key Findings\n\n")
    lines.append("- \n- \n- \n\n")
    lines.append("### Tradeoffs\n\n")
    lines.append("- \n- \n- \n\n")
    lines.append("### Recommendation\n\n")
    lines.append("> Which model to pick for which use case, based on the data.\n\n")

    report_path.write_text("".join(lines), encoding="utf-8")
    console.print(f"\n[bold green]✓[/bold green] Report generated → [cyan]{report_path}[/cyan]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 — Multi-model comparison runner"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model",  help="Single model to benchmark, e.g. llama3.1:8b")
    group.add_argument("--all",    action="store_true", help="Run all 4 comparison models")
    group.add_argument("--report", action="store_true", help="Generate report from saved results")
    parser.add_argument("--runs",  type=int, default=2, help="Warm runs per prompt (default: 2)")
    parser.add_argument("--num-ctx", type=int, default=4096, help="Context window size (default: 4096, use 16384 for Qwen thinking mode)")
    args = parser.parse_args()

    client = OllamaClient()
    console.print()
    console.print("[bold]Checking Ollama...[/bold] ", end="")
    if not client.is_available():
        console.print("[red]✗ Not running.[/red]")
        console.print("  Start with:  [cyan]ollama serve[/cyan]")
        sys.exit(1)
    console.print("[green]✓ Running[/green]")

    if args.report:
        results = load_all_results()
        print_overall_comparison(results)
        for cat in sorted(set(
            c for d in results.values() for c in d.get("by_category", {})
        )):
            print_category_comparison(results, cat)
        generate_report(results)
        return

    models_to_run = COMPARISON_MODELS if args.all else [args.model]

    for model in models_to_run:
        console.rule(f"[bold]{model}[/bold]")
        result = run_model(model, runs_per_prompt=args.runs, num_ctx=args.num_ctx)
        if result:
            save_model_result(model, result)

    # After running, show current comparison state
    all_results = load_all_results()
    if len(all_results) > 1:
        print_overall_comparison(all_results)

    missing = [m for m in COMPARISON_MODELS if m not in all_results]
    if missing:
        console.print()
        console.print(f"[dim]Still to run: {', '.join(missing)}[/dim]")
        console.print(f"[dim]Run --report once all 4 models are complete.[/dim]")
    else:
        console.print()
        console.print("[bold green]All 4 models complete.[/bold green] Generating report...")
        generate_report(all_results)


if __name__ == "__main__":
    main()
