"""
phase1_benchmarking/benchmark.py

Phase 1 — Inference Benchmarking

Runs all benchmark prompts against a specified model, collects metrics,
saves results to JSON, and prints a summary table to the terminal.

Usage:
    python phase1_benchmarking/benchmark.py --model llama3.1:8b
    python phase1_benchmarking/benchmark.py --model qwen3.5:4b --warm-runs 5
    python phase1_benchmarking/benchmark.py --model gemma4:e4b --no-cold
"""

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

from ollama_client import OllamaClient, BenchmarkMetrics
from prompts import BENCHMARK_PROMPTS
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"
HARDWARE_META = {
    "cpu": "11th Gen Intel Core i5-11320H @ 3.20GHz",
    "ram_gb": 16,
    "gpu": "Intel Iris Xe (integrated, 128MB dedicated)",
    "os": "Windows 11 64-bit",
    "inference_backend": "Ollama (CPU-only, no GPU offload)",
}

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(model: str, warm_runs: int, include_cold: bool):
    console.rule("[bold cyan]Local AI Inference Benchmark — Phase 1[/bold cyan]")
    console.print(f"  [bold]Model:[/bold]      {model}")
    console.print(f"  [bold]Warm runs:[/bold]  {warm_runs} per prompt")
    console.print(f"  [bold]Cold run:[/bold]   {'yes (run 0)' if include_cold else 'skipped'}")
    console.print(f"  [bold]Prompts:[/bold]    {len(BENCHMARK_PROMPTS)}")
    console.print()


def summarise(runs: list[BenchmarkMetrics]) -> dict:
    """Compute aggregate stats over a list of warm runs."""
    def _safe_stdev(vals):
        return round(stdev(vals), 3) if len(vals) > 1 else 0.0

    tps    = [r.tokens_per_second for r in runs]
    ttft   = [r.time_to_first_token_s for r in runs]
    lat    = [r.total_latency_s for r in runs]
    ram    = [r.ram_delta_mb for r in runs]
    cpu    = [r.cpu_percent_avg for r in runs]
    tokens = [r.completion_tokens for r in runs]

    return {
        "runs_included": len(runs),
        "tokens_per_second":        {"mean": round(mean(tps), 2),  "stdev": _safe_stdev(tps)},
        "time_to_first_token_s":    {"mean": round(mean(ttft), 4), "stdev": _safe_stdev(ttft)},
        "total_latency_s":          {"mean": round(mean(lat), 4),  "stdev": _safe_stdev(lat)},
        "ram_delta_mb":             {"mean": round(mean(ram), 2),  "stdev": _safe_stdev(ram)},
        "cpu_percent_avg":          {"mean": round(mean(cpu), 2),  "stdev": _safe_stdev(cpu)},
        "completion_tokens":        {"mean": round(mean(tokens), 1)},
    }


def print_summary_table(results: list[BenchmarkMetrics], model: str):
    """Print a rich summary table of per-prompt warm-run averages."""
    console.print()
    console.rule("[bold green]Results Summary[/bold green]")
    console.print()

    table = Table(
        title=f"Phase 1 Benchmark — {model}",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold magenta",
    )

    table.add_column("Prompt ID",    style="cyan",  no_wrap=True)
    table.add_column("Category",     style="dim")
    table.add_column("TPS",          justify="right", style="green")
    table.add_column("TTFT (s)",     justify="right")
    table.add_column("Latency (s)",  justify="right")
    table.add_column("Tokens",       justify="right")
    table.add_column("RAM Δ (MB)",   justify="right", style="yellow")
    table.add_column("CPU %",        justify="right", style="red")

    # Group by prompt ID, average warm runs
    from itertools import groupby
    warm = [r for r in results if r.run_type == "warm"]
    warm.sort(key=lambda r: r.prompt_id)

    for prompt_id, group in groupby(warm, key=lambda r: r.prompt_id):
        grp = list(group)
        category = grp[0].prompt_id.split("_")[0]

        avg_tps    = round(mean(r.tokens_per_second for r in grp), 2)
        avg_ttft   = round(mean(r.time_to_first_token_s for r in grp), 3)
        avg_lat    = round(mean(r.total_latency_s for r in grp), 3)
        avg_tokens = round(mean(r.completion_tokens for r in grp))
        avg_ram    = round(mean(r.ram_delta_mb for r in grp), 1)
        avg_cpu    = round(mean(r.cpu_percent_avg for r in grp), 1)

        table.add_row(
            prompt_id,
            category,
            str(avg_tps),
            str(avg_ttft),
            str(avg_lat),
            str(avg_tokens),
            str(avg_ram),
            f"{avg_cpu}%",
        )

    console.print(table)


def print_cold_vs_warm(results: list[BenchmarkMetrics]):
    """Print a compact cold vs warm comparison."""
    cold = [r for r in results if r.run_type == "cold"]
    warm = [r for r in results if r.run_type == "warm"]

    if not cold:
        return

    console.print()
    console.rule("[bold yellow]Cold Start vs Warm Runs[/bold yellow]")
    console.print()

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold")
    table.add_column("Metric")
    table.add_column("Cold Run (avg)", justify="right", style="yellow")
    table.add_column("Warm Runs (avg)", justify="right", style="green")
    table.add_column("Δ", justify="right", style="dim")

    def fmt(val):
        return f"{val:.3f}"

    pairs = [
        ("Tokens / sec",     "tokens_per_second"),
        ("TTFT (s)",         "time_to_first_token_s"),
        ("Total latency (s)","total_latency_s"),
        ("Load duration (s)","load_duration_s"),
    ]

    for label, attr in pairs:
        c = mean(getattr(r, attr) for r in cold)
        w = mean(getattr(r, attr) for r in warm)
        delta = w - c
        sign  = "+" if delta >= 0 else ""
        table.add_row(label, fmt(c), fmt(w), f"{sign}{fmt(delta)}")

    console.print(table)


def save_results(model: str, results: list[BenchmarkMetrics], summary: dict):
    """Save all results to a timestamped JSON file."""
    RESULTS_DIR.mkdir(exist_ok=True)

    tag = model.replace(":", "_").replace("/", "_")
    ts  = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{tag}_{ts}.json"

    payload = {
        "model": model,
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": HARDWARE_META,
        "prompt_count": len(BENCHMARK_PROMPTS),
        "summary": summary,
        "runs": [r.to_dict() for r in results],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    console.print()
    console.print(f"[bold green]✓[/bold green] Results saved → [cyan]{out_path}[/cyan]")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 — Benchmark a local Ollama model"
    )
    parser.add_argument(
        "--model", required=True,
        help="Ollama model tag, e.g. llama3.1:8b"
    )
    parser.add_argument(
        "--warm-runs", type=int, default=3,
        help="Number of warm runs per prompt (default: 3)"
    )
    parser.add_argument(
        "--no-cold", action="store_true",
        help="Skip the cold-start run (first run per prompt)"
    # )
    args = parser.parse_args()

    model      = args.model
    warm_runs  = args.warm_runs
    skip_cold  = args.no_cold
    include_cold = not skip_cold

    # --- Preflight checks ---
    client = OllamaClient()

    console.print()
    console.print("[bold]Checking Ollama...[/bold] ", end="")
    if not client.is_available():
        console.print("[red]✗ Ollama is not running.[/red]")
        console.print("  Start it with:  [cyan]ollama serve[/cyan]")
        sys.exit(1)
    console.print("[green]✓ Running[/green]")

    local_models = client.list_local_models()
    console.print(f"[bold]Checking model '{model}'...[/bold] ", end="")
    if not any(model in m for m in local_models):
        console.print(f"[red]✗ Not found locally.[/red]")
        console.print(f"  Pull it with:  [cyan]ollama pull {model}[/cyan]")
        sys.exit(1)
    console.print("[green]✓ Available[/green]")

    print_header(model, warm_runs, include_cold)

    # --- Run benchmarks ---
    all_results: list[BenchmarkMetrics] = []
    total_runs = len(BENCHMARK_PROMPTS) * ((1 if include_cold else 0) + warm_runs)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        task = progress.add_task("Benchmarking...", total=total_runs)

        for prompt in BENCHMARK_PROMPTS:
            run_offset = 0

            # Cold run (run_index = 0)
            if include_cold:
                progress.update(task, description=f"[yellow]COLD[/yellow]  {prompt.id}")
                metrics = client.benchmark_generate(
                    model=model,
                    prompt=prompt.text,
                    prompt_id=prompt.id,
                    run_index=0,
                    run_type="cold",
                )
                all_results.append(metrics)
                progress.advance(task)
                run_offset = 1

            # Warm runs
            for i in range(warm_runs):
                progress.update(
                    task,
                    description=f"[green]WARM {i+1}/{warm_runs}[/green]  {prompt.id}"
                )
                metrics = client.benchmark_generate(
                    model=model,
                    prompt=prompt.text,
                    prompt_id=prompt.id,
                    run_index=run_offset + i,
                    run_type="warm",
                )
                all_results.append(metrics)
                progress.advance(task)

    # --- Summarise and display ---
    warm_results = [r for r in all_results if r.run_type == "warm"]
    summary = summarise(warm_results)

    print_summary_table(all_results, model)
    print_cold_vs_warm(all_results)

    # --- Overall summary line ---
    console.print()
    console.print(
        f"[bold]Overall (warm runs):[/bold]  "
        f"TPS: [green]{summary['tokens_per_second']['mean']}[/green]  |  "
        f"TTFT: [cyan]{summary['time_to_first_token_s']['mean']}s[/cyan]  |  "
        f"Latency: {summary['total_latency_s']['mean']}s  |  "
        f"RAM Δ: [yellow]{summary['ram_delta_mb']['mean']} MB[/yellow]  |  "
        f"CPU: [red]{summary['cpu_percent_avg']['mean']}%[/red]"
    )

    save_results(model, all_results, summary)


if __name__ == "__main__":
    main()
