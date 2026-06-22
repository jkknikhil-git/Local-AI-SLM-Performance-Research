"""
Phase2/temperature_exp.py

Temperature Experiment — 0.0 vs 0.7

Runs the same set of prompts at temperature 0.0 and 0.7, multiple times each,
and carefully documents the variance in outputs.

At temp 0.0: responses should be near-deterministic (near-identical across runs).
At temp 0.7: responses should show meaningful variation in wording and details.

The experiment uses SentimentAnalysis — it has a clear "correct" field (sentiment)
that should be stable across temperatures, alongside variable fields (key_phrases,
explanation) where temperature effects will be most visible.

Usage:
    cd "D:\PROJECTS\Local AI & SLM Performance Research"
    python Phase2/temperature_exp.py --model llama3.1:8b
    python Phase2/temperature_exp.py --model llama3.1:8b --runs 5
"""

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev

from rich.console import Console
from rich.table import Table
from rich import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Phase2.inference import StructuredInference, InferenceResult
from Phase2.schemas import SentimentAnalysis

# ---------------------------------------------------------------------------
# Experiment prompts
# ---------------------------------------------------------------------------

TEMP_PROMPTS = [
    {
        "id": "sentiment_positive",
        "text": (
            "Analyse the sentiment of this text:\n\n"
            "\"The new library is absolutely wonderful. The staff are incredibly helpful, "
            "the reading rooms are bright and spacious, and the collection is vast. "
            "I could spend hours here every day.\""
        ),
        "expected_sentiment": "positive",
    },
    {
        "id": "sentiment_negative",
        "text": (
            "Analyse the sentiment of this text:\n\n"
            "\"The service was appalling. We waited over an hour for our food, "
            "the waiter was rude, and when the meal finally arrived it was cold. "
            "We will not be returning.\""
        ),
        "expected_sentiment": "negative",
    },
    {
        "id": "sentiment_neutral",
        "text": (
            "Analyse the sentiment of this text:\n\n"
            "\"The report was published on Tuesday. It contains 42 pages and covers "
            "the financial results for the third quarter. A press conference is "
            "scheduled for Thursday morning.\""
        ),
        "expected_sentiment": "neutral",
    },
    {
        "id": "sentiment_mixed",
        "text": (
            "Analyse the sentiment of this text:\n\n"
            "\"The product has some genuinely impressive features and the build quality "
            "is solid. However, the software is buggy and customer support took three "
            "days to respond to a simple query.\""
        ),
        "expected_sentiment": "mixed — either positive or negative is defensible",
    },
    {
        "id": "sentiment_subtle",
        "text": (
            "Analyse the sentiment of this text:\n\n"
            "\"I suppose the presentation was fine. It covered the main points "
            "and finished on time, which is more than can be said for last month.\""
        ),
        "expected_sentiment": "neutral/mildly positive",
    },
]

RESULTS_DIR = Path(__file__).parent / "results"
console = Console()


# ---------------------------------------------------------------------------
# Variance analysis
# ---------------------------------------------------------------------------

def compute_variance(results: list[InferenceResult]) -> dict:
    """
    Analyse variance across multiple runs of the same prompt.

    Returns:
        - sentiment_consistency: fraction of successful runs with the same sentiment
        - unique_explanations: number of distinct explanation strings
        - unique_phrase_sets: number of distinct key_phrase sets
        - avg_confidence: mean confidence score across runs
        - confidence_stdev: stdev of confidence score (higher = more variable)
        - schema_success_rate: fraction of runs that passed validation
    """
    successful = [r for r in results if r.success and r.parsed]
    total = len(results)

    if not successful:
        return {
            "schema_success_rate": 0.0,
            "sentiment_consistency": 0.0,
            "unique_explanations": 0,
            "unique_phrase_sets": 0,
            "avg_confidence": 0.0,
            "confidence_stdev": 0.0,
        }

    sentiments    = [r.parsed["sentiment"] for r in successful]
    explanations  = [r.parsed["explanation"] for r in successful]
    phrase_sets   = [tuple(sorted(r.parsed["key_phrases"])) for r in successful]
    confidences   = [r.parsed["confidence_score"] for r in successful]

    most_common   = max(set(sentiments), key=sentiments.count)
    consistency   = sentiments.count(most_common) / len(sentiments)

    return {
        "schema_success_rate":  round(len(successful) / total, 3),
        "sentiment_consistency": round(consistency, 3),
        "unique_explanations":  len(set(explanations)),
        "unique_phrase_sets":   len(set(phrase_sets)),
        "avg_confidence":       round(mean(confidences), 3),
        "confidence_stdev":     round(stdev(confidences) if len(confidences) > 1 else 0.0, 3),
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_variance_table(experiment_results: dict, runs: int):
    console.print()
    console.rule("[bold cyan]Temperature Experiment — Variance Analysis[/bold cyan]")
    console.print()

    table = Table(
        title=f"Temp 0.0 vs 0.7  ·  {runs} runs per prompt per temperature",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold magenta",
    )

    table.add_column("Prompt",              style="cyan", no_wrap=True)
    table.add_column("Temp",                justify="center")
    table.add_column("Schema OK",           justify="center")
    table.add_column("Sentiment\nConsistency", justify="center")
    table.add_column("Unique\nExplanations", justify="center")
    table.add_column("Unique\nPhrase Sets", justify="center")
    table.add_column("Confidence\n(avg ± sd)", justify="center")

    for prompt_id, temps in experiment_results.items():
        first = True
        for temp_label, variance in temps.items():
            temp_str  = f"[yellow]{temp_label}[/yellow]" if temp_label == "0.0" else f"[green]{temp_label}[/green]"
            ok_rate   = f"{variance['schema_success_rate']*100:.0f}%"
            consist   = f"{variance['sentiment_consistency']*100:.0f}%"
            uniq_expl = str(variance["unique_explanations"])
            uniq_phr  = str(variance["unique_phrase_sets"])
            conf      = f"{variance['avg_confidence']:.2f} ± {variance['confidence_stdev']:.2f}"

            table.add_row(
                prompt_id if first else "",
                temp_str,
                ok_rate,
                consist,
                uniq_expl,
                uniq_phr,
                conf,
            )
            first = False

    console.print(table)


def print_sample_outputs(experiment_raw: dict, prompt_id: str):
    """Print first successful response at each temperature for a given prompt."""
    console.print()
    console.rule(f"[bold]Sample outputs for: {prompt_id}[/bold]")

    for temp_label, runs in experiment_raw.get(prompt_id, {}).items():
        successful = [r for r in runs if r.success and r.parsed]
        if not successful:
            console.print(f"  [red]Temp {temp_label}: no successful runs[/red]")
            continue

        sample = successful[0].parsed
        console.print(f"\n  [bold]Temp {temp_label}:[/bold]")
        console.print(f"    Sentiment:   [cyan]{sample['sentiment']}[/cyan]")
        console.print(f"    Confidence:  {sample['confidence_score']:.2f}")
        console.print(f"    Key phrases: {sample['key_phrases']}")
        console.print(f"    Explanation: {sample['explanation']}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(model: str, experiment_results: dict, experiment_raw: dict, runs: int):
    RESULTS_DIR.mkdir(exist_ok=True)
    tag = model.replace(":", "_").replace("/", "_")
    ts  = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"temp_exp_{tag}_{ts}.json"

    serialisable_raw = {}
    for prompt_id, temps in experiment_raw.items():
        serialisable_raw[prompt_id] = {}
        for temp_label, runs_list in temps.items():
            serialisable_raw[prompt_id][temp_label] = [r.to_dict() for r in runs_list]

    payload = {
        "experiment": "temperature_0.0_vs_0.7",
        "model": model,
        "schema": "SentimentAnalysis",
        "runs_per_temp": runs,
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "variance_summary": experiment_results,
        "raw_runs": serialisable_raw,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    console.print()
    console.print(f"[bold green]✓[/bold green] Results saved → [cyan]{out_path}[/cyan]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 — Temperature experiment (0.0 vs 0.7)"
    )
    parser.add_argument("--model", required=True, help="Ollama model tag, e.g. llama3.1:8b")
    parser.add_argument("--runs", type=int, default=5, help="Runs per prompt per temperature (default: 5)")
    args = parser.parse_args()

    model = args.model
    runs  = args.runs
    temps = [0.0, 0.7]

    client = StructuredInference()

    console.print()
    console.rule("[bold cyan]Phase 2 — Temperature Experiment[/bold cyan]")
    console.print(f"  Model:       {model}")
    console.print(f"  Schema:      SentimentAnalysis")
    console.print(f"  Prompts:     {len(TEMP_PROMPTS)}")
    console.print(f"  Temperatures: {temps}")
    console.print(f"  Runs each:   {runs}")
    console.print(f"  Total calls: {len(TEMP_PROMPTS) * len(temps) * runs}")
    console.print()

    # experiment_raw:     prompt_id → temp_label → list[InferenceResult]
    # experiment_results: prompt_id → temp_label → variance dict
    experiment_raw:     dict = {}
    experiment_results: dict = {}

    total = len(TEMP_PROMPTS) * len(temps) * runs
    done  = 0

    for prompt in TEMP_PROMPTS:
        experiment_raw[prompt["id"]]     = {}
        experiment_results[prompt["id"]] = {}

        for temp in temps:
            temp_label = str(temp)
            run_results: list[InferenceResult] = []

            for i in range(runs):
                done += 1
                console.print(
                    f"  [{done}/{total}] prompt=[cyan]{prompt['id']}[/cyan]  "
                    f"temp=[yellow]{temp}[/yellow]  run={i+1}/{runs}",
                    end="\r"
                )
                result = client.generate(
                    model=model,
                    prompt=prompt["text"],
                    schema_class=SentimentAnalysis,
                    temperature=temp,
                )
                run_results.append(result)

            experiment_raw[prompt["id"]][temp_label]     = run_results
            experiment_results[prompt["id"]][temp_label] = compute_variance(run_results)

    console.print(" " * 80, end="\r")  # clear progress line

    # --- Display results ---
    print_variance_table(experiment_results, runs)
    print_sample_outputs(experiment_raw, TEMP_PROMPTS[0]["id"])  # show first prompt as example
    save_results(model, experiment_results, experiment_raw, runs)

    # --- Summary interpretation ---
    console.print()
    console.rule("[bold]What to look for[/bold]")
    console.print("""
  [bold]Sentiment consistency[/bold]:  Should be high at both temps for clear-cut prompts.
    If temp 0.7 drops below 80%, the model is flipping sentiment randomly — a reliability concern.

  [bold]Unique explanations[/bold]:    At temp 0.0 this should be 1 (deterministic).
    At temp 0.7 you want this close to the number of runs — it means the model is
    generating genuinely varied outputs, not parroting the same text.

  [bold]Unique phrase sets[/bold]:     Same story — low at temp 0.0, high at temp 0.7.

  [bold]Confidence stdev[/bold]:       High stdev at temp 0.7 means the model is
    uncertain and temperature is amplifying that uncertainty.
    Low stdev means the confidence is stable regardless of wording variation.
""")


if __name__ == "__main__":
    main()
