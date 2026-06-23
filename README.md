# Local AI Inference & Benchmarking Pipeline

A rigorous, CPU-only benchmarking and structured inference pipeline for small language models (SLMs), built to run entirely on consumer hardware — no GPU required.

Three phases: inference performance measurement, structured output validation, and multi-model comparison study. All models run locally via [Ollama](https://ollama.com). No cloud APIs. No CUDA.

---

## Hardware Constraints (By Design)

| Component | Spec |
|-----------|------|
| CPU | 11th Gen Intel Core i5-11320H @ 3.20GHz |
| RAM | 16GB (15.8GB usable) |
| GPU | Intel Iris Xe — integrated, 128MB dedicated |
| OS | Windows 11 64-bit |

> Inference runs entirely on CPU + system RAM. This is a deliberate constraint — the goal is to measure and document what's actually possible at the edge.

---

## Results

### Phase 1 — Llama 3.1 8B Baseline (10 prompts, CPU-only)

| Metric | Value |
|--------|-------|
| Tokens per second (warm avg) | **6.45 TPS** |
| Time to first token (warm avg) | **0.644s** |
| Time to first token (cold) | 2.013s |
| Cold TTFT penalty | +1.37s (CPU cache warmup, not model load) |
| CPU utilization | ~59% |

### Phase 2 — Structured Output & Temperature Experiment

| Metric | Value |
|--------|-------|
| Schema compliance rate | **100%** (50/50 calls) |
| Retries triggered | 0 |
| Temp 0.0 unique explanations | 1 per prompt (near-deterministic) |
| Temp 0.7 unique explanations | 5/5 per prompt (fully diverse) |
| Sentiment consistency | 100% on clear-cut prompts, 80% on ambiguous |

### Phase 3 — Multi-Model Comparison (30 prompts, 6 categories)

| Model | TPS | TTFT (s) | Latency (s) | Avg Tokens | Inference Profile |
|-------|-----|----------|-------------|------------|-------------------|
| llama3.1:8b | 6.16 | **0.743** | **17.6** | 104 | Pure generation |
| gemma4:e4b | **8.87** | 34.763 | 51.8 | 447 | Light reasoning chain |
| qwen3.5:4b | 8.79 | 165.483 | 272.2 | 2326 | Heavy reasoning chain |
| qwen3.5:9b | — | — | — | — | Excluded: est. >15hr benchmark time |

**Key finding:** Raw TPS does not predict usability. Llama 3.1 8B has the lowest TPS but the fastest actual response delivery because every token it generates is a visible output token. Gemma 4 and Qwen 3.5 include internal reasoning chains that consume tokens before producing visible output.

Full analysis → [`Phase3/report/technical_report.md`](./Phase3/report/technical_report.md)

---

## Project Phases

### Phase 1 — Inference Benchmarking ✅

Rigorous measurement across 6 metrics: TTFT, TPS, total latency, RAM delta, CPU utilization, cold vs warm start. 10 prompts across 5 categories. Results saved as structured JSON.

### Phase 2 — Structured Output & Determinism ✅

JSON schema enforcement via Ollama's `format` parameter, Pydantic v2 validation, 1-retry mechanism, and a temperature experiment (0.0 vs 0.7) measuring output variance across 5 prompts × 5 runs × 2 temperatures.

### Phase 3 — Multi-Model Comparison Study ✅

Side-by-side benchmark of 3 SLMs on 30 standardized prompts across 6 categories (factual, reasoning, coding, instruction, summarization, creative). Full technical report included.

---

## Setup

### Prerequisites

- [Ollama](https://ollama.com/download) installed and running
- Python 3.11+
- Git

### Installation

```bash
git clone https://github.com/jkknikhil-git/Local-AI-SLM-Performance-Research.git
cd Local-AI-SLM-Performance-Research

python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Pull Models

```bash
ollama pull llama3.1:8b
ollama pull gemma4:e4b
ollama pull qwen3.5:4b
```

---

## Usage

### Phase 1 — Benchmark a model

```bash
cd Phase1
python benchmark.py --model llama3.1:8b
python benchmark.py --model llama3.1:8b --warm-runs 5
python benchmark.py --model llama3.1:8b --no-cold
```

### Phase 2 — Temperature experiment

```bash
python Phase2/temperature_exp.py --model llama3.1:8b
python Phase2/temperature_exp.py --model llama3.1:8b --runs 5
```

### Phase 3 — Multi-model comparison

```bash
# Run one model at a time (results accumulate):
python Phase3/comparison_runner.py --model llama3.1:8b
python Phase3/comparison_runner.py --model gemma4:e4b
python Phase3/comparison_runner.py --model qwen3.5:4b

# Generate report once all models are complete:
python Phase3/comparison_runner.py --report
```

---

## Project Structure

```
Local-AI-SLM-Performance-Research/
│
├── Phase1/
│   ├── ollama_client.py        # Shared Ollama wrapper (TTFT, CPU monitoring, RAM delta)
│   ├── benchmark.py            # CLI benchmark runner
│   ├── prompts.py              # 10-prompt benchmark suite
│   └── results/                # Timestamped JSON results (committed)
│
├── Phase2/
│   ├── schemas.py              # Pydantic v2 schemas (Factual, Reasoning, Code, Sentiment)
│   ├── inference.py            # Structured output engine + retry logic
│   ├── temperature_exp.py      # Temp 0.0 vs 0.7 experiment runner
│   └── results/                # Experiment results JSON (committed)
│
├── Phase3/
│   ├── test_prompts.py         # 30-prompt standardized benchmark suite
│   ├── comparison_runner.py    # Multi-model runner + report generator
│   ├── results/                # Per-model JSON results (committed)
│   └── report/
│       └── technical_report.md # Full analysis with numbers and recommendations
│
├── .vscode/
│   ├── extensions.json         # Recommended VSCode extensions
│   └── launch.json             # Debug configurations for all phases
│
├── docs/
│   └── notes.md                # Project notes and observations
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| [Ollama](https://ollama.com) | Local model serving (CPU inference) |
| Python 3.11 | Core language |
| [Pydantic v2](https://docs.pydantic.dev) | JSON schema validation |
| [psutil](https://psutil.readthedocs.io) | System resource monitoring |
| [rich](https://rich.readthedocs.io) | Terminal output and progress display |
| pytest | Unit tests for retry/validation logic |

---

## License

MIT © 2026 Nik
