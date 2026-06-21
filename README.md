# Local AI Inference & Benchmarking Pipeline

A rigorous, CPU-only benchmarking and structured inference pipeline for small language models (SLMs), built to run entirely on consumer hardware — no GPU required.

Built across three phases: performance measurement, structured output validation, and multi-model comparison study. All models run locally via [Ollama](https://ollama.com). No cloud APIs. No CUDA.

---

## Hardware Constraints (By Design)

This project intentionally targets constrained hardware to document the real-world tradeoffs of local AI inference on consumer-grade CPUs.

| Component | Spec |
|-----------|------|
| CPU | 11th Gen Intel Core i5-11320H @ 3.20GHz |
| RAM | 16GB (15.8GB usable) |
| GPU | Intel Iris Xe — integrated, 128MB dedicated |
| OS | Windows 11 64-bit |

> Inference runs entirely on CPU + system RAM. This is a deliberate constraint, not a limitation — the goal is to measure and document what's actually possible at the edge.

---

## Project Phases

### Phase 1 — Inference Benchmarking

Rigorous measurement of model inference performance across key metrics:

- **Time to First Token (TTFT)** — responsiveness under load
- **Tokens per Second (TPS)** — sustained generation throughput
- **Total Response Latency** — end-to-end wall-clock time
- **RAM delta** — memory footprint before vs. after model load
- **CPU utilization** — average % during active generation
- **Cold start vs. warm start** — model load time on first vs. subsequent runs

Results are written to `phase1_benchmarking/results/` as structured JSON for reproducibility.

### Phase 2 — Structured Output & Determinism

Enforcing structure and reliability on top of raw generation:

- JSON schema enforcement via Ollama's `format` parameter
- [Pydantic v2](https://docs.pydantic.dev/latest/) validation of model outputs
- Retry mechanism: one automatic re-prompt on schema failure, then graceful degradation
- **Temperature experiment**: identical prompt set run at `0.0` and `0.7`, with variance in outputs carefully documented

### Phase 3 — Multi-Model Comparison Study

Side-by-side benchmark of four SLMs on identical hardware, identical prompts, identical conditions:

| Model | Ollama Tag | Parameters | Architecture |
|-------|-----------|------------|--------------|
| Llama 3.1 | `llama3.1:8b` | 8B | Dense |
| Gemma 4 | `gemma4:e4b` | E4B (eff. 4B) | MoE |
| Qwen 3.5 | `qwen3.5:4b` | 4B | Dense |
| Qwen 3.5 | `qwen3.5:9b` | 9B | Dense |

Standardized prompt bank of 30–50 prompts spanning reasoning, coding, factual recall, and instruction-following. Quantized variants (Q4\_K\_M vs Q8\_0) tested where hardware permits.

Full technical report with real benchmark numbers and analysis in [`phase3_comparison/report/`](./phase3_comparison/report/).

---

## Results

> Results will be populated as benchmarks complete.
> - Phase 1 results → [`phase1_benchmarking/results/`](./phase1_benchmarking/results/)
> - Phase 3 report → [`phase3_comparison/report/`](./phase3_comparison/report/)

---

## Setup

### Prerequisites

- [Ollama](https://ollama.com/download) installed and running locally
- Python 3.11+
- Git

### Installation

```bash
git clone https://github.com/yourusername/local-ai-inference-benchmark.git
cd local-ai-inference-benchmark

python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### Pull Models

```bash
ollama pull llama3.1:8b
ollama pull gemma4:e4b
ollama pull qwen3.5:4b
ollama pull qwen3.5:9b
```

> Pull time and disk space will vary. Expect 2–6GB per model at default Q4 quantization.

---

## Project Structure

```
local-ai-inference-benchmark/
│
├── utils/
│   └── ollama_client.py        # Shared Ollama wrapper (timing, retries, resource monitoring)
│
├── phase1_benchmarking/
│   ├── benchmark.py            # Core benchmark runner
│   └── results/                # JSON output files (committed for reproducibility)
│
├── phase2_structured/
│   ├── schemas.py              # Pydantic models and JSON schemas
│   ├── inference.py            # Structured output + retry logic
│   └── temperature_exp.py      # Temp 0.0 vs 0.7 experiment runner
│
├── phase3_comparison/
│   ├── test_prompts.py         # Standardized prompt bank
│   ├── comparison_runner.py    # Runs all models on all prompts
│   └── report/                 # Technical report with numbers and analysis
│
├── docs/
│   └── notes.md                # Running project notes and observations
│
├── .vscode/
│   └── extensions.json         # Recommended VSCode extensions
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
