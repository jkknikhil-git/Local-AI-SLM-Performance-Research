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

### Phase 1 — Inference Benchmarking ✅

Rigorous measurement of model inference performance across key metrics:

- **Time to First Token (TTFT)** — responsiveness under load
- **Tokens per Second (TPS)** — sustained generation throughput
- **Total Response Latency** — end-to-end wall-clock time
- **RAM delta** — memory footprint before vs. after model load
- **CPU utilization** — average % during active generation
- **Cold start vs. warm start** — first-run vs. subsequent-run performance

Results committed to [`Phase1/results/`](./Phase1/results/) as structured JSON for reproducibility.

### Phase 2 — Structured Output & Determinism 🔄

Enforcing structure and reliability on top of raw generation:

- JSON schema enforcement via Ollama's `format` parameter
- [Pydantic v2](https://docs.pydantic.dev/latest/) validation of model outputs
- Retry mechanism: one automatic re-prompt on schema failure, then graceful degradation
- **Temperature experiment**: identical prompt set run at `0.0` and `0.7`, with variance in outputs carefully documented

### Phase 3 — Multi-Model Comparison Study 🔄

Side-by-side benchmark of four SLMs on identical hardware, identical prompts, identical conditions:

| Model | Ollama Tag | Parameters | Architecture |
|-------|-----------|------------|--------------|
| Llama 3.1 | `llama3.1:8b` | 8B | Dense |
| Gemma 4 | `gemma4:e4b` | E4B (eff. 4B) | MoE |
| Qwen 3.5 | `qwen3.5:4b` | 4B | Dense |
| Qwen 3.5 | `qwen3.5:9b` | 9B | Dense |

Standardized prompt bank of 30–50 prompts spanning reasoning, coding, factual recall, and instruction-following. Quantized variants (Q4\_K\_M vs Q8\_0) tested where hardware permits.

Full technical report in [`Phase3/report/`](./Phase3/report/).

---

## Results

### Phase 1 — Llama 3.1 8B (CPU-only, Q4_K_M)

Benchmark run on i5-11320H · 16GB RAM · No GPU · Windows 11

| Metric | Value |
|--------|-------|
| Tokens per second (warm avg) | **6.45 TPS** |
| Time to first token (warm avg) | **0.644s** |
| Time to first token (cold) | 2.013s |
| Cold TTFT penalty | +1.37s |
| CPU utilization (avg) | ~59% |

**Per-category breakdown (warm runs):**

| Category | TPS | TTFT (s) | Latency (s) | Avg Tokens |
|----------|-----|----------|-------------|------------|
| Factual | 7.11 | 0.625 | 5.2 | 32 |
| Reasoning | 6.41 | 0.639 | 32.3 | 203 |
| Coding | 5.96 | 0.649 | 50.8 | 300 |
| Instruction | 6.32 | 0.648 | 15.6 | 94 |
| Long-form | 6.44 | 0.663 | 35.1 | 221 |

**Key findings:**
- TPS is remarkably stable across prompt types (5.81–7.47 range) — throughput is CPU-bound, not task-bound
- Sub-second TTFT on warm runs despite CPU-only inference
- Cold TTFT penalty (1.37s) is from CPU cache warmup, not model loading — Ollama keeps the model resident in RAM
- Latency scales almost linearly with token count: tokens ÷ 6.45 ≈ wait time in seconds

Full results JSON → [`Phase1/results/`](./Phase1/results/)

---

## Setup

### Prerequisites

- [Ollama](https://ollama.com/download) installed and running locally
- Python 3.11+
- Git

### Installation

```bash
git clone https://github.com/jkknikhil-git/Local-AI-SLM-Performance-Research.git
cd Local-AI-SLM-Performance-Research

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

### Run Phase 1 Benchmark

```bash
cd Phase1
python benchmark.py --model llama3.1:8b
python benchmark.py --model llama3.1:8b --warm-runs 5
python benchmark.py --model llama3.1:8b --no-cold
```

---

## Project Structure

```
Local-AI-SLM-Performance-Research/
│
├── Phase1/
│   ├── ollama_client.py        # Shared Ollama wrapper (TTFT capture, CPU monitoring, RAM delta)
│   ├── benchmark.py            # CLI benchmark runner with rich terminal output
│   ├── prompts.py              # 10-prompt standardized benchmark suite
│   └── results/                # Timestamped JSON output (committed for reproducibility)
│
├── Phase2/                     # Structured output + temperature experiments (in progress)
│
├── Phase3/                     # Multi-model comparison study (in progress)
│   └── report/                 # Technical report with numbers and analysis
│
├── .vscode/
│   ├── extensions.json         # Recommended VSCode extensions
│   └── launch.json             # Debug configurations
│
├── docs/
│   └── notes.md                # Running project notes and observations
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
| [Pydantic v2](https://docs.pydantic.dev) | JSON schema validation (Phase 2) |
| [psutil](https://psutil.readthedocs.io) | System resource monitoring |
| [rich](https://rich.readthedocs.io) | Terminal output and progress display |
| pytest | Unit tests for retry/validation logic |

---

## License

MIT © 2026 Nik
