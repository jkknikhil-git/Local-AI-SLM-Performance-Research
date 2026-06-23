# Project Notes

Running observations and findings logged across all phases.

---

## Phase 1 — Inference Benchmarking (llama3.1:8b)

- TPS is remarkably flat across prompt types (5.81–7.47 range) — throughput is CPU-bound, not task-bound
- Cold TTFT penalty of +1.37s is from CPU cache warmup, not model load — Ollama keeps the model resident in RAM between calls
- Load duration barely changes between cold and warm runs (0.457s vs 0.476s), confirming the model stays loaded
- Latency scales almost linearly with token count: tokens ÷ 6.45 ≈ wait time in seconds
- RAM delta readings are noisy (negative values observed) because psutil measures system-wide RAM, not process-level

## Phase 2 — Structured Output & Temperature

- 100% schema compliance across all 50 calls with zero retries — Ollama's `format` parameter + constrained decoding is highly reliable
- Temp 0.0 produces 1 unique explanation per prompt (near-deterministic), temp 0.7 produces 5/5 (fully diverse)
- Sentiment classification stays consistent across both temperatures for clear-cut prompts
- `sentiment_subtle` (ambiguous prompt) flipped sentiment at 80% consistency at both temps — ambiguity is in the prompt, not introduced by temperature
- Counterintuitive finding: temp 0.7 increased confidence score on `sentiment_neutral` from 0.80 to 0.90 — temperature doesn't just add noise, it can shift expressed certainty unpredictably
- Even at temp 0.0, `sentiment_negative` produced 2 unique explanations — minor non-determinism exists even at zero temperature on CPU inference (likely floating-point variance)

## Phase 3 — Multi-Model Comparison

- Three distinct inference profiles emerged: pure generation (Llama), light CoT (Gemma), heavy CoT (Qwen)
- Raw TPS is a misleading metric when reasoning chains are present — Llama's "slow" 6.16 TPS delivers the fastest visible output
- All models converged on ~70–73% CPU utilization — the CPU is the hard bottleneck regardless of model architecture
- Qwen 3.5 4B's thinking mode could not be disabled via Ollama's Python API — `think: False` option had no effect
- `/no_think` prompt directive also had no effect via the generate API
- qwen3.5:9b excluded after estimating >15hr benchmark time based on 4b extrapolation — this is itself a data point about CPU feasibility limits
- Gemma 4 E4B produced the best balance of quality and speed — higher TPS than Llama with bounded reasoning overhead
- Thinking mode overhead in Qwen is non-linear: ~17 thinking tokens per visible output token vs ~1:1 in Gemma

## General Hardware Observations

- i5-11320H handles sustained multi-hour inference without thermal throttling
- 16GB RAM is sufficient for all tested models at Q4_K_M quantization (no swap observed)
- Windows 11 background processes contribute ~3-4GB baseline RAM usage
- Ollama keeps models resident between runs — no reload penalty on consecutive prompts with the same model
