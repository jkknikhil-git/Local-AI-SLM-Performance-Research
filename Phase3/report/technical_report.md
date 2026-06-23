# Phase 3 — Multi-Model Comparison: Technical Report

**Generated:** 2026-06-23  
**Author:** Nik  
**Models compared:** gemma4:e4b, llama3.1:8b, qwen3.5:4b  
**Prompts:** 30 across 6 categories (factual, reasoning, coding, instruction, summarization, creative)  
**Runs per prompt:** 2 warm (+ 1 cold, excluded from averages)  
**Note:** qwen3.5:9b excluded — estimated benchmark time >15 hours on this hardware (itself a finding).

---

## Hardware

| Component | Spec |
|-----------|------|
| CPU | 11th Gen Intel Core i5-11320H @ 3.20GHz |
| RAM | 16GB (15.8GB usable) |
| GPU | Intel Iris Xe (integrated, 128MB dedicated) |
| OS | Windows 11 64-bit |
| Backend | Ollama (CPU-only, no GPU offload) |

---

## Overall Performance

| Model | TPS | TTFT (s) | Latency (s) | Avg Tokens | CPU % |
|-------|-----|----------|-------------|------------|-------|
| llama3.1:8b | 6.16 | **0.743** | **17.604** | **104** | 73% |
| gemma4:e4b | **8.87** | 34.763 | 51.773 | 447 | 71% |
| qwen3.5:4b | 8.79 | 165.483 | 272.223 | 2326 | 72% |

---

## Key Findings

### 1. Three Distinct Inference Profiles Emerged

The most significant finding of this study is that the four models (three benchmarked, one excluded) represent three fundamentally different inference philosophies, each with distinct performance characteristics on CPU-constrained hardware:

**Profile A — Pure Generation (llama3.1:8b)**  
No internal reasoning chain. Every token generated is a visible output token. TTFT of 0.743s is near-instant for CPU-only inference. Latency scales predictably with output length.

**Profile B — Light Reasoning Chain (gemma4:e4b)**  
Built-in chain-of-thought that generates moderate internal reasoning before producing visible output. TTFT of 34.76s reflects this overhead. Average token count of 447 (vs Llama's 104) confirms reasoning overhead, but the model completes successfully on all prompts within a reasonable time.

**Profile C — Heavy Reasoning Chain (qwen3.5:4b)**  
Extensive internal reasoning chain that cannot be disabled via Ollama's API on this hardware configuration. TTFT of 165.48s and average token count of 2326 make it impractical for real-time interactive use. Most of those 2326 tokens are internal thinking tokens, not visible output.

---

### 2. Raw TPS Does Not Predict Usability

A counterintuitive result: Llama 3.1 8B has the *lowest* tokens-per-second (6.16) yet delivers the *fastest* actual responses. Gemma 4 E4B and Qwen 3.5 4B both generate tokens faster (8.87 and 8.79 TPS respectively) but their effective output delivery is slower because a large fraction of generated tokens are internal reasoning tokens.

This distinction between **raw throughput** and **effective output rate** is critical for evaluating SLMs on resource-constrained hardware. A benchmark that only reports TPS would rank Llama last — the opposite of the user experience result.

---

### 3. Thinking Mode Overhead Scales Non-Linearly

| Model | TTFT | Thinking Tokens (est.) | Visible Output (est.) |
|-------|------|------------------------|----------------------|
| llama3.1:8b | 0.7s | 0 | ~104 tokens |
| gemma4:e4b | 34.8s | ~240 tokens | ~207 tokens |
| qwen3.5:4b | 165.5s | ~2200 tokens | ~126 tokens |

Gemma 4's thinking overhead is proportionate — it generates roughly 1 thinking token per output token. Qwen 3.5's overhead is extreme — roughly 17 thinking tokens per visible output token. On a system where each thinking token costs the same compute as an output token, this makes Qwen 3.5's thinking mode unsustainable for practical use on CPU-only hardware.

---

### 4. CPU Utilisation Converges Regardless of Model

All three models saturated the CPU at approximately 70–73%. This confirms that on this hardware, the limiting factor is CPU compute bandwidth — not model size, architecture, or quantisation level. Adding reasoning chains (as in Gemma and Qwen) doesn't increase CPU load; it just increases the number of compute cycles per user-visible token.

---

### 5. qwen3.5:9b Exclusion is a Finding in Itself

Based on qwen3.5:4b taking approximately 9.5 hours to complete the 30-prompt benchmark, the 9B variant was estimated to require 15–20 hours on the same hardware. This exceeds any practical benchmarking window and represents a hard feasibility boundary: models with heavy thinking modes and larger parameter counts cross a threshold where CPU-only evaluation becomes operationally impractical.

---

## Tradeoffs Summary

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| Interactive chat / real-time assistant | **llama3.1:8b** | 0.7s TTFT, predictable latency |
| Quality-oriented reasoning tasks | **gemma4:e4b** | Higher TPS, balanced thinking overhead |
| Complex offline reasoning (no latency constraint) | **qwen3.5:4b** | Deeper reasoning chain, but requires patience |
| Structured output / JSON schema enforcement | **gemma4:e4b** | Native function calling trained into weights |
| CPU-constrained edge deployment | **llama3.1:8b** | Only model with sub-1s TTFT on this hardware |

---

## Connection to Phase 1 & Phase 2

**From Phase 1** (llama3.1:8b baseline, 10-prompt bank):  
The Phase 3 run of llama3.1:8b on 30 prompts produced consistent results — 6.16 TPS vs 6.45 in Phase 1, within normal variance. TTFT remained sub-second across both prompt banks. This validates Phase 1's methodology.

**From Phase 2** (structured output, temperature experiment):  
100% schema compliance was achieved across all 50 structured output calls at both temperatures, with zero retries triggered. This confirms that llama3.1:8b — the primary Phase 2 model — is highly reliable for structured generation tasks at CPU-scale inference speeds.

---

## Limitations

- **Single hardware configuration.** All results are specific to the i5-11320H + 16GB RAM setup. GPU-enabled systems would show dramatically different profiles, particularly for larger models.
- **Default quantisation only.** All models were benchmarked at Ollama's default Q4_K_M quantisation. Q8_0 variants would show higher quality at the cost of ~2x memory and slower TPS.
- **Thinking mode not fully controllable.** Attempts to disable Qwen 3.5's thinking mode via the Ollama API were unsuccessful. Results reflect default model behaviour.
- **Single benchmark run.** Due to the time cost of CPU inference, each model was run twice per prompt (2 warm runs). A larger run count would reduce variance in the averages.
- **No automated quality scoring.** Output quality was assessed by prompt category and response completeness, not by an automated judge model. A future extension could use an LLM-as-judge approach to score outputs against reference answers.

---

## Conclusion

For CPU-only deployment on consumer hardware, **llama3.1:8b remains the most practical choice** for interactive or latency-sensitive applications. Its 0.743s TTFT and predictable scaling make it uniquely suited to the constraints of this hardware class.

**gemma4:e4b** is the best option when reasoning quality matters more than raw latency — its thinking overhead is bounded and proportionate, and its higher TPS means long-form tasks complete faster once the initial thinking phase completes.

**qwen3.5:4b's** heavy reasoning chain, while demonstrating genuine capability, pushes CPU-only inference into a latency regime (~4.5 minutes average per prompt) that is incompatible with real-time use. It may be better suited to batch processing or hardware with GPU support.

The broader lesson: **model selection for edge deployment cannot rely on parameter count or benchmark rankings alone.** Inference architecture — particularly the presence and depth of built-in reasoning chains — has an equal or greater impact on practical performance than model size on CPU-constrained hardware.
