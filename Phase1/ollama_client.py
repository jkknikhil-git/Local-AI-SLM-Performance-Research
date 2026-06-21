"""
utils/ollama_client.py

Shared Ollama wrapper used across all phases.
Handles generation, timing instrumentation, and resource monitoring.
"""

import time
import threading
from dataclasses import dataclass, asdict
from typing import Optional

import psutil
import ollama


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkMetrics:
    """
    All metrics collected from a single generation run.
    Serialisable to dict/JSON via dataclasses.asdict().
    """
    # Identity
    model: str
    prompt_id: str
    prompt: str
    response: str
    run_index: int
    run_type: str               # "cold" | "warm"

    # Timing (seconds)
    time_to_first_token_s: float
    total_latency_s: float
    eval_duration_s: float      # Pure token generation time (from Ollama stats)
    prompt_eval_duration_s: float
    load_duration_s: float      # Model load time; non-zero on cold start

    # Throughput
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float    # completion_tokens / eval_duration_s

    # System resources
    ram_system_before_mb: float
    ram_system_after_mb: float
    ram_delta_mb: float
    cpu_percent_avg: float      # Average CPU % sampled during generation

    # Meta
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# CPU monitoring helper
# ---------------------------------------------------------------------------

class _CPUMonitor:
    """
    Samples CPU usage in a background thread.
    Usage:
        monitor = _CPUMonitor()
        monitor.start()
        # ... do work ...
        avg = monitor.stop()
    """
    def __init__(self, interval: float = 0.5):
        self._interval = interval
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._stop.clear()
        self._samples.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            self._samples.append(psutil.cpu_percent(interval=self._interval))

    def stop(self) -> float:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        if not self._samples:
            return 0.0
        return round(sum(self._samples) / len(self._samples), 2)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Thin wrapper around the Ollama Python client with benchmarking built in.
    All phases import and use this class directly.
    """

    def __init__(self, host: str = "http://localhost:11434"):
        self._client = ollama.Client(host=host)
        self._host = host

    def is_available(self) -> bool:
        """Return True if Ollama is running and reachable."""
        try:
            self._client.list()
            return True
        except Exception:
            return False

    def list_local_models(self) -> list[str]:
        """Return names of models already pulled locally."""
        try:
            return [m.model for m in self._client.list().models]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Core benchmarked generation
    # ------------------------------------------------------------------

    def benchmark_generate(
        self,
        model: str,
        prompt: str,
        prompt_id: str = "unnamed",
        run_index: int = 0,
        run_type: str = "warm",
    ) -> BenchmarkMetrics:
        """
        Generate a response and return a fully populated BenchmarkMetrics.

        Uses streaming so we can capture Time-to-First-Token (TTFT) —
        the wall-clock time from sending the request to receiving the
        first token. This is not available from Ollama's own stats.

        Args:
            model:      Ollama model tag, e.g. "llama3.1:8b"
            prompt:     The prompt string to generate from
            prompt_id:  Short identifier for the prompt (e.g. "factual_short")
            run_index:  Which run this is (0 = first / cold, 1+ = warm)
            run_type:   "cold" or "warm"
        """
        # --- RAM before ---
        ram_before = psutil.virtual_memory().used / 1_048_576  # bytes → MB

        # --- CPU monitor ---
        cpu_monitor = _CPUMonitor(interval=0.5)
        cpu_monitor.start()

        # --- Streaming generate ---
        first_token_time: Optional[float] = None
        full_response = ""
        final_chunk = None

        start_time = time.perf_counter()

        try:
            for chunk in self._client.generate(model=model, prompt=prompt, stream=True):
                if first_token_time is None and chunk.response:
                    first_token_time = time.perf_counter()
                full_response += chunk.response
                if chunk.done:
                    final_chunk = chunk
        except Exception as e:
            cpu_monitor.stop()
            raise RuntimeError(f"Generation failed for model '{model}': {e}") from e

        end_time = time.perf_counter()

        # --- Stop CPU monitor ---
        cpu_avg = cpu_monitor.stop()

        # --- RAM after ---
        ram_after = psutil.virtual_memory().used / 1_048_576

        # --- Extract Ollama timing stats (nanoseconds → seconds) ---
        def ns(val) -> float:
            return (val or 0) / 1_000_000_000

        eval_duration_s      = ns(getattr(final_chunk, "eval_duration", 0))
        prompt_eval_dur_s    = ns(getattr(final_chunk, "prompt_eval_duration", 0))
        load_duration_s      = ns(getattr(final_chunk, "load_duration", 0))
        eval_count           = getattr(final_chunk, "eval_count", 0) or 0
        prompt_eval_count    = getattr(final_chunk, "prompt_eval_count", 0) or 0

        ttft = round((first_token_time - start_time), 4) if first_token_time else 0.0
        tps  = round(eval_count / eval_duration_s, 2) if eval_duration_s > 0 else 0.0

        return BenchmarkMetrics(
            model=model,
            prompt_id=prompt_id,
            prompt=prompt,
            response=full_response.strip(),
            run_index=run_index,
            run_type=run_type,
            time_to_first_token_s=ttft,
            total_latency_s=round(end_time - start_time, 4),
            eval_duration_s=round(eval_duration_s, 4),
            prompt_eval_duration_s=round(prompt_eval_dur_s, 4),
            load_duration_s=round(load_duration_s, 4),
            prompt_tokens=prompt_eval_count,
            completion_tokens=eval_count,
            tokens_per_second=tps,
            ram_system_before_mb=round(ram_before, 2),
            ram_system_after_mb=round(ram_after, 2),
            ram_delta_mb=round(ram_after - ram_before, 2),
            cpu_percent_avg=cpu_avg,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
