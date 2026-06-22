"""
Phase2/inference.py

Structured output inference with Pydantic validation and retry logic.

Flow for each generation:
  1. Call Ollama with format=<pydantic schema> and a system prompt enforcing JSON
  2. Validate the response with Pydantic
  3. If validation fails → re-prompt once with an explicit correction message
  4. If retry also fails → return a graceful failure result

Usage:
    from Phase2.inference import StructuredInference
    from Phase2.schemas import FactualAnswer

    client = StructuredInference()
    result = client.generate(
        model="llama3.1:8b",
        prompt="What is the boiling point of water?",
        schema_class=FactualAnswer,
        temperature=0.7,
    )
    if result.success:
        print(result.parsed)
    else:
        print(f"Failed after {result.attempts} attempts: {result.validation_error}")
"""

import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, TypeVar, Type

from pydantic import BaseModel, ValidationError
import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """
    Full record of a single structured generation attempt.
    Captures success/failure, attempt count, timing, and the parsed output.
    """
    # Outcome
    success: bool
    attempts: int               # 1 = succeeded first try, 2 = needed retry, 0 = both failed

    # Identity
    model: str
    schema_name: str
    temperature: float
    prompt: str

    # Output
    parsed: Optional[dict]      # parsed.model_dump() if success, else None
    raw_response: str           # raw string from the model
    validation_error: Optional[str]

    # Performance
    latency_s: float
    tokens_per_second: float

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are a precise AI assistant that responds ONLY with valid JSON.
Your response must be a single JSON object matching this schema exactly:

{schema}

Rules:
- Output raw JSON only — no markdown, no code fences, no explanation
- Every required field must be present
- Respect all type constraints (strings, numbers, enums, arrays)
"""

RETRY_PROMPT_TEMPLATE = """\
Your previous response was not valid JSON or did not match the required schema.
Try again carefully.

Required schema:
{schema}

Respond with a single valid JSON object only. No markdown. No explanation.

Original request: {original_prompt}
"""


class StructuredInference:
    """
    Thin wrapper around Ollama's chat API that enforces JSON schema output
    via Pydantic validation, with one automatic retry on failure.
    """

    def __init__(self, host: str = "http://localhost:11434"):
        self._client = ollama.Client(host=host)

    def generate(
        self,
        model: str,
        prompt: str,
        schema_class: Type[T],
        temperature: float = 0.7,
    ) -> InferenceResult:
        """
        Generate a structured response and validate it against schema_class.

        Args:
            model:        Ollama model tag
            prompt:       User prompt
            schema_class: Pydantic model class defining the expected output
            temperature:  Sampling temperature (0.0 = deterministic, 0.7 = varied)

        Returns:
            InferenceResult with success status, parsed output, and metadata
        """
        schema_json = json.dumps(schema_class.model_json_schema(), indent=2)
        system = SYSTEM_PROMPT_TEMPLATE.format(schema=schema_json)
        wall_start = time.perf_counter()

        # --- Attempt 1 ---
        attempt1 = self._call(model, system, prompt, schema_class, temperature)

        if attempt1["success"]:
            return self._build_result(
                success=True,
                attempts=1,
                model=model,
                schema_name=schema_class.__name__,
                temperature=temperature,
                prompt=prompt,
                attempt=attempt1,
                wall_start=wall_start,
            )

        # --- Attempt 2 (retry) ---
        retry_prompt = RETRY_PROMPT_TEMPLATE.format(
            schema=schema_json,
            original_prompt=prompt,
        )
        attempt2 = self._call(model, system, retry_prompt, schema_class, temperature)

        return self._build_result(
            success=attempt2["success"],
            attempts=2,
            model=model,
            schema_name=schema_class.__name__,
            temperature=temperature,
            prompt=prompt,
            attempt=attempt2,
            wall_start=wall_start,
            prior_error=attempt1.get("error"),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call(
        self,
        model: str,
        system: str,
        prompt: str,
        schema_class: Type[T],
        temperature: float,
    ) -> dict:
        """
        Single chat call to Ollama. Returns a dict with success, parsed,
        raw, tps, and error fields.
        """
        raw = ""
        try:
            response = self._client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                format=schema_class.model_json_schema(),
                options={"temperature": temperature},
            )
            raw = response.message.content or ""

            # Compute TPS from Ollama's own timing stats
            tps = 0.0
            if response.eval_count and response.eval_duration:
                tps = round(response.eval_count / (response.eval_duration / 1e9), 2)

            # Validate with Pydantic
            parsed = schema_class.model_validate_json(raw)
            return {
                "success": True,
                "parsed": parsed.model_dump(),
                "raw": raw,
                "tps": tps,
                "error": None,
            }

        except ValidationError as e:
            return {
                "success": False,
                "parsed": None,
                "raw": raw,
                "tps": 0.0,
                "error": f"ValidationError: {e.error_count()} error(s) — {e.errors()[0]['msg']}",
            }
        except Exception as e:
            return {
                "success": False,
                "parsed": None,
                "raw": raw,
                "tps": 0.0,
                "error": f"{type(e).__name__}: {e}",
            }

    def _build_result(
        self,
        success: bool,
        attempts: int,
        model: str,
        schema_name: str,
        temperature: float,
        prompt: str,
        attempt: dict,
        wall_start: float,
        prior_error: Optional[str] = None,
    ) -> InferenceResult:
        error_msg = attempt.get("error")
        if prior_error and not success:
            error_msg = f"Attempt 1: {prior_error} | Attempt 2: {error_msg}"

        return InferenceResult(
            success=success,
            attempts=attempts,
            model=model,
            schema_name=schema_name,
            temperature=temperature,
            prompt=prompt,
            parsed=attempt.get("parsed"),
            raw_response=attempt.get("raw", ""),
            validation_error=error_msg,
            latency_s=round(time.perf_counter() - wall_start, 4),
            tokens_per_second=attempt.get("tps", 0.0),
        )
