"""
phase1_benchmarking/prompts.py

Standardized prompt bank for Phase 1 benchmarking.

Prompts are deliberately varied in type and expected output length
to surface how different prompt characteristics affect latency and TPS.
Each prompt has a short ID used in result filenames and reports.
"""

from dataclasses import dataclass


@dataclass
class Prompt:
    id: str
    category: str
    text: str
    note: str = ""


BENCHMARK_PROMPTS: list[Prompt] = [
    # ---- Factual / short output ----------------------------------------
    Prompt(
        id="factual_capital",
        category="factual",
        text="What is the capital of Japan?",
        note="Minimal output. Tests TTFT on trivial prompt.",
    ),
    Prompt(
        id="factual_definition",
        category="factual",
        text="In one sentence, what is the Transformer architecture in machine learning?",
        note="Constrained single-sentence output.",
    ),

    # ---- Reasoning -------------------------------------------------------
    Prompt(
        id="reasoning_math",
        category="reasoning",
        text=(
            "A train travels at 80 km/h. It departs at 9:15 AM and arrives at 1:00 PM. "
            "How many kilometres did it travel? Show your working."
        ),
        note="Arithmetic + multi-step reasoning.",
    ),
    Prompt(
        id="reasoning_logic",
        category="reasoning",
        text=(
            "All mammals are warm-blooded. Dolphins are mammals. "
            "Are dolphins warm-blooded? Explain your reasoning step by step."
        ),
        note="Simple syllogistic logic.",
    ),

    # ---- Coding ----------------------------------------------------------
    Prompt(
        id="coding_palindrome",
        category="coding",
        text=(
            "Write a Python function called `is_palindrome` that returns True if a string "
            "reads the same forwards and backwards, ignoring case and spaces. "
            "Include a docstring and two example calls."
        ),
        note="Short function. Tests code generation quality.",
    ),
    Prompt(
        id="coding_fibonacci",
        category="coding",
        text=(
            "Write a Python function that returns the first N Fibonacci numbers as a list. "
            "Use an iterative approach. Include type hints."
        ),
        note="Slightly longer function. Iterative constraint.",
    ),

    # ---- Instruction following -------------------------------------------
    Prompt(
        id="instruction_list",
        category="instruction",
        text=(
            "List exactly 5 practical benefits of regular exercise. "
            "Format your response as a numbered list. "
            "Each item should be one sentence only."
        ),
        note="Tests strict instruction following and output format.",
    ),
    Prompt(
        id="instruction_summarise",
        category="instruction",
        text=(
            "Summarise the following in exactly 3 bullet points:\n\n"
            "Machine learning is a subset of artificial intelligence that enables systems "
            "to learn and improve from experience without being explicitly programmed. "
            "It focuses on developing computer programs that can access data and use it "
            "to learn for themselves. The process begins with observations or data, such "
            "as examples, direct experience, or instruction, so that computers can look "
            "for patterns in data and make better decisions in the future."
        ),
        note="Constrained summarisation. Measures instruction compliance.",
    ),

    # ---- Long-form output ------------------------------------------------
    Prompt(
        id="longform_explanation",
        category="long_form",
        text=(
            "Explain how attention mechanisms work in transformer models. "
            "Cover: what a query, key, and value are; how the attention score is computed; "
            "and why this is better than RNNs. Write approximately 200 words."
        ),
        note="Longer generation. Stresses TPS measurement.",
    ),
    Prompt(
        id="longform_essay",
        category="long_form",
        text=(
            "Write a short essay (around 150 words) on the tradeoffs between "
            "running AI models locally versus using cloud APIs. "
            "Discuss privacy, cost, latency, and hardware requirements."
        ),
        note="Open-ended generation. Tests sustained throughput.",
    ),
]
