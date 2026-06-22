"""
Phase2/schemas.py

Pydantic v2 schemas used for structured output enforcement in Phase 2.

Each schema maps to a category of task (factual, reasoning, code, sentiment).
These are passed to Ollama's format parameter and validated after generation.
"""

from typing import Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Factual
# ---------------------------------------------------------------------------

class FactualAnswer(BaseModel):
    """
    For short, direct factual questions.
    Tests whether the model can produce clean structured answers.
    """
    question: str = Field(description="The original question, restated")
    answer: str = Field(description="Concise factual answer")
    confidence: Literal["low", "medium", "high"] = Field(
        description="Model's confidence in the answer"
    )


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------

class ReasoningStep(BaseModel):
    step_number: int
    description: str = Field(description="What this step does or concludes")


class ReasoningResponse(BaseModel):
    """
    For multi-step reasoning and math problems.
    Forces the model to externalise its reasoning chain as discrete steps.
    """
    question: str = Field(description="The original question, restated")
    steps: list[ReasoningStep] = Field(
        description="Step-by-step reasoning, minimum 2 steps",
        min_length=2
    )
    final_answer: str = Field(description="The final answer after reasoning")


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------

class CodeResponse(BaseModel):
    """
    For coding tasks.
    Separates code from explanation and enforces structure.
    """
    language: str = Field(description="Programming language, e.g. 'python'")
    function_name: str = Field(description="Name of the primary function")
    code: str = Field(description="Complete, runnable code")
    docstring: str = Field(description="One-sentence description of what the function does")
    example_call: str = Field(description="One example function call with expected output as a comment")


# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------

class SentimentAnalysis(BaseModel):
    """
    For opinion/sentiment classification.
    Useful for the temperature experiment — has a clear correct answer
    (sentiment) but variable fields (key_phrases, explanation) that
    reveal how much temperature affects output diversity.
    """
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment of the text"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in sentiment classification, between 0.0 and 1.0"
    )
    key_phrases: list[str] = Field(
        description="2 to 4 key phrases that drove the sentiment classification",
        min_length=2,
        max_length=4
    )
    explanation: str = Field(
        description="One sentence explaining the sentiment classification"
    )


# ---------------------------------------------------------------------------
# Registry — maps string name to schema class
# ---------------------------------------------------------------------------

SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "factual":    FactualAnswer,
    "reasoning":  ReasoningResponse,
    "code":       CodeResponse,
    "sentiment":  SentimentAnalysis,
}
