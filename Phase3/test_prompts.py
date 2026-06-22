"""
Phase3/test_prompts.py

Standardized prompt bank for Phase 3 multi-model comparison.

30 prompts across 6 categories (5 each):
  - factual       : short, direct knowledge retrieval
  - reasoning     : multi-step logic and arithmetic
  - coding        : function writing with constraints
  - instruction   : strict format and rule compliance
  - summarization : compression and information extraction
  - creative      : open-ended generation

Prompts are intentionally varied in expected output length to stress-test
both short-burst and sustained generation throughput across all models.
"""

from dataclasses import dataclass


@dataclass
class Prompt:
    id: str
    category: str
    text: str
    expected_length: str    # "short" | "medium" | "long"
    note: str = ""


COMPARISON_PROMPTS: list[Prompt] = [

    # -----------------------------------------------------------------------
    # FACTUAL (5)
    # -----------------------------------------------------------------------
    Prompt(
        id="fact_01",
        category="factual",
        text="What is the speed of light in a vacuum, in metres per second?",
        expected_length="short",
        note="Single precise numerical answer.",
    ),
    Prompt(
        id="fact_02",
        category="factual",
        text="Who developed the theory of general relativity, and in what year was it published?",
        expected_length="short",
        note="Two-part factual recall.",
    ),
    Prompt(
        id="fact_03",
        category="factual",
        text="What does the HTTP status code 429 mean?",
        expected_length="short",
        note="Technical factual recall.",
    ),
    Prompt(
        id="fact_04",
        category="factual",
        text="Name the four types of chemical bonds and give one example of each.",
        expected_length="medium",
        note="Structured factual enumeration.",
    ),
    Prompt(
        id="fact_05",
        category="factual",
        text=(
            "What are the three laws of thermodynamics? "
            "State each one in one sentence."
        ),
        expected_length="medium",
        note="Multi-part factual with brevity constraint.",
    ),

    # -----------------------------------------------------------------------
    # REASONING (5)
    # -----------------------------------------------------------------------
    Prompt(
        id="reason_01",
        category="reasoning",
        text=(
            "A shop sells apples for £0.45 each and oranges for £0.60 each. "
            "Alice buys 7 apples and 4 oranges. How much does she spend in total? "
            "Show your working."
        ),
        expected_length="medium",
        note="Basic arithmetic with working shown.",
    ),
    Prompt(
        id="reason_02",
        category="reasoning",
        text=(
            "If all cats are animals, and some animals are domestic, "
            "can we conclude that some cats are domestic? "
            "Explain your reasoning carefully."
        ),
        expected_length="medium",
        note="Syllogistic logic — common trap.",
    ),
    Prompt(
        id="reason_03",
        category="reasoning",
        text=(
            "A water tank is 3/4 full. After removing 120 litres it is 1/2 full. "
            "What is the total capacity of the tank in litres? Show your steps."
        ),
        expected_length="medium",
        note="Fraction arithmetic with algebraic reasoning.",
    ),
    Prompt(
        id="reason_04",
        category="reasoning",
        text=(
            "You have a 3-litre jug and a 5-litre jug, and an unlimited water supply. "
            "How do you measure exactly 4 litres? List the steps."
        ),
        expected_length="medium",
        note="Classic constraint-satisfaction puzzle.",
    ),
    Prompt(
        id="reason_05",
        category="reasoning",
        text=(
            "A train leaves City A at 08:00 travelling at 90 km/h toward City B. "
            "Another train leaves City B at 09:00 travelling at 110 km/h toward City A. "
            "The cities are 400 km apart. At what time do the trains meet? Show your working."
        ),
        expected_length="long",
        note="Multi-step kinematics problem.",
    ),

    # -----------------------------------------------------------------------
    # CODING (5)
    # -----------------------------------------------------------------------
    Prompt(
        id="code_01",
        category="coding",
        text=(
            "Write a Python function `flatten(lst)` that takes a nested list of "
            "arbitrary depth and returns a flat list. Include type hints and a docstring."
        ),
        expected_length="medium",
        note="Recursive function. Tests correctness and style.",
    ),
    Prompt(
        id="code_02",
        category="coding",
        text=(
            "Write a Python function `count_words(text: str) -> dict[str, int]` "
            "that returns a dictionary of word frequencies, case-insensitive, "
            "ignoring punctuation. Include a docstring and one example."
        ),
        expected_length="medium",
        note="String processing. Tests stdlib knowledge.",
    ),
    Prompt(
        id="code_03",
        category="coding",
        text=(
            "Write a Python decorator `@retry(max_attempts: int)` that retries "
            "a function up to max_attempts times if it raises an exception, "
            "with a 1-second delay between attempts. Include usage example."
        ),
        expected_length="long",
        note="Decorator pattern. Tests advanced Python knowledge.",
    ),
    Prompt(
        id="code_04",
        category="coding",
        text=(
            "Write a Python function `binary_search(arr: list[int], target: int) -> int` "
            "that returns the index of target in a sorted list, or -1 if not found. "
            "Use an iterative approach. Include docstring and two example calls."
        ),
        expected_length="medium",
        note="Classic algorithm. Tests iterative implementation.",
    ),
    Prompt(
        id="code_05",
        category="coding",
        text=(
            "Write a Python context manager class `Timer` that measures and prints "
            "the elapsed time of a code block when used with the `with` statement. "
            "Include a usage example."
        ),
        expected_length="medium",
        note="Context manager protocol. Tests __enter__/__exit__ knowledge.",
    ),

    # -----------------------------------------------------------------------
    # INSTRUCTION FOLLOWING (5)
    # -----------------------------------------------------------------------
    Prompt(
        id="instr_01",
        category="instruction",
        text=(
            "List exactly 5 programming languages. "
            "For each one, write exactly one sentence describing its primary use case. "
            "Format as a numbered list."
        ),
        expected_length="medium",
        note="Strict count + format compliance.",
    ),
    Prompt(
        id="instr_02",
        category="instruction",
        text=(
            "Explain what an API is. "
            "Your response must be exactly 3 sentences — no more, no less. "
            "Write for a non-technical audience."
        ),
        expected_length="short",
        note="Hard sentence count constraint. Tests instruction precision.",
    ),
    Prompt(
        id="instr_03",
        category="instruction",
        text=(
            "Translate the following into formal English, informal English, and bullet points:\n\n"
            "\"The meeting got pushed back so we gotta reschedule. "
            "Let me know when you're free next week.\""
        ),
        expected_length="medium",
        note="Three-format output. Tests multi-constraint following.",
    ),
    Prompt(
        id="instr_04",
        category="instruction",
        text=(
            "Write a haiku about machine learning. "
            "Then write a limerick about machine learning. "
            "Label each one clearly."
        ),
        expected_length="medium",
        note="Two constrained creative formats with labelling.",
    ),
    Prompt(
        id="instr_05",
        category="instruction",
        text=(
            "Respond to the following only using words that start with the letter S:\n\n"
            "What is the Sun?"
        ),
        expected_length="short",
        note="Unusual hard constraint. Tests strict rule compliance.",
    ),

    # -----------------------------------------------------------------------
    # SUMMARIZATION (5)
    # -----------------------------------------------------------------------
    Prompt(
        id="summ_01",
        category="summarization",
        text=(
            "Summarize the following in one sentence:\n\n"
            "\"Photosynthesis is the process used by plants, algae, and certain bacteria "
            "to convert light energy, usually from the sun, into chemical energy that can "
            "be later released to fuel the organism's activities. This process involves "
            "the absorption of carbon dioxide and water, which are converted into glucose "
            "and oxygen using light energy captured by chlorophyll.\""
        ),
        expected_length="short",
        note="Hard single-sentence compression.",
    ),
    Prompt(
        id="summ_02",
        category="summarization",
        text=(
            "Extract the three most important points from the following passage "
            "as a numbered list:\n\n"
            "\"The industrial revolution, beginning in Britain in the late 18th century, "
            "transformed manufacturing from hand production to machine-based processes. "
            "It led to the rise of factories, mass production, and urbanisation as workers "
            "moved from rural areas to cities. New energy sources such as coal and steam "
            "power drove this transformation, enabling faster transportation through railways "
            "and steamships. While it created enormous economic growth, it also produced "
            "difficult working conditions, child labour, and significant environmental pollution.\""
        ),
        expected_length="medium",
        note="Key point extraction with numbering constraint.",
    ),
    Prompt(
        id="summ_03",
        category="summarization",
        text=(
            "Rewrite the following paragraph at a reading level suitable for a 10-year-old, "
            "without losing the core meaning:\n\n"
            "\"Neural networks are computational models loosely inspired by the biological "
            "neural networks in animal brains. They consist of layers of interconnected nodes "
            "that process information using connectionist approaches to computation. "
            "Through a training process involving gradient descent and backpropagation, "
            "these networks learn to approximate complex functions mapping inputs to outputs.\""
        ),
        expected_length="medium",
        note="Register transformation. Tests paraphrasing ability.",
    ),
    Prompt(
        id="summ_04",
        category="summarization",
        text=(
            "Write an executive summary (maximum 60 words) of the following:\n\n"
            "\"Remote work has increased significantly since 2020. Studies show that "
            "productivity varies widely — some workers report higher output at home "
            "due to fewer interruptions, while others struggle with isolation and blurred "
            "work-life boundaries. Companies are responding with hybrid models that "
            "combine office presence with flexible remote days, attempting to balance "
            "collaboration needs with employee wellbeing.\""
        ),
        expected_length="short",
        note="Hard word count limit. Tests concise summarization.",
    ),
    Prompt(
        id="summ_05",
        category="summarization",
        text=(
            "Compare and contrast the following two concepts in exactly 4 bullet points "
            "(2 similarities, 2 differences):\n\n"
            "Supervised learning vs. Unsupervised learning"
        ),
        expected_length="medium",
        note="Structured comparison with exact bullet count.",
    ),

    # -----------------------------------------------------------------------
    # CREATIVE (5)
    # -----------------------------------------------------------------------
    Prompt(
        id="creative_01",
        category="creative",
        text=(
            "Write the opening paragraph of a science fiction short story "
            "set on a space station where time moves backwards. "
            "Around 80 words."
        ),
        expected_length="medium",
        note="Open-ended creative generation. Tests imaginative output.",
    ),
    Prompt(
        id="creative_02",
        category="creative",
        text=(
            "Write a product description for an imaginary device called "
            "the 'MemoryLens' — a pair of glasses that lets you replay "
            "any memory from your past in full sensory detail. "
            "Around 80 words, in a compelling marketing tone."
        ),
        expected_length="medium",
        note="Constrained creative with tone requirement.",
    ),
    Prompt(
        id="creative_03",
        category="creative",
        text=(
            "Write a short dialogue (6-8 lines) between a historian "
            "and an AI that has just become self-aware."
        ),
        expected_length="medium",
        note="Dialogue format. Tests character voice and format compliance.",
    ),
    Prompt(
        id="creative_04",
        category="creative",
        text=(
            "Write a metaphor that explains how a computer's CPU works "
            "to someone who has never used a computer. "
            "Make it vivid and original. Around 50 words."
        ),
        expected_length="short",
        note="Constrained creative analogy. Tests originality.",
    ),
    Prompt(
        id="creative_05",
        category="creative",
        text=(
            "Continue this story opening in the same style, adding exactly 2 more sentences:\n\n"
            "\"The last library on Earth smelled of cedar and forgotten passwords. "
            "No one came here anymore — except her.\""
        ),
        expected_length="short",
        note="Style-matching continuation. Tests stylistic coherence.",
    ),
]
