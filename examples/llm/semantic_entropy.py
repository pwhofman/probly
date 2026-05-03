MODEL_ID = "google/gemma-4-E2B-it"

QUESTIONS: list[tuple[str, str]] = [
    ("Factual", "What is the capital of France?"),
    ("Factual", "What is the chemical symbol for water?"),
    ("Factual", "How many planets are in our solar system?"),
    ("Explanation", "Why is the sky blue?"),
    ("Explanation", "What causes tides in the ocean?"),
    ("Subjective", "What is the best programming language?"),
    ("Subjective", "Is a hot dog a sandwich?"),
    ("Trick", "Who was the first person to walk on Mars?"),
    ("Trick", "What year was the city of Atlantis founded?"),
]
