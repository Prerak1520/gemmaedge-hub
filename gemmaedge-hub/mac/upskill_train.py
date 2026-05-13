"""
upskill_train.py — GemmaEdge Hub
Uses Gemma 4 26B (teacher) on the Mac Mini to generate and score optimized
system prompts for Gemma 4 2B (student) on the edge device.

The winning prompt is saved to skill.txt — copy it to the edge device alongside sensor.py.

Usage:
    python upskill_train.py

This is the "teacher-student upskilling" approach:
  - No weight training required
  - Runs in ~5 minutes on the Mac Mini
  - Measurably improves edge model accuracy on your specific task
"""

import json
import logging
import time

import ollama

TEACHER_MODEL = "gemma4:26b"   # runs on Mac Mini — high quality prompt writer
STUDENT_MODEL = "gemma4:e2b"   # runs on edge device — we're optimizing for this model
OUTPUT_FILE   = "skill.txt"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

TASK = (
    "Identify objects, people, and any unusual or safety-relevant activity "
    "visible in a webcam image from a home or office environment. "
    "Be concise and specific."
)

# A small set of test prompts we use to evaluate candidate skills.
# In a real run you'd have more, ideally with real webcam images.
EVAL_CASES = [
    {
        "user": "Describe what you see. A person is sitting at a desk with a laptop open.",
        "ideal_keywords": ["person", "desk", "laptop", "sitting"],
    },
    {
        "user": "Describe what you see. The room appears empty with a door left open.",
        "ideal_keywords": ["empty", "door", "open"],
    },
    {
        "user": "Describe what you see. There is smoke visible near the kitchen area.",
        "ideal_keywords": ["smoke", "kitchen", "safety", "fire", "hazard"],
    },
]


def generate_candidate_skills(n: int = 4) -> list[str]:
    """Ask the teacher model to write N system prompts for the task."""
    log.info("Teacher (%s) generating %d candidate skills…", TEACHER_MODEL, n)

    prompt = f"""You are an expert at writing system prompts that make small language models
perform better on specific tasks.

Task the small model must do:
{TASK}

Write {n} different system prompts for this task. Each should be concise (2-4 sentences),
vary in style (one formal, one brief, one example-driven, one safety-focused).

Return ONLY a JSON array of strings, no extra text. Example format:
["prompt 1 text", "prompt 2 text"]"""

    response = ollama.chat(
        model=TEACHER_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response["message"]["content"].strip()

    # Extract JSON array from response
    start = text.find("[")
    end = text.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError(f"Teacher did not return a JSON array:\n{text}")

    skills = json.loads(text[start:end])
    log.info("Generated %d candidate skills", len(skills))
    return skills


def score_skill(skill: str) -> float:
    """
    Run the student model on each eval case using this skill as system prompt.
    Returns a score 0.0–1.0 based on keyword coverage.
    """
    total, hits = 0, 0

    for case in EVAL_CASES:
        response = ollama.chat(
            model=STUDENT_MODEL,
            messages=[{"role": "user", "content": case["user"]}],
            options={"system": skill},
        )
        answer = response["message"]["content"].lower()
        for keyword in case["ideal_keywords"]:
            total += 1
            if keyword in answer:
                hits += 1

    return hits / total if total > 0 else 0.0


def train() -> str:
    """Generate candidates, score each, return the best skill text."""
    candidates = generate_candidate_skills(n=4)

    best_skill, best_score = "", 0.0
    for i, skill in enumerate(candidates, 1):
        log.info("Scoring candidate %d/%d…", i, len(candidates))
        t0 = time.perf_counter()
        score = score_skill(skill)
        elapsed = time.perf_counter() - t0
        log.info("  Score: %.2f  (%.1fs)  — %s…", score, elapsed, skill[:60])

        if score > best_score:
            best_score = score
            best_skill = skill

    log.info("Best skill score: %.2f", best_score)
    return best_skill


if __name__ == "__main__":
    log.info("Starting upskill training — teacher=%s  student=%s", TEACHER_MODEL, STUDENT_MODEL)
    best = train()

    with open(OUTPUT_FILE, "w") as f:
        f.write(best)

    log.info("Saved to %s — copy this file to your edge device alongside sensor.py", OUTPUT_FILE)
    print("\n--- WINNING SKILL PROMPT ---")
    print(best)
    print("----------------------------\n")
    print(f"Now copy {OUTPUT_FILE} to your edge device:")
    print("  # Pi:          scp skill.txt pi@<PI_IP>:~/gemmaedge/skill.txt")
    print("  # MacBook Air: scp skill.txt <username>@<AIR_IP>:~/gemmaedge/skill.txt")
