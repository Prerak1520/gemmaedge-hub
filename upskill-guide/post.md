# Two Ways to Make a Small Model Smarter for Edge Deployment (No GPU Required)

*A practical guide using Gemma 4 on a Mac Mini + Raspberry Pi 4*

---

When you deploy a small model to an edge device like a Raspberry Pi, you quickly hit a problem: the model is fast and private, but it gets confused on hard cases. The obvious fix is fine-tuning — but that takes hours, a dataset, and real compute.

This guide shows two paths to improve your edge model, ordered from fast to thorough:

1. **Prompt upskilling** — use a bigger model to write a better system prompt for the small one. 5 minutes, no training data needed.
2. **QLoRA fine-tuning** — adjust the model's actual weights on your Mac Mini. 1–2 hours, needs labeled examples.

Both approaches are real and useful. Start with path 1. Only go to path 2 if you need more.

---

## The Setup

**Hardware:**
- Mac Mini (24 GB unified RAM) — runs Gemma 4 26B as the "smart" model
- Raspberry Pi 4 (4 or 8 GB RAM) — runs Gemma 4 2B as the "edge" model
- USB webcam connected to the Pi

**Why these model sizes?**
- Gemma 4 2B fits in ~3 GB RAM, runs at 2–4 tokens/sec on Pi CPU — fast enough for real-time use
- Gemma 4 26B (MoE) fits in ~18 GB RAM on the Mac, uses Apple Metal for ~15–20 tok/s
- The 256K context window on the larger model lets it reason over full scene descriptions

Both run locally via [Ollama](https://ollama.com) — no cloud, no API keys.

---

## Install Ollama and Pull the Models

On both machines, install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

On the **Mac Mini**, pull the large model:

```bash
ollama pull gemma4:26b
```

On the **Pi**, pull the small model:

```bash
ollama pull gemma4:2b
```

Both will take a few minutes depending on your connection.

---

## Path 1: Prompt Upskilling with a Teacher Model

The idea is simple: ask the big model (teacher) to write an optimized system prompt for the small model (student), then score each candidate by testing it against real inputs.

This is the spirit behind the [huggingface/upskill](https://github.com/huggingface/upskill) library — use a capable model to generate skills that a cheaper model can rely on.

Here's the core of `upskill_train.py`:

```python
def generate_candidate_skills(n: int = 4) -> list[str]:
    """Ask Gemma 27B to write N system prompts for our webcam task."""
    prompt = f"""Write {n} system prompts for this task:
    Identify objects, people, and safety-relevant activity in a webcam image.
    Return a JSON array of strings."""

    response = ollama.chat(model="gemma4:26b", messages=[{"role": "user", "content": prompt}])
    return json.loads(extract_json(response["message"]["content"]))


def score_skill(skill: str) -> float:
    """Test each candidate against eval cases using Gemma 2B as student."""
    hits, total = 0, 0
    for case in EVAL_CASES:
        response = ollama.chat(
            model="gemma4:2b",
            messages=[{"role": "user", "content": case["user"]}],
            options={"system": skill},
        )
        answer = response["message"]["content"].lower()
        for keyword in case["ideal_keywords"]:
            total += 1
            if keyword in answer:
                hits += 1
    return hits / total
```

Run it on the Mac:

```bash
python upskill_train.py
```

It takes about 5 minutes. The winning system prompt is saved to `skill.txt`. Copy it to the Pi:

```bash
scp skill.txt pi@<PI_IP>:~/gemmaedge/skill.txt
```

The Pi's `sensor.py` loads it automatically on startup. In testing, this improved keyword coverage from ~55% to ~78% on our eval set — with zero training.

---

## Path 2: QLoRA Fine-Tuning on the Mac Mini

When prompt upskilling isn't enough, you can fine-tune the model's weights. On 24 GB of unified RAM you can comfortably run QLoRA on Gemma 4 2B in fp16.

**What you need first:** a labeled dataset. Each example is a prompt + ideal response:

```jsonl
{"prompt": "What do you see? A person near an open window.", "response": "A person is standing near an open window. This may be a safety concern if on a high floor."}
{"prompt": "What do you see? A laptop and a coffee mug on a desk.", "response": "A laptop computer and a coffee mug are on a desk. No safety concerns."}
```

20–50 examples is enough for style adaptation. 200+ for task-specific accuracy.

**Run the fine-tune:**

```bash
python finetune.py --dataset my_data.jsonl --output ./gemma2b-finetuned
```

Key settings in `finetune.py`:
- `r=16` (LoRA rank) — low enough to fit in RAM, high enough to learn
- `per_device_train_batch_size=2` + `gradient_accumulation_steps=4` — effective batch of 8
- 3 epochs takes about 90 minutes on Mac Mini with M-series chip

**Export to GGUF and load on Pi:**

```bash
# Install llama.cpp conversion tool
pip install llama-cpp-python

# Convert
python convert_hf_to_gguf.py ./gemma2b-finetuned --outfile gemma2b-custom.gguf

# Copy to Pi
scp gemma2b-custom.gguf pi@<PI_IP>:~/.ollama/models/

# Create Ollama modelfile on Pi
echo 'FROM gemma2b-custom.gguf' > Modelfile
ollama create gemma4-custom -f Modelfile
```

Update `LOCAL_MODEL = "gemma4-custom"` in `sensor.py` and restart.

---

## What the System Looks Like Running

The Mac server runs at `http://mac-mini.local:8000`. Open that in a browser and you get a live dashboard showing:

- Every frame the Pi analyzed
- The Pi's local answer and confidence score
- Whether it escalated (and why)
- The Mac's more detailed response
- Round-trip latency

When confidence is above 0.55, the answer stays local — no network call, instant response, fully private. When it drops below that threshold, the Pi sends the frame to the Mac for deeper reasoning.

---

## Results

After running `upskill_train.py` and updating the system prompt:

| Metric | Before (default prompt) | After (upskilled prompt) |
|--------|------------------------|--------------------------|
| Keyword coverage | 55% | 78% |
| False escalations | ~40% of frames | ~18% of frames |
| Avg latency (local) | 1.8s | 1.6s |

The model didn't change. Only the system prompt did. That's the point.

---

## Key Takeaway

A better system prompt, written by a smarter model, is often 80% of the improvement at 0% of the training cost. Fine-tuning is the right tool when you have labeled data and need the model to genuinely learn new behavior — not just be guided differently.

Start with upskilling. Add fine-tuning if you need it.

---

*Full code: [github.com/Prerak1520/gemmaedge-hub](https://github.com/Prerak1520/gemmaedge-hub)*
*Hardware: Mac Mini M4 (24 GB), Raspberry Pi 4 (4 GB), USB webcam*
