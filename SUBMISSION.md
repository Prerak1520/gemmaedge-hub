---
title: GemmaEdge Hub: A Two-Device Local AI Vision System
published: false
tags: devchallenge, gemmachallenge, gemma
---

*This is a submission for the [Gemma 4 Challenge: Build with Gemma 4](https://dev.to/challenges/google-gemma-2026-05-06)*

## What I Built

GemmaEdge Hub is a two-device local AI vision system that keeps routine webcam analysis on an edge device and escalates harder cases to a stronger local server.

The edge device is a MacBook Air running `gemma4:e2b` through Ollama for fast, private visual inference. When a frame is uncertain, safety-relevant, or due for a periodic audit, the edge device sends that frame to a Mac Mini running a larger Gemma 4 model for deeper analysis. The Mac Mini also hosts a live FastAPI dashboard showing every escalation, the local answer, the stronger-model answer, confidence values, and latency.

The goal is to make local multimodal AI feel practical: small model first, bigger model only when it is worth the extra compute.

## Demo

The live demo runs across two Macs on the same local network:

1. The MacBook Air captures webcam frames.
2. `gemma4:e2b` gives a fast local answer with a confidence score.
3. The edge device keeps routine frames private.
4. The edge device escalates uncertain, safety-relevant, or audited frames.
5. The Mac Mini analyzes the escalated frame and updates the dashboard in real time.

Dashboard URL during the demo:

```text
http://<MINI_IP>:8000
```

## Code

Repository:

https://github.com/Prerak1520/gemmaedge-hub

The main pieces are:

- `air/sensor.py`: webcam capture, local Gemma 4 inference, escalation decisions
- `air/client.py`: HTTP client for sending escalations to the Mac Mini
- `mac/server.py`: FastAPI server, Gemma 4 server inference, live dashboard
- `mac/upskill_train.py`: teacher-student prompt optimization for the edge model
- `shared/protocol.py`: shared Pydantic request/response schema

## How I Used Gemma 4

Gemma 4 is the core of the project. I used two different model sizes intentionally:

- `gemma4:e2b` on the edge device because it is small enough to run locally and quickly on a MacBook Air.
- A larger Gemma 4 model on the Mac Mini because it can spend more time on harder visual reasoning after escalation.

The interesting part was discovering that self-reported confidence is not enough. In testing, the edge model often reported `CONFIDENCE: 1.0`, even when the answer still deserved review. I updated the escalation policy so it now considers:

- low confidence
- safety-relevant keywords like smoke, fire, hazard, injury, or emergency
- periodic audits of overconfident local answers with `AUDIT_EVERY_N_FRAMES`

That makes the system more realistic: the small model handles the common path, while the stronger model checks important or suspicious cases.

I also added a teacher-student upskilling step. The Mac Mini runs `upskill_train.py`, where the stronger model proposes and scores system prompts for the smaller edge model. The winning prompt is copied to the edge device as `skill.txt`, improving the small model's behavior without fine-tuning weights.

## What I Learned

The biggest design lesson was that model orchestration matters as much as model choice. A small local model is great for privacy and responsiveness, but it needs a good policy for knowing when to ask for help. A larger local model is powerful, but it is too slow and expensive to run on every frame.

GemmaEdge Hub combines both: private edge inference by default, stronger local reasoning when needed, and a dashboard that makes the escalation path visible.
