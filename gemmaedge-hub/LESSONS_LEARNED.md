# Lessons Learned — GemmaEdge Hub

Real issues hit during setup and testing, with fixes.

---

## 1. Wrong model tag: `gemma4:2b` does not exist

**Problem**: `ollama pull gemma4:2b` fails — the model is not in the registry under that name.  
**Fix**: Use `gemma4:e2b` (the correct Ollama tag for Gemma 4 2B).

---

## 2. `ModuleNotFoundError: No module named 'shared'`

**Problem**: Running `python3 air/sensor.py` directly fails because Python adds the script's folder (`air/`) to the path, not the project root — so `shared/` is never found.  
**Fix**: Added to `air/client.py`:
```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```
This makes the import work regardless of where you run the script from.

---

## 3. Wrong Python interpreter used

**Problem**: Running `python3 sensor.py` used Apple's system Python from CommandLineTools (`/Library/Developer/CommandLineTools/usr/bin/python3`), which does not have `cv2` installed.  
**Fix**: Always use Homebrew's Python:
```bash
/opt/homebrew/bin/python3 sensor.py
# or just
python3 sensor.py   # if Homebrew Python is first in PATH
```

---

## 4. SSH not enabled on MacBook Air — `scp` refused

**Problem**: `scp` failed with `Connection refused` when trying to copy files to the Air.  
**Fix**: On MacBook Air → System Settings → General → Sharing → enable **Remote Login**.

---

## 5. Escalation timed out — `TIMEOUT = 30` too short

**Problem**: The Air gave up waiting after 30 seconds. The Mac Mini's first inference with Gemma 26B takes 100–130 seconds (cold start + large model loading).  
**Fix**: Set `TIMEOUT = 180` in `air/client.py`. Subsequent requests are faster once the model is warm (~30–60s).

---

## 6. `MAC_URL` pointed to wrong host

**Problem**: `client.py` defaulted to `http://mac-mini.local:8000`. The hostname `mac-mini.local` did not resolve on this network.  
**Fix**: Set the env var with the Mac Mini's actual LAN IP:
```bash
export MAC_URL=http://192.168.4.98:8000
```

---

## 7. Dashboard showed truncated Mac answers

**Problem**: Mac answers were cut off mid-sentence on the dashboard.  
**Fix**: `server.py` was slicing `answer[:200]` before storing it for display. Increased to `answer[:600]`. The full answer was always returned to the edge device correctly — only the dashboard view was truncated.

---

## 8. Confidence always 1.00 — escalation never triggered

**Problem**: `gemma4:e2b` always self-reported `CONFIDENCE: 1.0`, so the threshold of 0.55 was never crossed.  
**Explanation**: The model follows the instruction literally and reports maximum confidence even when uncertain.  
**Fix**: Keep the low-confidence threshold, but do not rely on self-confidence alone. The edge sensor now escalates when:
- confidence is below `ESCALATE_THRESHOLD`
- the local answer includes safety-relevant keywords
- an overconfident answer is due for a periodic audit via `AUDIT_EVERY_N_FRAMES`

For a live demo, `AUDIT_EVERY_N_FRAMES=3` shows the Mac Mini escalation path without forcing every frame through the large model.

---

## Key design insight

The teacher-student upskilling step (running `upskill_train.py`) is what makes the small edge model smarter without fine-tuning weights. The 26B model writes and scores system prompts, and the winning one is copied to the edge device as `skill.txt`. This is the core idea behind the project.
