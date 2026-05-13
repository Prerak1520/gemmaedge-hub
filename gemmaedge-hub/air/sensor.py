"""
Edge Sensor — GemmaEdge Hub
Captures webcam frames, runs Gemma 4 2B locally via Ollama,
and escalates to the Mac Mini when confidence is too low.

Why Gemma 4 2B here:
  - Fits in ~3 GB RAM (MacBook Air has 8 GB)
  - Handles routine object ID quickly via Apple Neural Engine
  - No network round-trip for common cases — privacy stays local
"""

import base64
import io
import json
import logging
import sys
import time
import uuid
from pathlib import Path

import cv2
import ollama
from PIL import Image

from client import escalate_to_mac

# ── config ────────────────────────────────────────────────────────────────────
LOCAL_MODEL = "gemma4:2b"
ESCALATE_THRESHOLD = 0.55      # escalate to Mac if local confidence < this
CAMERA_INDEX = 0               # 0 = first USB webcam
SKILL_FILE = Path("skill.txt") # system prompt written by upskill_train.py on Mac

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_skill() -> str:
    """Load the optimized system prompt if one has been generated."""
    if SKILL_FILE.exists():
        return SKILL_FILE.read_text().strip()
    return "You are a helpful vision assistant running on a MacBook Air."


def capture_frame() -> bytes:
    """Grab one JPEG frame from the USB webcam."""
    cam = cv2.VideoCapture(CAMERA_INDEX)
    if not cam.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")

    time.sleep(0.3)            # let auto-exposure settle
    ok, frame = cam.read()
    cam.release()

    if not ok:
        raise RuntimeError("Failed to capture frame")

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()


def local_inference(
    prompt: str,
    system: str,
    image_bytes: bytes | None = None,
) -> tuple[str, float]:
    """
    Run Gemma 4 2B via Ollama.
    Returns (answer_text, confidence_0_to_1).
    """
    user_content = (
        f"{prompt}\n\n"
        "After your answer, on a new line write exactly: CONFIDENCE: <number 0.0-1.0>"
    )

    message: dict = {"role": "user", "content": user_content}
    if image_bytes:
        message["images"] = [image_bytes]

    response = ollama.chat(
        model=LOCAL_MODEL,
        messages=[message],
        options={"system": system},
    )
    text = response["message"]["content"]

    confidence = 0.5
    lines = text.strip().splitlines()
    if lines and lines[-1].upper().startswith("CONFIDENCE:"):
        try:
            confidence = float(lines[-1].split(":")[1].strip())
            text = "\n".join(lines[:-1]).strip()
        except ValueError:
            pass

    return text, confidence


# ── main loop ─────────────────────────────────────────────────────────────────

def run() -> None:
    log.info("GemmaEdge edge sensor starting — model=%s", LOCAL_MODEL)
    system_prompt = load_skill()
    log.info("System prompt loaded (%d chars)", len(system_prompt))

    while True:
        session_id = str(uuid.uuid4())[:8]

        try:
            log.info("[%s] Capturing frame…", session_id)
            image_bytes = capture_frame()

            prompt = (
                "What do you see in this image? "
                "Describe objects, people, and any unusual or safety-relevant activity."
            )
            answer, confidence = local_inference(prompt, system_prompt, image_bytes)
            image_b64 = base64.b64encode(image_bytes).decode()

            log.info("[%s] Local answer (conf=%.2f): %s", session_id, confidence, answer)

            if confidence < ESCALATE_THRESHOLD:
                log.info("[%s] Low confidence — escalating to Mac…", session_id)
                mac_response = escalate_to_mac(
                    session_id=session_id,
                    modality="vision",
                    prompt=prompt,
                    image_b64=image_b64,
                    text=answer,
                    local_confidence=confidence,
                )
                log.info(
                    "[%s] Mac answer (conf=%.2f): %s",
                    session_id, mac_response.confidence, mac_response.answer,
                )
            else:
                log.info("[%s] Local result accepted.", session_id)

        except KeyboardInterrupt:
            log.info("Stopped.")
            sys.exit(0)
        except Exception as exc:
            log.error("[%s] Error: %s", session_id, exc)

        time.sleep(3)


if __name__ == "__main__":
    run()
