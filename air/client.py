"""
HTTP client used by sensor.py to escalate requests to the Mac Mini server.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from shared.protocol import EdgeRequest, EdgeResponse

# Set MAC_URL in your MacBook Air's environment, e.g.:
#   export MAC_URL=http://mac-mini.local:8000
MAC_URL = os.environ.get("MAC_URL", "http://mac-mini.local:8000")
TIMEOUT = 180  # seconds — 26B model can take 2+ min on first inference


def escalate_to_mac(
    session_id: str,
    modality: str,
    prompt: str,
    image_b64: str | None = None,
    audio_b64: str | None = None,
    text: str | None = None,
    local_confidence: float = 0.0,
) -> EdgeResponse:
    """Send an EdgeRequest to the Mac and return the parsed EdgeResponse."""
    payload = EdgeRequest(
        session_id=session_id,
        modality=modality,
        image_b64=image_b64,
        audio_b64=audio_b64,
        text=text,
        local_confidence=local_confidence,
        prompt=prompt,
    )

    resp = httpx.post(
        f"{MAC_URL}/reason",
        json=payload.model_dump(),
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return EdgeResponse(**resp.json())
