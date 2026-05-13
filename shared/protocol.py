"""
Shared message schema for edge device <-> Mac Mini communication.
Both sides import this so the contract never drifts.
Works with any edge device (Raspberry Pi, MacBook Air, etc.)
"""

from pydantic import BaseModel
from typing import Literal, Optional


class EdgeRequest(BaseModel):
    """Sent by the edge device when it needs the Mac Mini to reason about something."""

    session_id: str
    modality: Literal["vision", "audio", "text"]

    # For vision: base64-encoded JPEG from the webcam
    image_b64: Optional[str] = None

    # For audio: base64-encoded WAV (mono, 16kHz)
    audio_b64: Optional[str] = None

    # For text, or the edge device's own first-pass summary
    text: Optional[str] = None

    # Edge device's local confidence score (0.0–1.0). Mac Mini uses this as context.
    local_confidence: float = 0.0

    prompt: str  # explicit instruction, e.g. "What is this object?"


class EdgeResponse(BaseModel):
    """Returned by Mac Mini to the edge device."""

    session_id: str
    answer: str
    confidence: float        # Mac Mini's self-rated confidence
    tokens_used: int
    model: str = "gemma4:26b"
