"""
Monitor service — in-memory ring buffer of TTS request events.

Buffers the last MAX_EVENTS requests and supports SSE fan-out so the Monitor
tab stays live without polling.  All access is thread-safe.

Event dict keys:
    ts              float   Unix timestamp (request start)
    voice           str     Voice ID requested
    text_preview    str     First 200 chars of input text
    text_full       str     Full input text
    ttfa            float   Seconds to first audio (batch: generation time; stream: first chunk)
    audio_duration  float   Seconds of audio generated (from sample count)
    gen_time        float   Wall-clock seconds for the whole generation
    mode            str     'stream' | 'batch'
"""

import json
import queue
import threading
from collections import deque


_MAX_EVENTS = 200
_lock = threading.Lock()
_events: deque = deque(maxlen=_MAX_EVENTS)
_subscribers: list[queue.Queue] = []


def record_event(event: dict) -> None:
    """Record a TTS request event and fan-out to all SSE subscribers."""
    with _lock:
        _events.append(event)
        dead: list[queue.Queue] = []
        for q in _subscribers:
            try:
                q.put_nowait(event)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _subscribers.remove(q)


def get_events() -> list[dict]:
    """Return all buffered events as a list, oldest first."""
    with _lock:
        return list(_events)


def subscribe() -> queue.Queue:
    """Register a new SSE subscriber.  Returns a Queue; call unsubscribe() when done."""
    q: queue.Queue = queue.Queue(maxsize=100)
    with _lock:
        _subscribers.append(q)
    return q


def unsubscribe(q: queue.Queue) -> None:
    """Remove a subscriber queue (called when SSE connection closes)."""
    with _lock:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass


def clear_events() -> None:
    """Discard all buffered events (used by the UI Clear button)."""
    with _lock:
        _events.clear()
