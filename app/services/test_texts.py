"""
Test texts service: persistent labelled text snippets for TTS testing.

Each entry has a label, full text (can be very long), and user-defined
string tags for filtering.  Stored server-side in test-texts.json.
Thread-safe via threading.Lock; atomic write on save.
"""

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.logging_config import get_logger

logger = get_logger('test_texts')


class TestTextsService:
    """Manages saved test text snippets with labels and string tags."""

    def __init__(self, path: str):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._data = self._load()

    # ─────────────────────────────────────────────────────── persistence ───

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    data = json.load(f)
                logger.info(f'Loaded {len(data.get("texts", {}))} test texts')
                return data
            except Exception as e:
                logger.error(f'Failed to load test-texts.json: {e}')
        return {'version': '1', 'texts': {}}

    def _save(self):
        """Atomic write via temp file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix('.tmp')
        try:
            with open(tmp, 'w') as f:
                json.dump(self._data, f, indent=2)
            tmp.replace(self._path)
        except Exception as e:
            logger.error(f'Failed to save test-texts.json: {e}')
            raise

    # ─────────────────────────────────────────────────────── public API ───

    def list_texts(self) -> list:
        """Return summaries (no full text body — keeps payload small)."""
        with self._lock:
            return [
                {
                    'id': tid,
                    'label': t['label'],
                    'tags': t.get('tags', []),
                    'word_count': len(t['text'].split()),
                    'preview': t['text'][:150].replace('\n', ' '),
                    'created': t.get('created', ''),
                }
                for tid, t in self._data.get('texts', {}).items()
            ]

    def get_text(self, text_id: str) -> Optional[dict]:
        """Return a single entry including the full text body."""
        with self._lock:
            t = self._data.get('texts', {}).get(text_id)
            if not t:
                return None
            return {'id': text_id, **t}

    def create_text(self, label: str, text: str, tags: list = None) -> dict:
        tid = str(uuid.uuid4())
        entry = {
            'label': label,
            'text': text,
            'tags': tags or [],
            'created': datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._data.setdefault('texts', {})[tid] = entry
            self._save()
        logger.info(f'Created test text "{label}" ({tid})')
        return {'id': tid, **entry}

    def update_text(self, text_id: str, label: str = None,
                    text: str = None, tags: list = None) -> Optional[dict]:
        with self._lock:
            entry = self._data.get('texts', {}).get(text_id)
            if not entry:
                return None
            if label is not None:
                entry['label'] = label
            if text is not None:
                entry['text'] = text
            if tags is not None:
                entry['tags'] = tags
            self._save()
        return {'id': text_id, **entry}

    def delete_text(self, text_id: str) -> bool:
        with self._lock:
            if text_id not in self._data.get('texts', {}):
                return False
            del self._data['texts'][text_id]
            self._save()
        logger.info(f'Deleted test text {text_id}')
        return True


_instance: Optional[TestTextsService] = None


def get_test_texts_service() -> TestTextsService:
    global _instance
    if _instance is None:
        from app.config import Config
        _instance = TestTextsService(Config.TEST_TEXTS_PATH)
    return _instance
