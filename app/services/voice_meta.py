"""
Voice metadata service: persistent tags, hidden state, and notes for voices.

Stored server-side in voice-meta.json (never local storage).
Tags are keyed by UUID so names are just properties — safe to rename without
breaking assignments.  Thread-safe via threading.Lock.
"""

import json
import threading
import uuid
from pathlib import Path
from typing import Optional

from app.logging_config import get_logger

logger = get_logger('voice_meta')


class VoiceMetaService:
    """Manages per-voice metadata (tags, hidden state, notes) and tag catalogue."""

    def __init__(self, meta_path: str):
        self._path = Path(meta_path)
        self._lock = threading.Lock()
        self._data = self._load()

    # ─────────────────────────────────────────────────────── persistence ───

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    data = json.load(f)
                logger.info(f'Loaded voice metadata ({len(data.get("tags", {}))} tags, '
                            f'{len(data.get("voices", {}))} voice entries)')
                return data
            except Exception as e:
                logger.error(f'Failed to load voice-meta.json: {e}')
        return {'version': '1', 'tags': {}, 'voices': {}}

    def _save(self):
        """Atomic write via temp file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix('.tmp')
        try:
            with open(tmp, 'w') as f:
                json.dump(self._data, f, indent=2)
            tmp.replace(self._path)
        except Exception as e:
            logger.error(f'Failed to save voice-meta.json: {e}')
            raise

    # ──────────────────────────────────────────────────────────── tags ───

    def list_tags(self) -> list:
        with self._lock:
            return [
                {'id': tid, **tdata}
                for tid, tdata in self._data.get('tags', {}).items()
            ]

    def create_tag(self, name: str, color: str = '#5c6ef8',
                   parent: Optional[str] = None) -> dict:
        tid = str(uuid.uuid4())
        with self._lock:
            self._data.setdefault('tags', {})[tid] = {
                'name': name,
                'color': color,
                'parent': parent,
            }
            self._save()
        logger.info(f'Created tag "{name}" ({tid})')
        return {'id': tid, 'name': name, 'color': color, 'parent': parent}

    def update_tag(self, tag_id: str, name: Optional[str] = None,
                   color: Optional[str] = None,
                   parent: Optional[str] = None) -> Optional[dict]:
        with self._lock:
            tag = self._data.get('tags', {}).get(tag_id)
            if tag is None:
                return None
            if name is not None:
                tag['name'] = name
            if color is not None:
                tag['color'] = color
            if parent is not None:
                tag['parent'] = parent
            self._save()
            return {'id': tag_id, **tag}

    def delete_tag(self, tag_id: str) -> bool:
        with self._lock:
            tags = self._data.get('tags', {})
            if tag_id not in tags:
                return False
            tag_name = tags[tag_id].get('name', tag_id)
            del tags[tag_id]
            # Unassign removed tag from all voices
            for voice_data in self._data.get('voices', {}).values():
                vtags = voice_data.get('tags', [])
                if tag_id in vtags:
                    vtags.remove(tag_id)
            # Clear parent reference in child tags
            for other in tags.values():
                if other.get('parent') == tag_id:
                    other['parent'] = None
            self._save()
        logger.info(f'Deleted tag "{tag_name}" ({tag_id})')
        return True

    # ──────────────────────────────────────────────────────── voice meta ───

    def get_voice_meta(self, voice_id: str) -> dict:
        with self._lock:
            entry = self._data.get('voices', {}).get(voice_id, {})
            return {
                'hidden': entry.get('hidden', False),
                'tags': list(entry.get('tags', [])),
                'note': entry.get('note', ''),
            }

    def set_voice_meta(self, voice_id: str, hidden=None, tags=None, note=None) -> dict:
        with self._lock:
            voices = self._data.setdefault('voices', {})
            entry = voices.setdefault(voice_id, {'hidden': False, 'tags': [], 'note': ''})
            if hidden is not None:
                entry['hidden'] = bool(hidden)
            if tags is not None:
                valid = set(self._data.get('tags', {}).keys())
                entry['tags'] = [t for t in tags if t in valid]
            if note is not None:
                entry['note'] = str(note)
            self._save()
            return dict(entry)

    def all_voice_meta(self) -> dict:
        with self._lock:
            return {k: dict(v) for k, v in self._data.get('voices', {}).items()}

    def full_dump(self) -> dict:
        """Return full snapshot — tags catalogue + per-voice entries."""
        with self._lock:
            return {
                'version': self._data.get('version', '1'),
                'tags': {tid: dict(tdata) for tid, tdata in self._data.get('tags', {}).items()},
                'voices': {vid: dict(vmeta) for vid, vmeta in self._data.get('voices', {}).items()},
            }

    def voice_count_for_tag(self, tag_id: str) -> int:
        with self._lock:
            return sum(
                1 for v in self._data.get('voices', {}).values()
                if tag_id in v.get('tags', [])
            )


# ─────────────────────────────────────────────────────── singleton ───

_meta_service: Optional[VoiceMetaService] = None


def get_voice_meta_service() -> VoiceMetaService:
    """Return (and lazily initialise) the global voice meta service."""
    global _meta_service
    if _meta_service is None:
        from app.config import Config
        _meta_service = VoiceMetaService(Config.VOICES_META_PATH)
    return _meta_service
