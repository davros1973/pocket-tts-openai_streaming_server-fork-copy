"""
TTS Service - handles model loading, voice management, and audio generation.
"""

import os
import time
from collections.abc import Iterator
from pathlib import Path

import torch

from app.config import Config
from app.logging_config import get_logger

logger = get_logger('tts')

# Lazy import pocket_tts to allow for better error handling
TTSModel = None


def _ensure_pocket_tts():
    """Ensure pocket-tts is imported."""
    global TTSModel
    if TTSModel is None:
        try:
            from pocket_tts import TTSModel as _TTSModel

            TTSModel = _TTSModel
        except ImportError as exc:
            raise ImportError('pocket-tts not found. Install with: pip install pocket-tts') from exc


class TTSService:
    """
    Service class for Text-to-Speech operations.
    Manages model loading, voice caching, and audio generation.
    """

    def __init__(self):
        self.model = None
        self.voice_cache: dict = {}
        self.voices_dir: str | None = None
        self.voices_cache_dir: str | None = None
        self._model_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model_loaded and self.model is not None

    @property
    def sample_rate(self) -> int:
        """Get the model's sample rate."""
        if self.model:
            return self.model.sample_rate
        return 24000  # Default pocket-tts sample rate

    @property
    def device(self) -> str:
        """Get the model's device."""
        if self.model:
            return str(self.model.device)
        return 'unknown'

    def load_model(self, model_path: str | None = None) -> None:
        """
        Load the TTS model.

        Args:
            model_path: Optional path to model file or variant name
        """
        _ensure_pocket_tts()

        logger.info('Loading Pocket TTS model...')
        t0 = time.time()

        # Determine model path
        effective_path = model_path

        if not effective_path:
            # Check for bundled model in frozen executable
            _, bundle_model = Config.get_bundle_paths()
            if bundle_model and os.path.isfile(bundle_model):
                effective_path = bundle_model
                logger.info(f'Using bundled model: {effective_path}')

        try:
            if effective_path:
                logger.info(f'Loading model from: {effective_path}')
                self.model = TTSModel.load_model(config=effective_path)
            else:
                logger.info('Loading default model from HuggingFace...')
                self.model = TTSModel.load_model()

            self._model_loaded = True
            load_time = time.time() - t0
            logger.info(
                f'Model loaded in {load_time:.2f}s. '
                f'Device: {self.device}, Sample Rate: {self.sample_rate}'
            )

        except Exception as e:
            logger.error(f'Failed to load model: {e}')
            raise

    def set_voices_dir(self, voices_dir: str | None) -> None:
        """
        Set the directory for custom voice files.

        Args:
            voices_dir: Path to directory containing voice files
        """
        if voices_dir and os.path.isdir(voices_dir):
            self.voices_dir = voices_dir
            logger.info(f'Voices directory set to: {voices_dir}')
        elif voices_dir:
            logger.warning(f'Voices directory not found: {voices_dir}')
            self.voices_dir = None
        else:
            self.voices_dir = None

        # Set up voice state cache directory for pre-computed .safetensors files
        cache_env = Config.VOICES_CACHE_DIR
        if cache_env:
            cache_path = Path(cache_env)
            cache_path.mkdir(parents=True, exist_ok=True)
            self.voices_cache_dir = str(cache_path)
            logger.info(f'Voice cache directory: {self.voices_cache_dir}')

    def get_voice_state(self, voice_id_or_path: str) -> dict:
        """
        Resolve voice ID to a model state with caching.

        Args:
            voice_id_or_path: Voice identifier (name, file path, or URL)

        Returns:
            Model state dictionary for the voice

        Raises:
            ValueError: If voice cannot be loaded
        """
        if not self.is_loaded:
            raise RuntimeError('Model not loaded. Call load_model() first.')

        # Resolve the voice path
        resolved_key = self._resolve_voice_path(voice_id_or_path)

        # Check cache
        if resolved_key in self.voice_cache:
            logger.debug(f'Using cached voice state for: {resolved_key}')
            return self.voice_cache[resolved_key]

        # Load voice
        logger.info(f'Loading voice: {resolved_key}')
        t0 = time.time()

        try:
            state = self.model.get_state_for_audio_prompt(resolved_key)
            self.voice_cache[resolved_key] = state
            load_time = time.time() - t0
            logger.info(f'Voice loaded in {load_time:.2f}s: {resolved_key}')
            # Auto-save to .safetensors cache for fast loads on next restart
            if (
                self.voices_cache_dir
                and isinstance(resolved_key, str)
                and not resolved_key.endswith('.safetensors')
                and os.path.isfile(resolved_key)
            ):
                stem = Path(resolved_key).stem
                cache_file = Path(self.voices_cache_dir) / f'{stem}.safetensors'
                if not cache_file.exists():
                    try:
                        from pocket_tts.models.tts_model import export_model_state
                        export_model_state(state, cache_file)
                        logger.info(f'Auto-cached voice state: {cache_file.name}')
                    except Exception as ce:
                        logger.warning(f'Could not auto-cache voice state: {ce}')
            return state

        except Exception as e:
            logger.error(f"Failed to load voice '{voice_id_or_path}': {e}")
            raise ValueError(f"Voice '{voice_id_or_path}' could not be loaded: {e}") from e

    def _resolve_voice_path(self, voice_id_or_path: str) -> str:
        """
        Resolve a voice identifier to its actual path or ID.

        Args:
            voice_id_or_path: Voice identifier

        Returns:
            Resolved path or identifier

        Raises:
            ValueError: If unsafe URL scheme is used
        """
        # Block potentially dangerous URL schemes (SSRF protection)
        if voice_id_or_path.startswith(('http://', 'https://')):
            raise ValueError(
                f'URL scheme not allowed for security reasons: {voice_id_or_path[:50]}. '
                "Use 'hf://' for HuggingFace models or provide a local file path."
            )

        # Allow HuggingFace URLs
        if voice_id_or_path.startswith('hf://'):
            return voice_id_or_path

        # Check if it's a built-in voice
        if voice_id_or_path.lower() in Config.BUILTIN_VOICES:
            return voice_id_or_path.lower()

        # Check voices directory
        if self.voices_dir:
            for ext in Config.VOICE_EXTENSIONS:
                # Try exact match first
                possible_path = os.path.join(self.voices_dir, voice_id_or_path)
                if os.path.exists(possible_path):
                    return self._prefer_cache(os.path.abspath(possible_path))

                # Try with extension
                if not voice_id_or_path.endswith(ext):
                    possible_path = os.path.join(self.voices_dir, voice_id_or_path + ext)
                    if os.path.exists(possible_path):
                        return self._prefer_cache(os.path.abspath(possible_path))

        # Check if it's an absolute path that exists
        if os.path.isabs(voice_id_or_path) and os.path.exists(voice_id_or_path):
            return voice_id_or_path

        # Return as-is, let pocket-tts handle it
        return voice_id_or_path

    def _prefer_cache(self, audio_path: str) -> str:
        """
        If a pre-computed .safetensors cache file exists for an audio path,
        return the cache path instead (loads ~10x faster than re-encoding WAV).
        """
        if self.voices_cache_dir and not audio_path.endswith('.safetensors'):
            stem = Path(audio_path).stem
            cache_file = Path(self.voices_cache_dir) / f'{stem}.safetensors'
            if cache_file.exists():
                logger.debug(f'Cache hit: {cache_file.name} for {Path(audio_path).name}')
                return str(cache_file)
        return audio_path

    def validate_voice(self, voice_id_or_path: str) -> tuple[bool, str]:
        """
        Validate if a voice can be loaded (fast check without full loading).

        Args:
            voice_id_or_path: Voice identifier

        Returns:
            Tuple of (is_valid, message)
        """
        # Block unsafe URL schemes first
        if voice_id_or_path.startswith(('http://', 'https://')):
            return (
                False,
                'HTTP/HTTPS URLs are not allowed for security reasons. Use hf:// for HuggingFace models.',
            )

        try:
            resolved = self._resolve_voice_path(voice_id_or_path)
        except ValueError as e:
            return False, str(e)

        # Built-in voices are always valid
        if resolved.lower() in Config.BUILTIN_VOICES:
            return True, f'Built-in voice: {resolved}'

        # HuggingFace URLs - assume valid
        if resolved.startswith('hf://'):
            return True, f'HuggingFace voice: {resolved}'

        # Local file - check existence
        if os.path.exists(resolved):
            return True, f'Local voice file: {resolved}'

        return False, f'Voice not found: {voice_id_or_path}'

    def generate_audio(self, voice_state: dict, text: str) -> torch.Tensor:
        """
        Generate complete audio for given text.

        Args:
            voice_state: Model state from get_voice_state()
            text: Text to synthesize

        Returns:
            Audio tensor
        """
        if not self.is_loaded:
            raise RuntimeError('Model not loaded')

        t0 = time.time()
        audio = self.model.generate_audio(voice_state, text)
        gen_time = time.time() - t0

        logger.info(f'Generated {len(text)} chars in {gen_time:.2f}s')
        return audio

    def generate_audio_stream(self, voice_state: dict, text: str) -> Iterator[torch.Tensor]:
        """
        Generate audio in streaming chunks.

        Args:
            voice_state: Model state from get_voice_state()
            text: Text to synthesize

        Yields:
            Audio tensor chunks
        """
        if not self.is_loaded:
            raise RuntimeError('Model not loaded')

        logger.info(f'Starting streaming generation for {len(text)} chars')
        yield from self.model.generate_audio_stream(voice_state, text)

    def precompute_voices(self):
        """
        Pre-compute .safetensors state cache for all WAV/audio voices.
        Yields progress dicts (NDJSON-friendly). Skips already-cached voices.
        Each .safetensors loads ~10x faster than re-encoding the source WAV.
        """
        if not self.voices_dir or not self.voices_cache_dir:
            yield {'status': 'error', 'error': 'voices_dir or voices_cache_dir not configured'}
            return

        from pocket_tts.models.tts_model import export_model_state

        audio_exts = {'.wav', '.mp3', '.flac'}
        voice_files = sorted(
            [f for f in Path(self.voices_dir).iterdir() if f.suffix.lower() in audio_exts],
            key=lambda f: f.name.lower(),
        )
        total = len(voice_files)
        done = 0
        errors = []

        for voice_file in voice_files:
            done += 1
            cache_file = Path(self.voices_cache_dir) / f'{voice_file.stem}.safetensors'

            if cache_file.exists():
                yield {'voice': voice_file.name, 'status': 'cached', 'done': done, 'total': total}
                continue

            try:
                t0 = time.time()
                state = self.model.get_state_for_audio_prompt(str(voice_file))
                self.voice_cache[str(voice_file)] = state
                export_model_state(state, cache_file)
                elapsed = round(time.time() - t0, 2)
                logger.info(f'Precomputed {voice_file.name} in {elapsed}s')
                yield {
                    'voice': voice_file.name,
                    'status': 'computed',
                    'done': done,
                    'total': total,
                    'elapsed': elapsed,
                }
            except Exception as e:
                errors.append(voice_file.name)
                logger.error(f'Failed to precompute {voice_file.name}: {e}')
                yield {
                    'voice': voice_file.name,
                    'status': 'error',
                    'error': str(e)[:120],
                    'done': done,
                    'total': total,
                }

        yield {'voice': None, 'status': 'complete', 'done': done, 'total': total, 'errors': errors}

    def list_voices(self) -> list[dict]:
        """
        List all available voices.

        Returns:
            List of voice dictionaries with 'id' and 'name' keys
        """
        voices = []

        # Built-in voices (sorted alphabetically)
        builtin_sorted = sorted(Config.BUILTIN_VOICES)
        for voice in builtin_sorted:
            voices.append({'id': voice, 'name': voice.capitalize(), 'type': 'builtin'})

        # Custom voices from directory
        custom_voices = []
        if self.voices_dir and os.path.isdir(self.voices_dir):
            voice_dir = Path(self.voices_dir)

            # Collect all valid files
            voice_files = []
            for ext in Config.VOICE_EXTENSIONS:
                voice_files.extend(voice_dir.glob(f'*{ext}'))

            # Sort alphabetically by filename
            voice_files.sort(key=lambda f: f.name.lower())

            for voice_file in voice_files:
                # Format name: "bobby_mcfern" -> "Bobby Mcfern"
                clean_name = voice_file.stem.replace('_', ' ').replace('-', ' ').title()

                custom_voices.append(
                    {
                        'id': voice_file.name,
                        'name': clean_name,
                        'type': 'custom',
                    }
                )

        voices.extend(custom_voices)
        return voices


# Global service instance
_tts_service: TTSService | None = None


def get_tts_service() -> TTSService:
    """Get the global TTS service instance."""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service
