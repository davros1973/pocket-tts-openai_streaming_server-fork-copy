"""Audio Enhancement Service — LavaSR Bandwidth Extension + Denoising.

Applies LavaSR post-processing to TTS audio output for higher quality.

Third-party attribution
-----------------------
LavaSR — Copyright (c) 2024 Yatharth Sharma <yatharthsharma3501@gmail.com>
Apache License 2.0
https://github.com/ysharma3501/LavaSR
Full licence text: LICENSES/LICENSE-LAVASR

PocketTTS — MIT License — see LICENSE

MIT and Apache 2.0 are both permissive and fully compatible with each other
for private and commercial combined use.  The Apache 2.0 terms require that
the LavaSR copyright notice and licence file are preserved — both are
included in this repository under LICENSES/LICENSE-LAVASR and lavasr/LICENSE.
"""

import time

import torch
import torchaudio

from app.config import Config
from app.logging_config import get_logger

logger = get_logger('enhancer')


class EnhancerService:
    """Wraps LavaSR for post-processing TTS audio output.

    PocketTTS (Kyutai pocket-tts) outputs audio at 24 kHz.  LavaSR expects
    16 kHz input and produces 48 kHz output.

    Pipeline::

        TTS 24 kHz → resample → 16 kHz → LavaSR (denoise? + BWE) → 48 kHz

    The model is lazy-loaded on first use (or via an explicit load_model()
    call from a management endpoint).  Loading involves a one-time HuggingFace
    snapshot download (~150 MB) the first time; after that the HF cache is
    reused.
    """

    #: LavaSR always outputs 48 kHz regardless of input sample rate.
    OUTPUT_SR: int = 48000

    def __init__(self) -> None:
        self._loaded = False
        self.model = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        """True once the LavaSR model has been successfully loaded."""
        return self._loaded

    def load_model(self) -> None:
        """Load the LavaSR model weights from HuggingFace (or local path).

        The model path and device come from Config.LAVASR_MODEL /
        Config.LAVASR_DEVICE, which are set by the LAVASR_MODEL and
        LAVASR_DEVICE environment variables.

        Raises:
            ImportError: If the LavaSR package is not installed in the
                container (means the image needs to be rebuilt).
            Exception: On any model loading failure.
        """
        model_path = Config.LAVASR_MODEL
        device = Config.LAVASR_DEVICE
        try:
            from LavaSR.model import LavaEnhance  # noqa: PLC0415  (vendored)

            t0 = time.time()
            logger.info(f'Loading LavaSR model: {model_path!r} on device={device!r}')
            self.model = LavaEnhance(model_path=model_path, device=device)
            self._loaded = True
            logger.info(f'LavaSR ready in {time.time() - t0:.2f}s')
        except ImportError:
            logger.error(
                'LavaSR package not found inside the container. '
                'Rebuild the Docker image (it installs LavaSR from lavasr/).'
            )
            raise
        except Exception as exc:
            logger.error(f'LavaSR load failed: {exc}')
            raise

    def ensure_loaded(self) -> None:
        """Load the model if it has not been loaded yet (lazy load)."""
        if not self._loaded:
            self.load_model()

    # ── inference ──────────────────────────────────────────────────────────

    def enhance(
        self,
        audio: torch.Tensor,
        input_sr: int,
        do_enhance: bool = True,
        do_denoise: bool = False,
    ) -> tuple[torch.Tensor, int]:
        """Run LavaSR on a raw TTS audio tensor.

        Args:
            audio:      Input waveform, shape [T] or [1, T].
            input_sr:   Sample rate of ``audio`` (e.g. 24000 for PocketTTS).
            do_enhance: Apply bandwidth extension (BWE) — upsample to 48 kHz.
            do_denoise: Apply ULUNAS speech denoising before BWE.  Off by
                        default because TTS output is clean; enabling it adds
                        a small latency overhead.

        Returns:
            ``(enhanced_tensor [1, T], output_sample_rate)`` where
            ``output_sample_rate`` is always 48000.
        """
        self.ensure_loaded()

        # Normalise to 1-D float32 on CPU
        wav = audio.squeeze(0) if audio.dim() == 2 else audio
        wav = wav.float().cpu()

        # LavaSR was designed for 16 kHz speech; resample down first.
        # Note: going 24 kHz → 16 kHz discards some content above 8 kHz,
        # which the BWE model then reconstructs.  Net quality is empirically
        # positive on TTS output but a listening test is recommended.
        if input_sr != 16000:
            wav = torchaudio.functional.resample(wav, input_sr, 16000)

        # Add batch dimension → [1, T] as LavaSR expects
        wav = wav.unsqueeze(0)

        t0 = time.time()
        with torch.no_grad():
            enhanced = self.model.enhance(wav, enhance=do_enhance, denoise=do_denoise)
        logger.debug(f'LavaSR inference {time.time() - t0:.3f}s (enhance={do_enhance}, denoise={do_denoise})')

        # LavaSR.enhance() returns a 1-D tensor; reshape for torchaudio.save
        if enhanced.dim() == 1:
            enhanced = enhanced.unsqueeze(0)

        return enhanced.cpu(), self.OUTPUT_SR


# ── singleton ───────────────────────────────────────────────────────────────

_enhancer_service: EnhancerService | None = None


def get_enhancer_service() -> EnhancerService:
    """Return the process-wide EnhancerService singleton (created on first call)."""
    global _enhancer_service
    if _enhancer_service is None:
        _enhancer_service = EnhancerService()
    return _enhancer_service
