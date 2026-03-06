"""
Flask routes for the OpenAI-compatible TTS API.
"""

import json
import time

from flask import (
    Blueprint,
    Response,
    jsonify,
    render_template,
    request,
    send_file,
    stream_with_context,
)

from app.logging_config import get_logger
from app.services.audio import (
    convert_audio,
    get_mime_type,
    tensor_to_pcm_bytes,
    validate_format,
    write_wav_header,
)
from app.services.preprocess import TextPreprocessor
from app.services.tts import get_tts_service
from app.services.voice_meta import get_voice_meta_service
from app.services.test_texts import get_test_texts_service

logger = get_logger('routes')

# Create blueprint
api = Blueprint('api', __name__)

# Create text preprocessor instance, some options changed from defaults
text_preprocessor = TextPreprocessor(
    remove_urls=False,
    remove_emails=False,
    remove_html=True,
    remove_hashtags=True,
    remove_mentions=False,
    remove_punctuation=False,
    remove_stopwords=False,
    remove_extra_whitespace=False,
)


@api.route('/')
def home():
    """Serve the web interface."""
    from app.config import Config

    return render_template('index.html', is_docker=Config.IS_DOCKER)


@api.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint for container orchestration.

    Returns service status and basic model info.
    """
    tts = get_tts_service()

    # Validate a built-in voice quickly
    voice_valid, voice_msg = tts.validate_voice('alba')

    return jsonify(
        {
            'status': 'healthy' if tts.is_loaded else 'unhealthy',
            'model_loaded': tts.is_loaded,
            'device': tts.device if tts.is_loaded else None,
            'sample_rate': tts.sample_rate if tts.is_loaded else None,
            'voices_dir': tts.voices_dir,
            'voice_check': {'valid': voice_valid, 'message': voice_msg},
        }
    ), 200 if tts.is_loaded else 503


@api.route('/v1/voices', methods=['GET'])
def list_voices():
    """
    List available voices (OpenAI-compatible format, extended with meta).
    """
    from pathlib import Path
    tts = get_tts_service()
    voices = tts.list_voices()
    meta_svc = get_voice_meta_service()
    all_meta = meta_svc.all_voice_meta()

    def _has_vector(voice_id: str) -> bool:
        if not tts.voices_cache_dir:
            return False
        stem = Path(voice_id).stem if '.' in voice_id else voice_id
        return (Path(tts.voices_cache_dir) / f'{stem}.safetensors').exists()

    return jsonify(
        {
            'object': 'list',
            'data': [
                {
                    'id': v['id'],
                    'name': v['name'],
                    'object': 'voice',
                    'type': v.get('type', 'builtin'),
                    'user_uploaded': v.get('user_uploaded', False),
                    'hidden': all_meta.get(v['id'], {}).get('hidden', False),
                    'tags': all_meta.get(v['id'], {}).get('tags', []),
                    'note': all_meta.get(v['id'], {}).get('note', ''),
                    'has_vector': _has_vector(v['id']),
                }
                for v in voices
            ],
        }
    )


@api.route('/v1/voices/precompute', methods=['POST'])
def precompute_voices():
    """
    Pre-compute .safetensors voice state cache for all custom WAV voices.
    Returns a streaming NDJSON response with per-voice progress.
    Skips voices that are already cached.
    """
    tts = get_tts_service()

    def generate():
        for progress in tts.precompute_voices():
            yield json.dumps(progress) + '\n'

    return Response(stream_with_context(generate()), mimetype='application/x-ndjson')


@api.route('/v1/voices/<voice_id>/download/audio', methods=['GET'])
def download_voice_audio(voice_id):
    """
    Download the raw audio reference file for a custom voice (WAV/MP3/etc).
    Returns 404 for built-in voices (no audio file on disk).
    """
    from pathlib import Path
    tts = get_tts_service()
    for search_dir in filter(None, [tts.voices_dir, tts.voices_user_dir]):
        p = Path(search_dir) / voice_id
        if p.exists() and p.is_file():
            return send_file(str(p), as_attachment=True, download_name=p.name)
    return jsonify({'error': f'Audio file for voice {voice_id!r} not found on disk. Built-in voices have no local audio file.'}), 404


@api.route('/v1/voices/<voice_id>/download/vector', methods=['GET'])
def download_voice_vector(voice_id):
    """
    Download the pre-computed .safetensors state file for a voice (if cached).
    This is the voice embedding — import it to another pocket-tts instance to
    use the voice without the original audio file.
    """
    from pathlib import Path
    tts = get_tts_service()
    if not tts.voices_cache_dir:
        return jsonify({'error': 'Voice cache directory not configured'}), 503
    stem = Path(voice_id).stem if '.' in voice_id else voice_id
    cache_file = Path(tts.voices_cache_dir) / f'{stem}.safetensors'
    if not cache_file.exists():
        return jsonify({'error': f'No cached vector file for {voice_id!r}. Run Precompute on the Voices tab first.'}), 404
    return send_file(str(cache_file), as_attachment=True, download_name=cache_file.name)


# ══════════════════════════════════════════════════════════════════
# Voice Metadata — tags & hidden state (server-side persistence)
# ══════════════════════════════════════════════════════════════════

@api.route('/v1/voice-meta', methods=['GET'])
def get_voice_meta_dump():
    """Return full voice-meta snapshot: {version, tags, voices}."""
    return jsonify(get_voice_meta_service().full_dump())


@api.route('/v1/voices/<voice_id>/meta', methods=['GET'])
def get_single_voice_meta(voice_id):
    """Return metadata for a single voice."""
    return jsonify(get_voice_meta_service().get_voice_meta(voice_id))


@api.route('/v1/voices/<voice_id>/meta', methods=['PATCH'])
def patch_voice_meta(voice_id):
    """Update hidden/tags/note for a voice. All fields optional."""
    data = request.json or {}
    result = get_voice_meta_service().set_voice_meta(
        voice_id,
        hidden=data.get('hidden'),
        tags=data.get('tags'),
        note=data.get('note'),
    )
    return jsonify(result)


# ──────────────────────────────────────────────────────────── tags ───

@api.route('/v1/tags', methods=['GET'])
def get_tags():
    """List all tags including voice counts."""
    svc = get_voice_meta_service()
    tags = svc.list_tags()
    meta_dump = svc.all_voice_meta()
    # Annotate with voice counts
    for tag in tags:
        tag['voice_count'] = sum(
            1 for v in meta_dump.values() if tag['id'] in v.get('tags', [])
        )
    return jsonify({'object': 'list', 'data': tags})


@api.route('/v1/tags', methods=['POST'])
def create_tag():
    """Create a new tag. Body: {name, color?, parent?}."""
    data = request.json or {}
    name = (data.get('name') or '').strip()
    if not name:
        return jsonify({'error': 'name is required'}), 400
    color = data.get('color', '#5c6ef8')
    parent = data.get('parent')
    tag = get_voice_meta_service().create_tag(name, color=color, parent=parent)
    return jsonify(tag), 201


@api.route('/v1/tags/<tag_id>', methods=['PATCH'])
def update_tag(tag_id):
    """Rename/recolor a tag. Body: {name?, color?, parent?}."""
    data = request.json or {}
    result = get_voice_meta_service().update_tag(
        tag_id,
        name=data.get('name'),
        color=data.get('color'),
        parent=data.get('parent'),
    )
    if result is None:
        return jsonify({'error': 'tag not found'}), 404
    return jsonify(result)


@api.route('/v1/tags/<tag_id>', methods=['DELETE'])
def delete_tag(tag_id):
    """Delete a tag and unassign it from all voices."""
    deleted = get_voice_meta_service().delete_tag(tag_id)
    if not deleted:
        return jsonify({'error': 'tag not found'}), 404
    return jsonify({'deleted': True, 'id': tag_id})


@api.route('/v1/tags/order', methods=['PUT'])
def save_tag_order():
    """Persist user-preferred tag display order. Body: {order: [tid,...]}."""
    data = request.json or {}
    order = data.get('order', [])
    if not isinstance(order, list):
        return jsonify({'error': 'order must be a list'}), 400
    saved = get_voice_meta_service().set_tag_order(order)
    return jsonify({'order': saved})


# ─────────────────────────────────────────── user-uploaded voices ───

@api.route('/v1/voices/user', methods=['GET'])
def list_user_voices():
    """List user-uploaded voices (private, from voices_user dir)."""
    tts = get_tts_service()
    if not tts.voices_user_dir:
        return jsonify({'object': 'list', 'data': []})
    from pathlib import Path
    from app.config import Config
    meta_svc = get_voice_meta_service()
    all_meta = meta_svc.all_voice_meta()
    user_dir = Path(tts.voices_user_dir)
    audio_exts = {e for e in Config.VOICE_EXTENSIONS if e != '.safetensors'}
    files = sorted(
        [f for f in user_dir.iterdir() if f.suffix.lower() in audio_exts],
        key=lambda f: f.name.lower(),
    )
    data = []
    for vf in files:
        clean_name = vf.stem.replace('_', ' ').replace('-', ' ').title()
        voice_id = vf.name
        vm = all_meta.get(voice_id, {})
        data.append({
            'id': voice_id,
            'name': clean_name,
            'type': 'custom',
            'user_uploaded': True,
            'hidden': vm.get('hidden', False),
            'tags': vm.get('tags', []),
            'note': vm.get('note', ''),
            'size_bytes': vf.stat().st_size,
        })
    return jsonify({'object': 'list', 'data': data})


@api.route('/v1/voices/upload', methods=['POST'])
def upload_voice():
    """
    Upload a WAV (or other audio) file as a user voice.
    Multipart form field: 'file' (required), 'name' (optional label).
    Saves to voices_user_dir.
    """
    from pathlib import Path
    from werkzeug.utils import secure_filename
    from app.config import Config

    tts = get_tts_service()
    if not tts.voices_user_dir:
        return jsonify({'error': 'User voices directory not configured (POCKET_TTS_VOICES_USER_DIR)'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'No filename provided'}), 400

    filename = secure_filename(f.filename)
    ext = Path(filename).suffix.lower()
    allowed = {e for e in Config.VOICE_EXTENSIONS if e != '.safetensors'}
    if ext not in allowed:
        return jsonify({'error': f'File type {ext} not allowed. Accepted: {sorted(allowed)}'}), 400

    dest = Path(tts.voices_user_dir) / filename
    if dest.exists():
        # Avoid silent overwrite — require explicit overwrite flag
        if not request.form.get('overwrite'):
            return jsonify({'error': f'{filename} already exists. Send overwrite=true to replace.'}), 409

    f.save(str(dest))
    logger.info(f'User voice uploaded: {filename} ({dest.stat().st_size} bytes)')
    clean_name = dest.stem.replace('_', ' ').replace('-', ' ').title()
    return jsonify({
        'id': filename,
        'name': clean_name,
        'type': 'custom',
        'user_uploaded': True,
        'size_bytes': dest.stat().st_size,
    }), 201


@api.route('/v1/voices/user/<path:filename>', methods=['DELETE'])
def delete_user_voice(filename):
    """Permanently delete a user-uploaded voice file and its metadata."""
    from pathlib import Path
    from werkzeug.utils import secure_filename

    tts = get_tts_service()
    if not tts.voices_user_dir:
        return jsonify({'error': 'User voices directory not configured'}), 503

    safe = secure_filename(filename)
    target = Path(tts.voices_user_dir) / safe
    if not target.exists():
        return jsonify({'error': f'{filename} not found'}), 404

    target.unlink()
    logger.info(f'User voice deleted: {filename}')

    # Also remove any cached safetensors for this voice
    tts_svc = get_tts_service()
    if tts_svc.voices_cache_dir:
        cache_file = Path(tts_svc.voices_cache_dir) / f'{target.stem}.safetensors'
        if cache_file.exists():
            cache_file.unlink()
            logger.info(f'Removed cache for deleted voice: {cache_file.name}')

    # Remove from in-memory voice cache
    abs_path = str(target.resolve())
    tts.voice_cache.pop(abs_path, None)

    return jsonify({'deleted': True, 'id': filename})


# ══════════════════════════════════════════════════════════════════
# Test Texts — server-side persistence for labelled TTS test snippets
# ══════════════════════════════════════════════════════════════════

@api.route('/v1/test-texts', methods=['GET'])
def list_test_texts():
    svc = get_test_texts_service()
    return jsonify({'data': svc.list_texts()})


@api.route('/v1/test-texts/<text_id>', methods=['GET'])
def get_test_text(text_id):
    svc = get_test_texts_service()
    entry = svc.get_text(text_id)
    if not entry:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(entry)


@api.route('/v1/test-texts', methods=['POST'])
def create_test_text():
    data = request.json or {}
    label = (data.get('label') or '').strip()
    text = (data.get('text') or '').strip()
    if not label or not text:
        return jsonify({'error': 'label and text are required'}), 400
    tags = [t.strip() for t in data.get('tags', []) if str(t).strip()]
    svc = get_test_texts_service()
    return jsonify(svc.create_text(label, text, tags)), 201


@api.route('/v1/test-texts/<text_id>', methods=['PUT'])
def update_test_text(text_id):
    data = request.json or {}
    label = data.get('label')
    text = data.get('text')
    tags = data.get('tags')
    if tags is not None:
        tags = [t.strip() for t in tags if str(t).strip()]
    svc = get_test_texts_service()
    updated = svc.update_text(text_id, label=label, text=text, tags=tags)
    if not updated:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(updated)


@api.route('/v1/test-texts/<text_id>', methods=['DELETE'])
def delete_test_text(text_id):
    svc = get_test_texts_service()
    if not svc.delete_text(text_id):
        return jsonify({'error': 'Not found'}), 404
    return jsonify({'deleted': True, 'id': text_id})


# ══════════════════════════════════════════════════════════════════
# Monitor — SSE feed and REST snapshot of recent TTS requests
# ══════════════════════════════════════════════════════════════════

@api.route('/v1/monitor/stream', methods=['GET'])
def monitor_stream():
    """
    SSE endpoint — streams new TTS request events as they arrive.
    On connect: replays all buffered events (oldest first), then streams new ones.
    Sends a keepalive comment every 15s to prevent proxy timeouts.
    """
    import queue as _queue
    from app.services import monitor_service

    def generate():
        # Replay existing buffer so new subscriber sees recent history
        for evt in monitor_service.get_events():
            yield f'data: {json.dumps(evt)}\n\n'

        q = monitor_service.subscribe()
        try:
            while True:
                try:
                    evt = q.get(timeout=15)
                    yield f'data: {json.dumps(evt)}\n\n'
                except _queue.Empty:
                    # Keepalive comment — keeps connection alive through proxies
                    yield ': keepalive\n\n'
        finally:
            monitor_service.unsubscribe(q)

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


@api.route('/v1/monitor/events', methods=['GET'])
def monitor_events():
    """REST snapshot — returns all buffered events as JSON (oldest first)."""
    from app.services import monitor_service
    return jsonify({'object': 'list', 'data': monitor_service.get_events()})


@api.route('/v1/monitor/events', methods=['DELETE'])
def monitor_events_clear():
    """Clear all buffered monitor events (server-side + triggers UI refresh)."""
    from app.services import monitor_service
    monitor_service.clear_events()
    return jsonify({'cleared': True})


# ══════════════════════════════════════════════════════════════════
# LavaSR Audio Enhancement — runtime control
# LavaSR — Apache License 2.0, Copyright (c) 2024 Yatharth Sharma
# MIT (this codebase) + Apache 2.0 (LavaSR) are fully compatible.
# See LICENSES/LICENSE-LAVASR for the full Apache 2.0 text.
# ══════════════════════════════════════════════════════════════════

@api.route('/v1/enhancer/status', methods=['GET'])
def enhancer_status():
    """Return LavaSR availability, load state, and current configuration."""
    from flask import current_app
    from app.config import Config
    from app.services.enhancer import get_enhancer_service
    svc = get_enhancer_service()
    return jsonify({
        'available': True,
        'enabled': current_app.config.get('LAVASR_ENABLED', False),
        'model_loaded': svc.is_loaded,
        'enhance_default': current_app.config.get('LAVASR_ENHANCE', True),
        'denoise_default': current_app.config.get('LAVASR_DENOISE', False),
        'model_id': Config.LAVASR_MODEL,
        'device': Config.LAVASR_DEVICE,
        'output_sr': 48000,
    })


@api.route('/v1/enhancer/config', methods=['GET'])
def get_enhancer_config():
    """Return current runtime enhancer configuration."""
    from flask import current_app
    return jsonify({
        'enabled': current_app.config.get('LAVASR_ENABLED', False),
        'enhance': current_app.config.get('LAVASR_ENHANCE', True),
        'denoise': current_app.config.get('LAVASR_DENOISE', False),
    })


@api.route('/v1/enhancer/config', methods=['POST'])
def set_enhancer_config():
    """Update runtime enhancer settings.  Changes take effect immediately but
    reset to env-var defaults on container restart."""
    from flask import current_app
    data = request.json or {}
    changed = {}
    if 'enabled' in data:
        current_app.config['LAVASR_ENABLED'] = bool(data['enabled'])
        changed['enabled'] = current_app.config['LAVASR_ENABLED']
    if 'enhance' in data:
        current_app.config['LAVASR_ENHANCE'] = bool(data['enhance'])
        changed['enhance'] = current_app.config['LAVASR_ENHANCE']
    if 'denoise' in data:
        current_app.config['LAVASR_DENOISE'] = bool(data['denoise'])
        changed['denoise'] = current_app.config['LAVASR_DENOISE']
    return jsonify({
        'updated': changed,
        'config': {
            'enabled': current_app.config.get('LAVASR_ENABLED', False),
            'enhance': current_app.config.get('LAVASR_ENHANCE', True),
            'denoise': current_app.config.get('LAVASR_DENOISE', False),
        },
    })


@api.route('/v1/enhancer/load', methods=['POST'])
def load_enhancer_model():
    """Pre-load the LavaSR model (otherwise it loads lazily on first enhanced request).
    Useful for warming up on startup rather than incurring latency on the first call."""
    from app.services.enhancer import get_enhancer_service
    svc = get_enhancer_service()
    if svc.is_loaded:
        return jsonify({'status': 'already_loaded'})
    try:
        svc.load_model()
        return jsonify({'status': 'loaded'})
    except Exception as exc:
        return jsonify({'status': 'error', 'error': str(exc)}), 500


@api.route('/v1/audio/speech', methods=['POST'])
def generate_speech():
    """
    OpenAI-compatible speech generation endpoint.

    Request body:
        model: string (ignored, for compatibility)
        input: string (required) - Text to synthesize
        voice: string (optional) - Voice ID or path
        response_format: string (optional) - Audio format
        stream: boolean (optional) - Enable streaming

    Returns:
        Audio file or streaming audio response
    """
    from flask import current_app

    data = request.json

    if not data:
        return jsonify({'error': 'Missing JSON body'}), 400

    text = data.get('input')
    if not text:
        return jsonify({'error': "Missing 'input' text"}), 400

    voice = data.get('voice', 'alba')
    stream_request = data.get('stream', False)

    response_format = data.get('response_format', 'mp3')
    target_format = validate_format(response_format)

    tts = get_tts_service()

    # Validate voice first
    is_valid, msg = tts.validate_voice(voice)
    if not is_valid:
        available = [v['id'] for v in tts.list_voices()]
        return jsonify(
            {
                'error': f"Voice '{voice}' not found",
                'available_voices': available[:10],  # Limit to first 10
                'hint': 'Use /v1/voices to see all available voices',
            }
        ), 400

    try:
        voice_state = tts.get_voice_state(voice)

        # Check if streaming should be used
        use_streaming = stream_request or current_app.config.get('STREAM_DEFAULT', False)

        # Streaming supports only PCM/WAV today; fall back to file for other formats.
        if use_streaming and target_format not in ('pcm', 'wav'):
            logger.warning(
                "Streaming format '%s' is not supported; returning full file instead.",
                target_format,
            )
            use_streaming = False
        # Check if text preprocessing should be used
        use_text_preprocess = current_app.config.get('TEXT_PREPROCESS_DEFAULT', False)
        # Preprocess text
        if use_text_preprocess:
            # logger.info(f'Preprocessing text: {text}')
            text = text_preprocessor.process(text)
            # logger.info(f'Preprocessed text: {text}')

        # ── LavaSR audio enhancement ─────────────────────────────────────
        # Per-request 'enhance'/'denoise' override server defaults.
        # Pass null/omit to use whatever the server default is.
        lavasr_enabled = current_app.config.get('LAVASR_ENABLED', False)
        _enhance_req = data.get('enhance')   # None | True | False
        _denoise_req = data.get('denoise')   # None | True | False
        do_enhance = (
            bool(_enhance_req) if _enhance_req is not None
            else current_app.config.get('LAVASR_ENHANCE', True)
        ) and lavasr_enabled
        do_denoise = (
            bool(_denoise_req) if _denoise_req is not None
            else current_app.config.get('LAVASR_DENOISE', False)
        ) and lavasr_enabled

        # Enhancement requires buffering the full waveform — incompatible with streaming.
        if (do_enhance or do_denoise) and use_streaming:
            logger.info('LavaSR enhancement requested — switching from streaming to batch mode')
            use_streaming = False

        if use_streaming:
            return _stream_audio(tts, voice_state, text, target_format, voice_name=voice, req_text=text)
        return _generate_file(
            tts, voice_state, text, target_format,
            do_enhance=do_enhance, do_denoise=do_denoise,
            voice_name=voice, req_text=text,
        )

    except ValueError as e:
        logger.warning(f'Voice loading failed: {e}')
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception('Generation failed')
        return jsonify({'error': str(e)}), 500


def _generate_file(tts, voice_state, text: str, fmt: str, do_enhance: bool = False, do_denoise: bool = False, voice_name: str = '', req_text: str = ''):
    """Generate complete audio, optionally enhance with LavaSR, and return as file."""
    t0 = time.time()
    audio_tensor = tts.generate_audio(voice_state, text)
    generation_time = time.time() - t0

    logger.info(f'Generated {len(text)} chars in {generation_time:.2f}s')

    # Apply LavaSR bandwidth extension / denoising if requested.
    # LavaSR outputs 48 kHz regardless of TTS native sample rate (24 kHz).
    sample_rate = tts.sample_rate
    if do_enhance or do_denoise:
        from app.services.enhancer import get_enhancer_service
        try:
            audio_tensor, sample_rate = get_enhancer_service().enhance(
                audio_tensor,
                input_sr=tts.sample_rate,
                do_enhance=do_enhance,
                do_denoise=do_denoise,
            )
            logger.info(f'LavaSR: {tts.sample_rate} Hz → {sample_rate} Hz (enhance={do_enhance}, denoise={do_denoise})')
        except Exception as exc:
            logger.error(f'LavaSR enhancement failed — falling back to raw audio: {exc}')
            sample_rate = tts.sample_rate

    # Record monitor event — for batch mode TTFA equals the full generation time
    audio_duration = audio_tensor.shape[-1] / max(sample_rate, 1)
    from app.services import monitor_service
    monitor_service.record_event({
        'ts': t0,
        'voice': voice_name or 'unknown',
        'text_preview': req_text[:200],
        'text_full': req_text,
        'ttfa': round(generation_time, 3),
        'audio_duration': round(audio_duration, 2),
        'gen_time': round(generation_time, 2),
        'mode': 'batch',
        'enhanced': do_enhance or do_denoise,
    })

    audio_buffer = convert_audio(audio_tensor, sample_rate, fmt)
    mimetype = get_mime_type(fmt)

    return send_file(
        audio_buffer, mimetype=mimetype, as_attachment=True, download_name=f'speech.{fmt}'
    )


def _stream_audio(tts, voice_state, text: str, fmt: str, voice_name: str = '', req_text: str = ''):
    """Stream audio chunks."""
    # Normalize streaming format: we always emit PCM bytes, optionally wrapped
    # in a WAV container. For non-PCM/WAV formats (e.g. mp3, opus), coerce to
    # raw PCM to avoid mismatched content-type vs. payload.
    stream_fmt = fmt
    if stream_fmt not in ('pcm', 'wav'):
        logger.warning(
            "Requested streaming format '%s' is not supported for streaming; "
            "falling back to 'pcm'.",
            stream_fmt,
        )
        stream_fmt = 'pcm'

    t_start = time.time()
    _ttfa: list[float | None] = [None]
    _total_samples: list[int] = [0]

    def generate():
        stream = tts.generate_audio_stream(voice_state, text)
        for chunk_tensor in stream:
            if _ttfa[0] is None:
                _ttfa[0] = time.time() - t_start
            _total_samples[0] += chunk_tensor.shape[-1]
            yield tensor_to_pcm_bytes(chunk_tensor)
        # Record monitor event after the last chunk is yielded
        from app.services import monitor_service
        audio_duration = _total_samples[0] / max(tts.sample_rate, 1)
        monitor_service.record_event({
            'ts': t_start,
            'voice': voice_name or 'unknown',
            'text_preview': req_text[:200],
            'text_full': req_text,
            'ttfa': round(_ttfa[0] or 0.0, 3),
            'audio_duration': round(audio_duration, 2),
            'gen_time': round(time.time() - t_start, 2),
            'mode': 'stream',
        })

    def stream_with_header():
        # Yield WAV header first if streaming as WAV
        if stream_fmt == 'wav':
            yield write_wav_header(tts.sample_rate, num_channels=1, bits_per_sample=16)
        yield from generate()

    mimetype = get_mime_type(stream_fmt)

    return Response(stream_with_context(stream_with_header()), mimetype=mimetype)
