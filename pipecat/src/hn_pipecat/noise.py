"""Optional noise cancellation with DeepFilterNet3 on the CPU (no PyTorch).

DeepFilterNet (MIT OR Apache-2.0, code and weights) ships a C API in its Rust
library `libDF`. Built with the `capi` feature it runs the DeepFilterNet3 ONNX
model frame by frame through `tract` (pure Rust inference), so the image needs
neither PyTorch nor onnxruntime for it. The Dockerfile builds `libdf.so` from a
pinned commit and copies it with the model archive to /opt/deepfilternet (the
same build as the LiveKit Agents starter in this repository).

Pipecat lets a transport run an audio filter on the caller's audio before VAD
and speech recognition (`TransportParams.audio_in_filter`). `DeepFilterAudioFilter`
is that filter: it resamples the pipeline's input rate (16 kHz by default) to
the 48 kHz DeepFilterNet needs, cleans it in 10 ms hops and resamples back.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import tarfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np
from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import FilterControlFrame, FilterEnableFrame

logger = logging.getLogger("hn_pipecat.noise")

DF_SAMPLE_RATE: Final = 48_000
DEFAULT_LIB_PATH: Final = "/opt/deepfilternet/libdf.so"
DEFAULT_MODEL_PATH: Final = "/opt/deepfilternet/DeepFilterNet3_onnx.tar.gz"
DEFAULT_ATTENUATION_DB: Final = 100.0  # DeepFilterNet's own default: no limit
_MODEL_MEMBERS: Final = ("enc.onnx", "erb_dec.onnx", "df_dec.onnx", "config.ini")

_FloatPtr = ctypes.POINTER(ctypes.c_float)


@dataclass(frozen=True)
class DeepFilterRuntime:
    lib: ctypes.CDLL
    model_path: str
    attenuation_db: float

    def create_engine(self, *, background: bool = True) -> DeepFilterEngine:
        return DeepFilterEngine(self, background=background)


_runtime_lock = threading.Lock()
_runtime_cache: dict[tuple[str, str, float], DeepFilterRuntime | str] = {}


def load_runtime(env: Mapping[str, str]) -> DeepFilterRuntime | str:
    """The loaded library, or a short reason why noise cancellation is unavailable."""
    lib_path = env.get("HN_DFN_LIB") or DEFAULT_LIB_PATH
    model_path = env.get("HN_DFN_MODEL") or DEFAULT_MODEL_PATH
    try:
        attenuation = float(env.get("HN_NC_ATTENUATION_DB") or DEFAULT_ATTENUATION_DB)
    except ValueError:
        attenuation = DEFAULT_ATTENUATION_DB
    attenuation = min(max(attenuation, 0.0), 100.0)

    key = (lib_path, model_path, attenuation)
    with _runtime_lock:
        cached = _runtime_cache.get(key)
        if cached is None:
            cached = _load(lib_path, model_path, attenuation)
            _runtime_cache[key] = cached
        return cached


def _load(lib_path: str, model_path: str, attenuation: float) -> DeepFilterRuntime | str:
    if not os.path.isfile(lib_path):
        return f"DeepFilterNet library not found ({lib_path})"
    # libDF aborts the whole process on a bad model file, so check the archive first.
    problem = _model_problem(model_path)
    if problem is not None:
        return problem
    try:
        lib = ctypes.CDLL(lib_path)
        lib.df_create.argtypes = [ctypes.c_char_p, ctypes.c_float, ctypes.c_char_p]
        lib.df_create.restype = ctypes.c_void_p
        lib.df_get_frame_length.argtypes = [ctypes.c_void_p]
        lib.df_get_frame_length.restype = ctypes.c_size_t
        lib.df_process_frame.argtypes = [ctypes.c_void_p, _FloatPtr, _FloatPtr]
        lib.df_process_frame.restype = ctypes.c_float
        lib.df_free.argtypes = [ctypes.c_void_p]
        lib.df_free.restype = None
    except (OSError, AttributeError) as err:
        return f"DeepFilterNet library could not be loaded ({type(err).__name__})"
    return DeepFilterRuntime(lib=lib, model_path=model_path, attenuation_db=attenuation)


def _model_problem(model_path: str) -> str | None:
    if not os.path.isfile(model_path):
        return f"DeepFilterNet model not found ({model_path})"
    try:
        with tarfile.open(model_path, "r:gz") as archive:
            names = {os.path.basename(name) for name in archive.getnames()}
    except (OSError, tarfile.TarError):
        return "DeepFilterNet model archive is unreadable"
    missing = [m for m in _MODEL_MEMBERS if m not in names]
    if missing:
        return f"DeepFilterNet model archive lacks {', '.join(missing)}"
    return None


class DeepFilterEngine:
    """Streams 48 kHz mono float samples through DeepFilterNet3, 10 ms at a time.

    Loading the model takes one to three seconds of CPU, so by default it runs
    in a background thread; audio passes through unchanged until it is ready.
    """

    def __init__(self, runtime: DeepFilterRuntime, *, background: bool = True) -> None:
        self._lib = runtime.lib
        self._model_path = runtime.model_path
        self._attenuation_db = runtime.attenuation_db
        self._lock = threading.Lock()
        self._state: int | None = None
        self._closed = False
        self._hop = 0
        self._pending = np.zeros(0, dtype=np.float32)  # input not yet processed
        self._ready = np.zeros(0, dtype=np.float32)  # processed, not yet returned
        self._hop_in = np.zeros(0, dtype=np.float32)
        self._hop_out = np.zeros(0, dtype=np.float32)
        self.ready = threading.Event()
        if background:
            threading.Thread(
                target=self._load_state, name="deepfilternet-load", daemon=True
            ).start()
        else:
            self._load_state()

    def _load_state(self) -> None:
        try:
            state = self._lib.df_create(self._model_path.encode(), self._attenuation_db, None)
            if not state:
                logger.warning("noise cancellation could not start: df_create returned no state")
                return
            hop = int(self._lib.df_get_frame_length(state))
            with self._lock:
                if self._closed:
                    self._lib.df_free(state)
                    return
                self._state = state
                self._hop = hop
                self._hop_in = np.zeros(hop, dtype=np.float32)
                self._hop_out = np.zeros(hop, dtype=np.float32)
        finally:
            self.ready.set()

    @property
    def hop_size(self) -> int:
        return self._hop

    @property
    def active(self) -> bool:
        return self._state is not None

    def process(self, pcm: bytes) -> bytes:
        """Clean 48 kHz mono int16 PCM; the same number of samples comes back."""
        samples = np.frombuffer(pcm, dtype=np.int16)
        with self._lock:
            if self._state is None:
                return pcm  # still loading, or closed
            cleaned = self._filter(samples.astype(np.float32) / 32768.0)
        out = np.clip(np.round(cleaned * 32768.0), -32768, 32767).astype(np.int16)
        return out.tobytes()

    def _filter(self, audio: np.ndarray) -> np.ndarray:
        """Return exactly len(audio) cleaned samples (delayed by the model's lookahead)."""
        pending = np.concatenate((self._pending, audio)) if len(self._pending) else audio
        hops = len(pending) // self._hop
        outputs = [self._ready] if len(self._ready) else []
        in_ptr = self._hop_in.ctypes.data_as(_FloatPtr)
        out_ptr = self._hop_out.ctypes.data_as(_FloatPtr)
        for i in range(hops):
            self._hop_in[:] = pending[i * self._hop : (i + 1) * self._hop]
            self._lib.df_process_frame(self._state, in_ptr, out_ptr)
            outputs.append(self._hop_out.copy())
        self._pending = pending[hops * self._hop :].copy()
        ready = np.concatenate(outputs) if outputs else np.zeros(0, dtype=np.float32)
        n = len(audio)
        if len(ready) < n:
            # Frames that are not a multiple of 10 ms: delay the output by one hop
            # of silence, once. After that the buffer can never run short again.
            pad = max(self._hop, n - len(ready))
            ready = np.concatenate((np.zeros(pad, dtype=np.float32), ready))
        self._ready = ready[n:].copy()
        return ready[:n]

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._state is not None:
                self._lib.df_free(self._state)
                self._state = None


class DeepFilterAudioFilter(BaseAudioFilter):
    """Pipecat input-audio filter that runs DeepFilterNet3 on the caller's audio."""

    def __init__(self, runtime: DeepFilterRuntime, *, background: bool = True) -> None:
        self._runtime = runtime
        self._background = background
        self._engine: DeepFilterEngine | None = None
        self._sample_rate = 0
        self._enabled = True
        self._resample_in = None
        self._resample_out = None

    @property
    def engine(self) -> DeepFilterEngine | None:
        return self._engine

    async def start(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._engine = self._runtime.create_engine(background=self._background)
        if sample_rate != DF_SAMPLE_RATE:
            self._resample_in = create_stream_resampler()
            self._resample_out = create_stream_resampler()

    async def stop(self) -> None:
        engine, self._engine = self._engine, None
        if engine is not None:
            engine.close()
        self._resample_in = self._resample_out = None

    async def process_frame(self, frame: FilterControlFrame) -> None:
        if isinstance(frame, FilterEnableFrame):
            self._enabled = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        engine = self._engine
        if engine is None or not self._enabled or not engine.active or not audio:
            return audio
        pcm = audio
        if self._resample_in is not None:
            pcm = await self._resample_in.resample(audio, self._sample_rate, DF_SAMPLE_RATE)
            if not pcm:
                return b""
        # libDF releases the GIL (ctypes), so other rooms keep running meanwhile.
        cleaned = await asyncio.to_thread(engine.process, pcm)
        if self._resample_out is not None:
            return await self._resample_out.resample(cleaned, DF_SAMPLE_RATE, self._sample_rate)
        return cleaned
