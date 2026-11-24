import ctypes
import io
import os
import tarfile
import threading
from pathlib import Path

import numpy as np
import pytest
from pipecat.frames.frames import FilterEnableFrame

from hn_pipecat.noise import (
    DF_SAMPLE_RATE,
    DeepFilterAudioFilter,
    DeepFilterEngine,
    DeepFilterRuntime,
    load_runtime,
)

HOP = 480


class IdentityLib:
    """Stands in for libdf: copies each 10 ms hop unchanged."""

    def __init__(self) -> None:
        self.freed = False
        self.hops = 0

    def df_create(self, path: bytes, atten: float, log: object) -> int:
        return 1

    def df_get_frame_length(self, state: int) -> int:
        return HOP

    def df_process_frame(self, state: int, inp: object, out: object) -> float:
        self.hops += 1
        ctypes.memmove(out, inp, HOP * ctypes.sizeof(ctypes.c_float))
        return 0.0

    def df_free(self, state: int) -> None:
        self.freed = True


def runtime_with(lib: IdentityLib) -> DeepFilterRuntime:
    return DeepFilterRuntime(lib=lib, model_path="unused", attenuation_db=100.0)  # type: ignore[arg-type]


def pcm(samples: np.ndarray) -> bytes:
    return samples.astype(np.int16).tobytes()


def test_engine_keeps_length_and_only_delays() -> None:
    lib = IdentityLib()
    engine = DeepFilterEngine(runtime_with(lib), background=False)
    signal = np.arange(-3000, 3000, dtype=np.int16)
    chunks = [signal[:700], signal[700:1500], signal[1500:4000], signal[4000:]]
    out = np.concatenate([np.frombuffer(engine.process(pcm(c)), dtype=np.int16) for c in chunks])
    assert len(out) == len(signal)
    delay = int(np.argmax(out != 0)) - int(np.argmax(signal != 0))
    assert delay == HOP
    np.testing.assert_array_equal(out[delay:], signal[: len(signal) - delay])


def test_engine_whole_hops_pass_unchanged() -> None:
    lib = IdentityLib()
    engine = DeepFilterEngine(runtime_with(lib), background=False)
    signal = (np.sin(np.arange(2400) / 10) * 8000).astype(np.int16)
    out = np.frombuffer(engine.process(pcm(signal)), dtype=np.int16)
    assert lib.hops == 5
    np.testing.assert_allclose(out, signal, atol=1)


def test_engine_passes_audio_through_while_loading_and_after_close() -> None:
    lib = IdentityLib()
    gate = threading.Event()
    original_create = lib.df_create

    def slow_create(path: bytes, atten: float, log: object) -> int:
        gate.wait(5)
        return original_create(path, atten, log)

    lib.df_create = slow_create  # type: ignore[method-assign]
    engine = runtime_with(lib).create_engine()
    early = pcm(np.ones(2400))
    assert engine.process(early) is early
    gate.set()
    assert engine.ready.wait(5)
    assert engine.hop_size == HOP
    assert engine.process(pcm(np.ones(2400))) is not early
    engine.close()
    engine.close()
    assert lib.freed
    late = pcm(np.ones(480))
    assert engine.process(late) is late


async def test_filter_at_48k_runs_without_resampling() -> None:
    lib = IdentityLib()
    f = DeepFilterAudioFilter(runtime_with(lib), background=False)
    await f.start(DF_SAMPLE_RATE)
    signal = (np.sin(np.arange(960) / 7) * 6000).astype(np.int16)
    out = await f.filter(pcm(signal))
    assert len(out) == len(signal) * 2
    assert lib.hops == 2
    await f.stop()
    assert lib.freed


async def test_filter_resamples_16k_input_and_keeps_pace() -> None:
    lib = IdentityLib()
    f = DeepFilterAudioFilter(runtime_with(lib), background=False)
    await f.start(16_000)
    frame = pcm((np.sin(np.arange(320) / 5) * 6000).astype(np.int16))  # 20 ms at 16 kHz
    total = 0
    for _ in range(50):  # one second
        total += len(await f.filter(frame))
    # Output keeps pace with input (within the resamplers' start-up delay).
    assert abs(total - 50 * len(frame)) <= 2 * len(frame)
    assert lib.hops > 90
    await f.stop()


async def test_filter_can_be_switched_off_at_runtime() -> None:
    lib = IdentityLib()
    f = DeepFilterAudioFilter(runtime_with(lib), background=False)
    await f.start(DF_SAMPLE_RATE)
    await f.process_frame(FilterEnableFrame(enable=False))
    audio = pcm(np.ones(480))
    assert await f.filter(audio) is audio
    assert lib.hops == 0
    await f.process_frame(FilterEnableFrame(enable=True))
    await f.filter(audio)
    assert lib.hops == 1


async def test_filter_before_start_is_a_pass_through() -> None:
    f = DeepFilterAudioFilter(runtime_with(IdentityLib()), background=False)
    audio = pcm(np.ones(320))
    assert await f.filter(audio) is audio


def test_missing_library_or_model_is_reported(tmp_path: Path) -> None:
    reason = load_runtime({"HN_DFN_LIB": str(tmp_path / "nope.so")})
    assert isinstance(reason, str)
    assert "library not found" in reason

    lib = tmp_path / "libdf.so"
    lib.write_bytes(b"")
    reason = load_runtime({"HN_DFN_LIB": str(lib), "HN_DFN_MODEL": str(tmp_path / "m.tar.gz")})
    assert isinstance(reason, str)
    assert "model not found" in reason

    model = tmp_path / "bad.tar.gz"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        info = tarfile.TarInfo("DeepFilterNet3_onnx/enc.onnx")
        archive.addfile(info, io.BytesIO(b""))
    model.write_bytes(buf.getvalue())
    reason = load_runtime({"HN_DFN_LIB": str(lib), "HN_DFN_MODEL": str(model)})
    assert isinstance(reason, str)
    assert "lacks erb_dec.onnx" in reason


@pytest.mark.skipif(
    not (os.environ.get("HN_TEST_DFN_LIB") and os.environ.get("HN_TEST_DFN_MODEL")),
    reason="set HN_TEST_DFN_LIB and HN_TEST_DFN_MODEL to test the real library",
)
async def test_real_deepfilternet_reduces_noise() -> None:
    runtime = load_runtime(
        {
            "HN_DFN_LIB": os.environ["HN_TEST_DFN_LIB"],
            "HN_DFN_MODEL": os.environ["HN_TEST_DFN_MODEL"],
        }
    )
    assert isinstance(runtime, DeepFilterRuntime), runtime
    f = DeepFilterAudioFilter(runtime, background=False)
    await f.start(16_000)
    rng = np.random.default_rng(1)
    noise = (rng.normal(0, 2000, 16_000 * 3)).astype(np.int16)
    out = b""
    for i in range(0, len(noise), 320):
        out += await f.filter(pcm(noise[i : i + 320]))
    cleaned = np.frombuffer(out, dtype=np.int16)[16_000:]
    assert float(np.sqrt(np.mean(cleaned.astype(np.float64) ** 2))) < 2000 * 0.5
    await f.stop()
