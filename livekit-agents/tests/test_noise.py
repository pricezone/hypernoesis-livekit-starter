import ctypes
import io
import os
import tarfile
import threading
from pathlib import Path

import numpy as np
import pytest
from livekit import rtc

from hn_agent.noise import (
    DF_SAMPLE_RATE,
    DeepFilterProcessor,
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


def fake_processor() -> tuple[DeepFilterProcessor, IdentityLib]:
    lib = IdentityLib()
    runtime = DeepFilterRuntime(lib=lib, model_path="unused", attenuation_db=100.0)  # type: ignore[arg-type]
    return DeepFilterProcessor(runtime, background=False), lib


def frame(samples: np.ndarray, rate: int = DF_SAMPLE_RATE) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=samples.astype(np.int16).tobytes(),
        sample_rate=rate,
        num_channels=1,
        samples_per_channel=len(samples),
    )


def test_50ms_frames_pass_through_unchanged_length() -> None:
    processor, lib = fake_processor()
    signal = (np.sin(np.arange(2400) / 10) * 8000).astype(np.int16)
    out = processor._process(frame(signal))
    assert out.samples_per_channel == 2400
    assert out.sample_rate == DF_SAMPLE_RATE
    assert lib.hops == 5
    np.testing.assert_allclose(np.frombuffer(out.data, dtype=np.int16), signal, atol=1)


def test_odd_frame_sizes_keep_length_and_order() -> None:
    processor, _ = fake_processor()
    signal = np.arange(-3000, 3000, dtype=np.int16)
    chunks = [signal[:700], signal[700:1500], signal[1500:4000], signal[4000:]]
    out = np.concatenate(
        [np.frombuffer(processor._process(frame(c)).data, dtype=np.int16) for c in chunks]
    )
    assert len(out) == len(signal)
    # The identity filter only delays: the output is the input shifted by one
    # hop of silence, padded once when the first frame was not a whole hop.
    delay = int(np.argmax(out != 0)) - int(np.argmax(signal != 0))
    assert delay == HOP
    np.testing.assert_array_equal(out[delay:], signal[: len(signal) - delay])


def test_wrong_format_is_passed_through() -> None:
    processor, lib = fake_processor()
    original = frame(np.ones(240, dtype=np.int16), rate=24_000)
    assert processor._process(original) is original
    assert lib.hops == 0


def test_close_frees_the_state_once() -> None:
    processor, lib = fake_processor()
    processor._close()
    processor._close()
    assert lib.freed
    passthrough = frame(np.ones(480, dtype=np.int16))
    assert processor._process(passthrough) is passthrough


def test_frames_pass_through_while_the_model_loads() -> None:
    lib = IdentityLib()
    gate = threading.Event()
    original_create = lib.df_create

    def slow_create(path: bytes, atten: float, log: object) -> int:
        gate.wait(5)
        return original_create(path, atten, log)

    lib.df_create = slow_create  # type: ignore[method-assign]
    runtime = DeepFilterRuntime(lib=lib, model_path="unused", attenuation_db=100.0)  # type: ignore[arg-type]
    processor = runtime.create()
    early = frame(np.ones(2400, dtype=np.int16))
    assert processor._process(early) is early
    gate.set()
    assert processor.ready.wait(5)
    assert processor.hop_size == HOP
    assert processor._process(frame(np.ones(2400, dtype=np.int16))) is not early
    assert lib.hops == 5


def test_close_during_load_frees_the_late_state() -> None:
    lib = IdentityLib()
    gate = threading.Event()
    original_create = lib.df_create

    def slow_create(path: bytes, atten: float, log: object) -> int:
        gate.wait(5)
        return original_create(path, atten, log)

    lib.df_create = slow_create  # type: ignore[method-assign]
    runtime = DeepFilterRuntime(lib=lib, model_path="unused", attenuation_db=100.0)  # type: ignore[arg-type]
    processor = runtime.create()
    processor._close()
    gate.set()
    assert processor.ready.wait(5)
    assert lib.freed


def test_enabled_toggle() -> None:
    processor, _ = fake_processor()
    assert processor.enabled
    processor.enabled = False
    assert not processor.enabled
    passthrough = frame(np.ones(2400, dtype=np.int16))
    assert processor._process(passthrough) is passthrough


def test_missing_library_is_reported(tmp_path: Path) -> None:
    result = load_runtime({"HN_DFN_LIB": str(tmp_path / "nope.so")})
    assert isinstance(result, str)
    assert "library not found" in result


def _archive(path: Path, members: list[str]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name in members:
            info = tarfile.TarInfo(f"DeepFilterNet3_onnx/{name}")
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))


def test_bad_model_archive_is_reported_before_loading(tmp_path: Path) -> None:
    lib = tmp_path / "libdf.so"
    lib.write_bytes(b"")
    model = tmp_path / "model.tar.gz"
    _archive(model, ["enc.onnx", "config.ini"])
    result = load_runtime({"HN_DFN_LIB": str(lib), "HN_DFN_MODEL": str(model)})
    assert result == "DeepFilterNet model archive lacks erb_dec.onnx, df_dec.onnx"

    model.write_bytes(b"not a tarball")
    result = load_runtime(
        {"HN_DFN_LIB": str(lib), "HN_DFN_MODEL": str(model), "HN_NC_ATTENUATION_DB": "40"}
    )
    assert result == "DeepFilterNet model archive is unreadable"


REAL_LIB = os.environ.get("HN_TEST_DFN_LIB")
REAL_MODEL = os.environ.get("HN_TEST_DFN_MODEL")


@pytest.mark.skipif(not (REAL_LIB and REAL_MODEL), reason="needs a built libdf.so")
def test_real_deepfilternet_reduces_noise() -> None:
    runtime = load_runtime({"HN_DFN_LIB": REAL_LIB or "", "HN_DFN_MODEL": REAL_MODEL or ""})
    assert isinstance(runtime, DeepFilterRuntime), runtime
    processor = runtime.create(background=False)
    assert processor.hop_size == HOP
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(DF_SAMPLE_RATE * 2) * 3000).astype(np.int16)
    out = np.concatenate(
        [
            np.frombuffer(processor._process(frame(noise[i : i + 2400])).data, dtype=np.int16)
            for i in range(0, len(noise), 2400)
        ]
    )
    processor._close()
    tail = slice(DF_SAMPLE_RATE // 2, None)  # after the model has settled
    rms_in = float(np.sqrt(np.mean(noise[tail].astype(np.float64) ** 2)))
    rms_out = float(np.sqrt(np.mean(out[tail].astype(np.float64) ** 2)))
    assert rms_out < rms_in * 0.25
