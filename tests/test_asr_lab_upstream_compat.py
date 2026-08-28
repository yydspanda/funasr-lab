from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_WRAPPERS = (
    "runtime/python/libtorch/funasr_torch/paraformer_bin.py",
    "runtime/python/libtorch/funasr_torch/sensevoice_bin.py",
    "runtime/python/onnxruntime/funasr_onnx/paraformer_bin.py",
    "runtime/python/onnxruntime/funasr_onnx/paraformer_online_bin.py",
    "runtime/python/onnxruntime/funasr_onnx/punc_bin.py",
    "runtime/python/onnxruntime/funasr_onnx/sensevoice_bin.py",
    "runtime/python/onnxruntime/funasr_onnx/vad_bin.py",
)


class UpstreamFrontendCompatibilityTest(unittest.TestCase):
    def test_locked_torchaudio_fbank_path_is_numerically_identical(self) -> None:
        original_sys_path = list(sys.path)
        try:
            import torch
            import torchaudio.compliance.kaldi as torchaudio_kaldi

            from funasr.utils import fbank as funasr_fbank

            self.assertTrue(
                funasr_fbank._HAS_TORCHAUDIO,
                "the locked lab environment must select torchaudio, not the optional fallback",
            )
            waveform = torch.linspace(-1.0, 1.0, steps=16_000).unsqueeze(0)
            options = {
                "num_mel_bins": 80,
                "frame_length": 25.0,
                "frame_shift": 10.0,
                "dither": 0.0,
                "energy_floor": 0.0,
                "window_type": "hamming",
                "sample_frequency": 16_000.0,
                "snip_edges": True,
            }

            wrapped = funasr_fbank.fbank(waveform, **options)
            direct = torchaudio_kaldi.fbank(waveform, **options)

            torch.testing.assert_close(wrapped, direct, rtol=0.0, atol=0.0)
        finally:
            sys.path[:] = original_sys_path


class UpstreamRuntimeWrapperCompatibilityTest(unittest.TestCase):
    def test_dependency_and_download_failures_raise_chained_exceptions(self) -> None:
        for relative_path in RUNTIME_WRAPPERS:
            with self.subTest(path=relative_path):
                tree = ast.parse(
                    (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
                )
                string_raises = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Raise)
                    and isinstance(node.exc, ast.Constant)
                    and isinstance(node.exc.value, str)
                ]
                typed_raises = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Raise)
                    and isinstance(node.exc, ast.Call)
                    and isinstance(node.exc.func, ast.Name)
                    and node.exc.func.id in {"ImportError", "RuntimeError"}
                ]

                self.assertEqual(string_raises, [])
                self.assertGreaterEqual(len(typed_raises), 3)
                self.assertTrue(all(node.cause is not None for node in typed_raises))


if __name__ == "__main__":
    unittest.main()
