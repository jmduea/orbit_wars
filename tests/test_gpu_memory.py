from __future__ import annotations

from unittest.mock import patch

from src.telemetry.gpu_memory import GpuMemoryTracker, probe_gpu_memory


def test_probe_gpu_memory_uses_nvidia_smi_output() -> None:
    with patch(
        "src.telemetry.gpu_memory.shutil.which",
        return_value="/usr/bin/nvidia-smi",
    ), patch(
        "src.telemetry.gpu_memory.subprocess.run",
        return_value=type(
            "Result",
            (),
            {"stdout": "NVIDIA Test GPU, 2048, 16384\n", "stderr": ""},
        )(),
    ):
        snapshot = probe_gpu_memory()

    assert snapshot is not None
    assert snapshot.gpu_name == "NVIDIA Test GPU"
    assert snapshot.memory_used_gb == 2.0
    assert snapshot.memory_total_gb == 16.0
    assert snapshot.source == "nvidia-smi"


def test_gpu_memory_tracker_tracks_running_peak() -> None:
    baseline = type(
        "Snap",
        (),
        {
            "gpu_name": "NVIDIA Test GPU",
            "memory_used_gb": 4.0,
            "memory_total_gb": 16.0,
            "source": "nvidia-smi",
        },
    )()
    higher = type(
        "Snap",
        (),
        {
            "gpu_name": "NVIDIA Test GPU",
            "memory_used_gb": 6.5,
            "memory_total_gb": 16.0,
            "source": "nvidia-smi",
        },
    )()
    lower = type(
        "Snap",
        (),
        {
            "gpu_name": "NVIDIA Test GPU",
            "memory_used_gb": 5.0,
            "memory_total_gb": 16.0,
            "source": "nvidia-smi",
        },
    )()
    with patch(
        "src.telemetry.gpu_memory.probe_gpu_memory",
        side_effect=[baseline, baseline, higher, lower],
    ):
        tracker = GpuMemoryTracker()
        first = tracker.sample_update_metrics()
        peak = tracker.sample_update_metrics()
        third = tracker.sample_update_metrics()

    assert first["gpu_memory_used_gb"] == 4.0
    assert peak["gpu_memory_peak_gb"] == 6.5
    assert third["gpu_memory_peak_gb"] == 6.5


def test_gpu_memory_tracker_run_metadata() -> None:
    with patch(
        "src.telemetry.gpu_memory.probe_gpu_memory",
        return_value=type(
            "Snap",
            (),
            {
                "gpu_name": "NVIDIA Test GPU",
                "memory_used_gb": 1.0,
                "memory_total_gb": 16.0,
                "source": "nvidia-smi",
            },
        )(),
    ):
        tracker = GpuMemoryTracker()

    assert tracker.run_metadata() == {
        "gpu_name": "NVIDIA Test GPU",
        "gpu_memory_total_gb": 16.0,
    }


def test_probe_gpu_memory_returns_none_without_gpu() -> None:
    with patch("src.telemetry.gpu_memory.shutil.which", return_value=None), patch(
        "src.telemetry.gpu_memory._probe_jax_memory_stats",
        return_value=None,
    ):
        assert probe_gpu_memory() is None
