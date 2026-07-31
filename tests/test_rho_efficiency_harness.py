import importlib.util
from pathlib import Path


HARNESS_PATH = (
    Path(__file__).resolve().parents[1]
    / 'experiments'
    / 'efficiency-amazon-tfinance-019fad8b'
    / 'rho_efficiency.py'
)


def load_harness():
    spec = importlib.util.spec_from_file_location('rho_efficiency_harness', HARNESS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_current_has_no_external_scope_dependencies(monkeypatch):
    harness = load_harness()
    monkeypatch.setattr(harness.torch.cuda, 'memory_allocated', lambda device: 123)
    monkeypatch.setattr(harness.torch.cuda, 'memory_reserved', lambda device: 456)

    assert harness.cuda_current(object()) == {
        'allocated_bytes': 123,
        'reserved_bytes': 456,
    }
