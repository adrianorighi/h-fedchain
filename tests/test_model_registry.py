from services.model_registry import ModelRegistry
from core.cloud.model_validation import ValidationResult


def test_register_and_retrieve():
    reg = ModelRegistry()
    vr = ValidationResult(accepted=True, loss=0.5, delta_norm=0.01)
    mv = reg.register({"W1": [0.1, 0.2]}, vr)
    assert mv.version == 1
    assert mv.model_hash.startswith("0"*5) or len(mv.model_hash) == 64
    latest = reg.get_latest()
    assert latest.version == 1


def test_rollback():
    reg = ModelRegistry()
    vr1 = ValidationResult(accepted=True, loss=0.5, delta_norm=0.01)
    vr2 = ValidationResult(accepted=True, loss=0.3, delta_norm=0.02)
    reg.register({"W1": [0.1]}, vr1)
    reg.register({"W1": [0.2]}, vr2)
    assert reg.get_latest().version == 2
    reg.rollback(1)
    assert reg.get_latest().version == 1


def test_list_versions():
    reg = ModelRegistry()
    vr = ValidationResult(accepted=True, loss=0.5, delta_norm=0.01)
    reg.register({"W1": [0.1]}, vr)
    reg.register({"W1": [0.2]}, vr)
    reg.register({"W1": [0.3]}, vr)
    assert reg.list_versions() == [1, 2, 3]
    assert reg.count() == 3


def test_get_version():
    reg = ModelRegistry()
    vr = ValidationResult(accepted=True, loss=0.5, delta_norm=0.01)
    reg.register({"W1": [0.1]}, vr)
    v = reg.get_version(1)
    assert v is not None
    assert v.version == 1
    assert reg.get_version(99) is None
