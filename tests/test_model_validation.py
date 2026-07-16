from core.cloud.model_validation import ModelValidationGate


def test_validates_good_model():
    gate = ModelValidationGate(loss_max=10.0, delta_conv=100.0)
    w_old = {"W1": [[0.1, 0.2], [0.3, 0.4]]}
    w_new = {"W1": [[0.11, 0.21], [0.31, 0.41]]}
    result = gate.validate(w_old, w_new, loss=0.5)
    assert result.accepted is True


def test_rejects_high_loss():
    gate = ModelValidationGate(loss_max=10.0)
    result = gate.validate({}, {}, loss=15.0)
    assert result.accepted is False
    assert "loss" in result.reason


def test_rejects_large_delta():
    gate = ModelValidationGate(delta_conv=1.0)
    w_old = {"W1": [[0.0, 0.0]]}
    w_new = {"W1": [[100.0, 100.0]]}
    result = gate.validate(w_old, w_new, loss=0.5)
    assert result.accepted is False
    assert "delta_norm" in result.reason


def test_accepts_no_old_weights():
    gate = ModelValidationGate()
    result = gate.validate(None, {"W1": [[1.0, 2.0]]}, loss=0.5)
    assert result.accepted is True


def test_delta_norm_zero_with_no_common_keys():
    gate = ModelValidationGate()
    result = gate.validate({"W1": [1.0]}, {"W2": [2.0]}, loss=0.5)
    assert result.accepted is True
