import pytest
import torch

from PPOAgent import PPOAgent


def test_component_return_contract_allows_only_cancellation_scaled_roundoff():
    components = torch.tensor([[800.0, 799.0]], dtype=torch.float32)
    weights = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    reconstructed = (components * weights).sum(-1)

    _, small_error, tolerance = PPOAgent._component_return_consistency(
        components, weights, reconstructed + 0.001
    )
    assert bool((small_error <= tolerance).all())

    _, material_error, tolerance = PPOAgent._component_return_consistency(
        components, weights, reconstructed + 0.1
    )
    assert bool((material_error > tolerance).all())


def test_component_return_contract_keeps_original_small_scale_strictness():
    components = torch.tensor([[0.7, 0.2]], dtype=torch.float32)
    weights = torch.tensor([[1.0, -0.5]], dtype=torch.float32)
    reconstructed = (components * weights).sum(-1)

    _, error, tolerance = PPOAgent._component_return_consistency(
        components, weights, reconstructed + 0.001
    )
    assert float(tolerance.item()) < 0.001
    assert bool((error > tolerance).all())


def test_component_return_contract_reports_nonfinite_dtype_assumption():
    components = torch.tensor([[1.0, 0.5]], dtype=torch.float64)
    weights = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
    target = torch.tensor([0.75], dtype=torch.float64)
    reconstructed, error, tolerance = PPOAgent._component_return_consistency(
        components, weights, target
    )
    torch.testing.assert_close(reconstructed, target)
    assert error.item() == pytest.approx(0.0)
    assert tolerance.item() >= 2e-4
