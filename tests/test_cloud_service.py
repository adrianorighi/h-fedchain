import pytest
from services.cloud_service import CloudService


def test_cloud_service_init():
    service = CloudService()
    assert service is not None
    assert service.n_expected_clusters == 1
    assert service.grpc_port == 50052
    assert service.learning_rate == 0.01
    assert service.converged is False
    assert service.global_weights is None
