import importlib


def test_edge_client_module_imports():
    import services.edge_client as ec
    assert ec is not None
    assert hasattr(ec, "main")
    assert callable(ec.main)
