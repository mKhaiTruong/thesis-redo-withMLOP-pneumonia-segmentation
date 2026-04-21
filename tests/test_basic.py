def test_sanity():
    assert 1 + 1 == 2

def test_imports():
    from core.logging import logger
    assert logger is not None