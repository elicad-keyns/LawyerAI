import pytest
from src.config import Settings


def test_temperature_range():
    assert Settings(temperature=0.15).temperature == 0.15
    with pytest.raises(ValueError):
        Settings(temperature=2.1)
