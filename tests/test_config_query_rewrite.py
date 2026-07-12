from src.config import Settings


def test_query_rewrite_is_disabled_by_default():
    assert Settings().enable_query_rewrite is False
