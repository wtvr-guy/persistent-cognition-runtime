"""Pure policy tests exercise LLM=null without requiring PostgreSQL."""
import pytest


@pytest.fixture(scope="session")
def _prepare_test_database():
    yield


@pytest.fixture
def _reset_test_database():
    yield
