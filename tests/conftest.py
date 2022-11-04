import pytest


@pytest.fixture
def text_message() -> str:
    return 'hello world'


@pytest.fixture(scope='session')
def ranged_message() -> list:
    return list(range(10))
