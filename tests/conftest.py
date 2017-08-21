import pytest
from sklearn import datasets


@pytest.fixture(scope='session')
def circles():
    return datasets.make_circles()
