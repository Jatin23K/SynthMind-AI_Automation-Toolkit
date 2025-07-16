import pytest
from dotenv import load_dotenv
import sys
import os

@pytest.fixture(scope='session', autouse=True)
def load_env():
    load_dotenv()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)