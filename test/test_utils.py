import os

from src.utils import agnostic_path


def test_agnostic_path():
    if os.name != "nt":
        return
    assert agnostic_path("/mnt/c/path/to.txt") == "C:\\path\\to.txt"
    assert agnostic_path("/mnt/c/path/to") == "C:\\path\\to"
    assert agnostic_path("/mnt/c/path/to/") == "C:\\path\\to\\"
