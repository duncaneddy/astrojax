"""Local conftest for tests/nondim/.

This file's presence causes pytest to add the directory to sys.path
so sibling helper modules (e.g. _brahe_reference.py) are importable
without a sys.path shim.
"""
