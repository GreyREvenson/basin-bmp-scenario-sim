"""Compatibility import for the driver-managed results module.

New code should import :mod:`src.driver.results.bmp_summary` directly.
"""
import importlib as _importlib
import sys as _sys
_sys.modules[__name__] = _importlib.import_module("src.driver.results.bmp_summary")
