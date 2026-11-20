"""Backwards-compatible alias for the Monte Carlo driver module.

Maintains patching behavior for legacy tests and callers that import src.model.
New code should use :mod:`src.driver.monte_carlo_driver` directly.
"""
import importlib as _importlib
import sys as _sys
_module = _importlib.import_module("src.driver.monte_carlo_driver")
_sys.modules[__name__] = _module
