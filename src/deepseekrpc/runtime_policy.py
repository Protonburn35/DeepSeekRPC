"""Explicit fidelity settings established before importing ExLlama kernels."""
import os
import sys
import threading

from .jsonutil import ValidationError

_mutex = threading.Lock()
_configured = False
_EXACT_ENV = {"EXL3_INT8_GEMV": "0", "EXL3_GR_INT8": "0"}


def configure_exact_exl3():
    """Disable activation/int8 gate shortcuts before their cached first use.

    The pinned Python and CUDA sources disagree on the unset GEMV default.
    An explicit zero before either import is required. No native execution or
    precision qualification is claimed merely by setting these variables.
    """
    global _configured
    with _mutex:
        if _configured:
            assert_exact_exl3()
            return dict(_EXACT_ENV)
        if any(name == "exllamav3_ext" or name == "exllamav3" or name.startswith("exllamav3.")
               for name in sys.modules):
            raise ValidationError("ExLlama was imported before exact policy; start a fresh worker process")
        os.environ.update(_EXACT_ENV)
        _configured = True
        return dict(_EXACT_ENV)


def assert_exact_exl3():
    if not _configured or any(os.environ.get(name) != value for name, value in _EXACT_ENV.items()):
        raise ValidationError("exact ExLlama policy is missing or changed after configuration")
