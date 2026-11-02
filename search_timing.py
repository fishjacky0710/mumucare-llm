"""Content-free, request-correlated search timing logs."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import inspect
import logging
import time


request_id = ContextVar("search_request_id", default=None)
logger = logging.getLogger("mumucare.search_timing")
logger.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.propagate = False


@contextmanager
def stage(name, **counts):
    correlation = request_id.get()
    started = time.perf_counter()
    def emit(event, **fields):
        if correlation is not None:
            logger.info(json.dumps({"event": event, "request_id": correlation,
                                    "stage": name, **counts, **fields}))
    emit("search_stage_started")
    try:
        yield
    except BaseException as error:
        emit("search_stage_failed", elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
             error_type=type(error).__name__)
        raise
    else:
        emit("search_stage_completed", elapsed_ms=round((time.perf_counter() - started) * 1000, 2))


def timed(name):
    def decorate(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with stage(name):
                return fn(*args, **kwargs)
        wrapper.__signature__ = inspect.signature(fn, eval_str=True)
        return wrapper
    return decorate
