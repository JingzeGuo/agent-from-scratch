import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

import httpx

P = ParamSpec("P")
R = TypeVar("R")


def is_transient_error(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        return status_code in {408, 429} or status_code >= 500

    return isinstance(
        error,
        (TimeoutError, ConnectionError, httpx.TransportError),
    )


def retry(
    max_attempts: int = 3,
    backoff: float = 2.0,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if backoff < 1:
        raise ValueError("backoff must be at least 1")

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            wait_time = 1.0

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    if not is_transient_error(e):
                        raise
                    time.sleep(wait_time)
                    wait_time *= backoff

            raise RuntimeError("Retry loop ended unexpectedly")

        return wrapper

    return decorator
