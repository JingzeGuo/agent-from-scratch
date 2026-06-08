import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


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
                except Exception:
                    if attempt == max_attempts:
                        raise
                    time.sleep(wait_time)
                    wait_time *= backoff

            raise RuntimeError("Retry loop ended unexpectedly")

        return wrapper

    return decorator
