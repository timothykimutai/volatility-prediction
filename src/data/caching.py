import os
import pickle
import time
from functools import wraps
import logging
from typing import TypeVar, Callable, Any

CACHE_DIR = ".cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

F = TypeVar('F', bound=Callable[..., Any])


def disk_cache(ttl_seconds: int) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key based on function name, args, and kwargs
            # This is a simple but effective way to uniquely identify a call
            arg_str = "_".join(map(str, args))
            kwarg_str = "_".join(f"{k}_{v}" for k, v in sorted(kwargs.items()))
            cache_filename = f"{func.__name__}_{arg_str}_{kwarg_str}.pkl"
            cache_filepath = os.path.join(CACHE_DIR, cache_filename)

            # Check if a valid cache file exists
            if os.path.exists(cache_filepath):
                try:
                    # Check if the cache file is still within its TTL
                    if time.time() - os.path.getmtime(cache_filepath) < ttl_seconds:
                        with open(cache_filepath, "rb") as f:
                            logging.info(f"Loading from cache: {cache_filename}")
                            return pickle.load(f)
                except (IOError, pickle.PickleError) as e:
                    logging.warning(f"Cache file corrupted or unreadable: {e}. Refetching.")
                    try:
                        os.remove(cache_filepath)
                    except OSError:
                        pass

            # If no valid cache, execute the function
            result = func(*args, **kwargs)

            # Save the result to the cache
            try:
                with open(cache_filepath, "wb") as f:
                    pickle.dump(result, f)
                    logging.info(f"Saved to cache: {cache_filename}")
            except (IOError, pickle.PickleError) as e:
                logging.error(f"Failed to write to cache file: {e}")

            return result

        return wrapper  # type: ignore[return-value]

    return decorator