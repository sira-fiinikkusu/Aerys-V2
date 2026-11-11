"""Exact raw context reads scoped to one action invocation."""
from functools import wraps


def _key(value):
    if isinstance(value, dict):
        return (dict, frozenset((_key(k), _key(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value), tuple(_key(v) for v in value))
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return (type(value), value)
    raise TypeError("unsupported memo argument")


def memoized(fn, memo, *, cache_if=bool, namespace=None):
    """Wrap only the read; callers still apply fences and side effects each pass.

    Empty results may represent swallowed failures, so retry them. Raised
    errors never reach the cache. Unsupported argument types fall back to a read.
    A stable namespace identifies a seam whose lazy reader is rebuilt each pass.
    """
    if fn is None or memo is None:
        return fn

    @wraps(fn)
    def read(*args, **kwargs):
        try:
            key = (fn if namespace is None else namespace, _key(args), _key(kwargs))
            hash(key)
        except TypeError:
            return fn(*args, **kwargs)
        if key not in memo:
            value = fn(*args, **kwargs)
            if cache_if(value):
                memo[key] = value
            return value
        return memo[key]

    return read
