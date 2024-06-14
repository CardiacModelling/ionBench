from functools import lru_cache
import copy
import ionbench

cachedFunctions = []


def cache():
    def decorator(func):
        func = lru_cache(maxsize=None)(func)
        cachedFunctions.append(func)
        return func
    return decorator


def clear_all_caches():
    for func in cachedFunctions:
        func.cache_clear()


def get_cached_cost(bm):
    """
    Returns a cached version of the bm.cost() function for the inputted benchmark.

    Parameters
    ----------
    bm : benchmarker
        Benchmarker object.

    Returns
    -------
    cost_func : function
        Cached cost function.
    """
    if ionbench.cache_enabled:
        @cache()
        def cached_func(p):
            return bm.cost(p)
    else:  # pragma: no cover
        def cached_func(p):
            return bm.cost(p)

    def cost_func(p):
        return cached_func(tuple(p))

    return cost_func


def get_cached_signed_error(bm):
    """
    Returns a cached version of the bm.signed_error() function for the inputted benchmark.

    Parameters
    ----------
    bm : benchmarker
        Benchmarker object.

    Returns
    -------
    signed_error : function
        Cached signed error (residuals) function.
    """

    if ionbench.cache_enabled:
        @cache()
        def cached_func(p):
            return bm.signed_error(p)
    else:  # pragma: no cover
        def cached_func(p):
            return bm.signed_error(p)

    def signed_error(p):
        return cached_func(tuple(p))

    return signed_error


def get_cached_grad(bm, **kwargs):
    """
    Returns a cached version of the bm.grad() function for the inputted benchmark.

    Parameters
    ----------
    bm : benchmarker
        Benchmarker object.
    kwargs : dict
        Keyword arguments to pass to the gradient function.

    Returns
    -------
    grad : function
        Cached grad function.
    """

    if ionbench.cache_enabled:
        @cache()
        def cached_func(p):
            return bm.grad(p, **kwargs)
    else:  # pragma: no cover
        def cached_func(p):
            return bm.grad(p, **kwargs)

    def grad(p):
        return copy.deepcopy(cached_func(tuple(p)))

    return grad
