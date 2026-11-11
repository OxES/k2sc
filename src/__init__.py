"""K2SC: Kepler/K2 and TESS light curve systematics correction with JAX Gaussian Processes."""

from ._jax_env import ensure_float64

ensure_float64()

name = "K2SC"
__version__ = "2.0.0"

__all__ = ["K2SCCorrector", "K2SCResult", "__version__", "detrend", "name"]


def __getattr__(attr: str):
    if attr in ("detrend", "K2SCResult"):
        try:
            from . import pipeline
        except ModuleNotFoundError as exc:
            if exc.name == f"{__name__}.pipeline":
                raise AttributeError(f"module {__name__!r} has no attribute {attr!r}") from exc
            raise
        return getattr(pipeline, attr)
    if attr == "K2SCCorrector":
        try:
            from . import corrector
        except ModuleNotFoundError as exc:
            if exc.name == f"{__name__}.corrector":
                raise AttributeError(f"module {__name__!r} has no attribute {attr!r}") from exc
            raise
        return corrector.K2SCCorrector
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")
