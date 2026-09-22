"""TypeSafe Jev research module."""

from pm_research.research.jev_openrouter import (
    CONDITION_BLIND,
    CONDITION_MARKET_AWARE,
    JEV_MODEL_PIN,
    JEV_SCHEMA_VERSION,
    JevCaptureRecord,
    JevForecast,
    JevOpenRouterClient,
    JevResolutionScore,
)

__all__ = [
    "CONDITION_BLIND",
    "CONDITION_MARKET_AWARE",
    "JEV_MODEL_PIN",
    "JEV_SCHEMA_VERSION",
    "JevCaptureRecord",
    "JevCycleSummary",
    "JevForecast",
    "JevOpenRouterClient",
    "JevResolutionScore",
    "JevResolutionScorer",
    "JevScoringSummary",
    "JevShadowRunner",
]


def __getattr__(name: str):
    if name in {
        "JevCycleSummary",
        "JevResolutionScorer",
        "JevScoringSummary",
        "JevShadowRunner",
    }:
        import pm_research.research.jev_shadow as js

        return getattr(js, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


