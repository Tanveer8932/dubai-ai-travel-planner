"""Dubai AI Travel Planner — multi-agent itinerary planning (LangGraph vs CrewAI)."""

from __future__ import annotations

from dubai_planner.config import Settings, get_settings
from dubai_planner.logging_config import configure_logging

__version__ = "0.1.0"
__all__ = ["Settings", "__version__", "bootstrap", "get_settings"]


def bootstrap() -> Settings:
    """Initialise the process: load settings, configure logging, export telemetry opt-outs.

    Every entry point (Streamlit app, scripts, tests) calls this before importing
    orchestration frameworks.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.apply_runtime_env()
    return settings
