"""
utils.py — Shared utilities for the Redrob ranking engine.

Provides:
  - Configuration loader (YAML → dataclass)
  - Structured logging via loguru
  - Text normalization helpers
  - Profiling decorators
  - Memory monitoring
"""

from __future__ import annotations

import functools
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

import psutil
import yaml
from loguru import logger


# ── Configuration ────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent / "config.yaml"
_config_cache: dict[str, Any] | None = None


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """
    Load and cache the YAML configuration file.

    Args:
        path: Override path to config.yaml. Defaults to ./config.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    config_path = Path(path) if path else _CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        _config_cache = yaml.safe_load(fh)

    logger.info(f"Configuration loaded from {config_path}")
    return _config_cache


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Retrieve a nested config value using dot notation.

    Example:
        get_config_value("weights.semantic_similarity")  → 0.40
    """
    cfg = load_config()
    keys = key_path.split(".")
    value = cfg
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default


# ── Logging Setup ─────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    """Configure loguru with a sensible format for the application."""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    # Also log to file
    log_dir = Path(__file__).parent.parent / "output" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        sink=str(log_dir / "ranker_{time:YYYY-MM-DD}.log"),
        level="DEBUG",
        rotation="1 day",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} — {message}",
    )


# ── Text Normalization ────────────────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALPHANUM_RE = re.compile(r"[^a-z0-9\s]")


def normalize_text(text: str) -> str:
    """
    Lowercase, strip punctuation, and collapse whitespace.

    Args:
        text: Raw input string.

    Returns:
        Cleaned, lowercased string.
    """
    if not text:
        return ""
    text = text.lower()
    text = _NON_ALPHANUM_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def normalize_skill_name(name: str) -> str:
    """
    Normalize a skill name for fuzzy matching.
    Keeps dots for abbreviations like 'C++', 'Node.js'.
    """
    if not name:
        return ""
    return name.lower().strip()


def clean_text_for_embedding(parts: list[str]) -> str:
    """
    Join non-empty text parts into a single embedding-ready string.

    Args:
        parts: List of text segments (some may be None/empty).

    Returns:
        Single joined string, max ~512 tokens worth.
    """
    joined = " ".join(p for p in parts if p and p.strip())
    # Sentence transformers truncate at 256 tokens for MiniLM-L6
    # Rough approximation: 1 token ≈ 4 chars; 256 tokens ≈ 1024 chars
    return joined[:1024]


def coerce_float(value: Any, default: float = 0.0) -> float:
    """Safely coerce any value to float, returning default on failure."""
    try:
        v = float(value)
        return v if not (v != v) else default  # NaN check
    except (TypeError, ValueError):
        return default


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float to [lo, hi]."""
    return max(lo, min(hi, value))


# ── Profiling Decorator ───────────────────────────────────────────────────────

def timed(label: str | None = None) -> Callable:
    """
    Decorator that logs the elapsed time of a function call.

    Usage:
        @timed("embedding")
        def embed_candidates(...): ...
    """
    def decorator(fn: Callable) -> Callable:
        tag = label or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            logger.info(f"[{tag}] completed in {elapsed:.2f}s")
            return result

        return wrapper

    return decorator


def timed_block(label: str):
    """
    Context manager for timing arbitrary code blocks.

    Usage:
        with timed_block("scoring"):
            ...
    """
    import contextlib

    @contextlib.contextmanager
    def _inner():
        t0 = time.perf_counter()
        yield
        logger.info(f"[{label}] completed in {time.perf_counter() - t0:.2f}s")

    return _inner()


# ── Memory Monitoring ────────────────────────────────────────────────────────

def get_memory_usage_mb() -> float:
    """Return current process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def log_memory(label: str = "") -> None:
    """Log current memory usage."""
    mb = get_memory_usage_mb()
    prefix = f"[{label}] " if label else ""
    logger.debug(f"{prefix}Memory usage: {mb:.1f} MB")


# ── Output Directory Helper ──────────────────────────────────────────────────

def ensure_output_dir() -> Path:
    """Create and return the output directory."""
    cfg = load_config()
    out_dir = Path(__file__).parent / cfg["data"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir
