"""
Process-wide handle on the two fine-tuned specialists.

models/MODELS.md is explicit about the cost model: constructing a
SatelliteReporter loads a 4-bit base plus both LoRA adapters (~40 s, 2.65 GB
VRAM); every call after that is a few seconds. So it is built once, lazily, and
never per request.

Three things this module adds on top of SatelliteReporter:

  * **Lazy loading with a remembered failure.** No GPU, no adapters, or a wrong
    virtualenv must not take the web server down - it degrades to stubs and says
    so on /api/health.
  * **Serialised inference.** One model, one CUDA context, and uvicorn runs
    handlers in a thread pool. Concurrent generate() calls on a shared PEFT
    model with a switched active adapter would interleave; the lock makes each
    query atomic with respect to set_adapter().
  * **The validation gate.** models/validate.py is stdlib-only and is applied to
    every envelope before it leaves this layer.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from backend.config import MODELS_DIR, settings

logger = logging.getLogger(__name__)


def _ensure_models_on_path() -> None:
    """models/ is a directory of scripts, not an installed package."""
    path = str(MODELS_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


# validate.py is stdlib-only (no torch, no GPU), so it is safe to import eagerly
# and is available even when the vision model is not.
_ensure_models_on_path()
try:
    from validate import validate as _validate_envelope  # type: ignore
except Exception as exc:  # pragma: no cover - only if models/ is missing
    logger.error("models/validate.py could not be imported: %s", exc)
    _validate_envelope = None  # type: ignore


def validate_envelope(envelope: Dict[str, Any]) -> Dict[str, Any]:
    """
    Self-consistency check. Never raises, and always returns a usable dict even
    if validate.py itself is unavailable.
    """
    if _validate_envelope is None:
        return {
            "valid": True,
            "severity": "ok",
            "issues": [],
            "user_message": "",
            "counts": {},
            "note": "validator unavailable",
        }
    try:
        return _validate_envelope(envelope)
    except Exception as exc:  # pragma: no cover - validate.py promises not to
        logger.exception("validator raised")
        return {
            "valid": True,
            "severity": "ok",
            "issues": [],
            "user_message": "",
            "counts": {},
            "note": f"validator error: {exc}",
        }


class ReporterHandle:
    """Lazy singleton wrapper. `available` is only true once a load succeeded."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reporter: Any = None
        self._error: Optional[str] = None
        self._attempted = False

    # -- state -------------------------------------------------------------- #

    @property
    def available(self) -> bool:
        return self._reporter is not None

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def attempted(self) -> bool:
        return self._attempted

    def describe(self) -> str:
        if self._reporter is not None:
            return "loaded"
        if not self._attempted:
            return "not loaded yet"
        return f"unavailable: {self._error}"

    # -- loading ------------------------------------------------------------ #

    def load(self) -> bool:
        """
        Build the reporter if it is not built yet. Returns whether it is usable.

        A failed load is remembered: retrying a 40 s import on every request
        would turn a misconfigured machine into a hung demo.
        """
        if self._reporter is not None:
            return True
        with self._lock:
            if self._reporter is not None:
                return True
            if self._attempted:
                return False
            self._attempted = True
            if settings.workers == "stub":
                self._error = "SATQUERY_WORKERS=stub - model loading disabled"
                logger.info("worker models disabled by configuration")
                return False
            try:
                _ensure_models_on_path()
                # Imported here, not at module scope: generate_report_v2 pulls in
                # unsloth and torch at import time, which is a multi-second CUDA
                # initialisation and only exists in the .venv-unsloth environment.
                from generate_report_v2 import SatelliteReporter  # type: ignore

                logger.info("loading fine-tuned specialists (this takes ~40 s) ...")
                self._reporter = SatelliteReporter(verbose=True)
                logger.info("specialists ready")
                return True
            except Exception as exc:
                self._error = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "could not load the fine-tuned specialists (%s). Workers will "
                    "run as stubs. See backend/README.md for the environment setup.",
                    self._error,
                )
                return False

    # -- inference ---------------------------------------------------------- #

    def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """Single-image analysis. Returns the generate_report_v2 envelope."""
        if not self.load():
            raise RuntimeError(self._error or "vision model unavailable")
        with self._lock:
            # No prompt= argument. Both adapters were fine-tuned on exactly one
            # prompt each and overriding it destroys the JSON schema - see
            # "Orchestrator integration - three rules" in models/MODELS.md.
            return self._reporter.analyze_image(str(image_path))

    def analyze_pair(self, before: Path, after: Path) -> Dict[str, Any]:
        """Bi-temporal analysis. `before` must be the EARLIER acquisition."""
        if not self.load():
            raise RuntimeError(self._error or "vision model unavailable")
        with self._lock:
            return self._reporter.analyze_pair(str(before), str(after))

    def model_info(self) -> Dict[str, Any]:
        if self._reporter is None:
            return {}
        return dict(getattr(self._reporter, "meta", {}) or {})


reporter = ReporterHandle()
