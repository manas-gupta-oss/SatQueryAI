"""
SatQueryAI backend - runtime configuration.

Everything tunable lives here and is overridable by environment variable, so a
demo machine never needs a code edit. Defaults are chosen for a *live demo on
one laptop*: nothing here blocks startup, and every heavyweight component
degrades to something honest rather than crashing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Settings:
    # --- storage ----------------------------------------------------------- #
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("SATQUERY_DATA_DIR", str(REPO_ROOT / "backend" / "_data"))
        ).resolve()
    )

    # --- router ------------------------------------------------------------ #
    #: "heuristic" - deterministic rule router, zero VRAM, instant. The demo default.
    #: "llm"       - the Qwen2.5-3B BOSS in orchestration/nodes/boss_node.py.
    #: Both emit the SAME BossDecision through the SAME deterministic gate
    #: (tool_schema.validate_decision), so routing behaviour is comparable.
    router: str = field(default_factory=lambda: os.getenv("SATQUERY_ROUTER", "heuristic").strip().lower())

    #: Path handed to the BOSS when router == "llm".
    boss_config_path: str = field(
        default_factory=lambda: os.getenv("SATQUERY_BOSS_CONFIG", str(REPO_ROOT / "configs" / "boss_config.yaml"))
    )

    # --- workers ----------------------------------------------------------- #
    #: "auto"  - use the fine-tuned adapters if they load, else fall back to stubs.
    #: "real"  - require the adapters; a load failure surfaces as a query error.
    #: "stub"  - never load a model. Useful for UI work on a machine with no GPU.
    workers: str = field(default_factory=lambda: os.getenv("SATQUERY_WORKERS", "auto").strip().lower())

    #: Load the vision model at startup instead of on the first query. Costs
    #: ~40 s of boot time and buys a fast first demo query.
    preload_models: bool = field(default_factory=lambda: _env_bool("SATQUERY_PRELOAD", False))

    # --- input policy ------------------------------------------------------ #
    #: Both specialists are benchmark-tuned (VRSBench / LEVIR-CC), which are PNG
    #: datasets, so PNG/JPEG uploads are the normal case here. tool_schema only
    #: accepts them when the bundle is flagged benchmark_mode, so this is on by
    #: default. Turn it off to enforce the GeoTIFF-only operational contract.
    benchmark_mode: bool = field(default_factory=lambda: _env_bool("SATQUERY_BENCHMARK_MODE", True))

    max_upload_bytes: int = field(
        default_factory=lambda: int(os.getenv("SATQUERY_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
    )

    # --- http -------------------------------------------------------------- #
    cors_origins: List[str] = field(
        default_factory=lambda: _env_list(
            "SATQUERY_CORS_ORIGINS",
            ["http://localhost:5173", "http://127.0.0.1:5173",
             "http://localhost:4173", "http://127.0.0.1:4173"],
        )
    )

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def overlays_dir(self) -> Path:
        return self.data_dir / "overlays"

    def ensure_dirs(self) -> None:
        for path in (self.uploads_dir, self.reports_dir, self.overlays_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
