"""Which model and data the API serves right now.

The active model comes from `model_versions.is_active` and the data version from
`player_current_features`; both are re-read every `metadata_refresh_seconds`, never taken from
config. A model registered with `make register` is picked up within one refresh. Cache keys carry
both versions, so an answer from the previous model or data can't be served after a switch.

Valuations are refused (HTTP 503) while the state has problems: no active model, more than one
data version in the serving table, a missing artifact, or a recentering offset calibrated on
different data than the serving rows (run `make calibrate && make register`).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.api.repository import active_model, serving_data_versions
from src.ml.predictor import Predictor

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServingState:
    model_version: str | None
    data_version: str | None
    predictor: Predictor | None
    model_info: dict[str, Any] = field(default_factory=dict)
    problems: tuple[str, ...] = ()


class Runtime:
    def __init__(
        self,
        session_factory: Callable[[], Session],
        refresh_seconds: float,
        loader: Callable[[Path, dict[str, Any]], Predictor] = Predictor.from_artifacts,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.session_factory = session_factory
        self.refresh_seconds = refresh_seconds
        self.loader = loader
        self.clock = clock
        self._lock = threading.Lock()
        self._state: ServingState | None = None
        self._checked_at = float("-inf")
        self._predictor_key: tuple[str, str] | None = None

    def state(self) -> ServingState:
        if self._state is None or self.clock() - self._checked_at >= self.refresh_seconds:
            with self._lock:
                if self._state is None or self.clock() - self._checked_at >= self.refresh_seconds:
                    self._state = self._refresh(self._state)
                    self._checked_at = self.clock()
        return self._state

    def _refresh(self, previous: ServingState | None) -> ServingState:
        try:
            with self.session_factory() as db:
                row = active_model(db)
                versions = serving_data_versions(db)
                model = None if row is None else {
                    "model_version": row.model_version,
                    "model_type": row.model_type,
                    "training_data_version": row.data_version,
                    "trained_at": row.trained_at.isoformat(),
                    "artifact_path": row.artifact_path,
                    "feature_list": list(row.feature_list),
                    "metrics": dict(row.metrics),
                    "conformal": dict(row.conformal_quantiles),
                }
        except SQLAlchemyError as exc:
            log.warning("could not refresh serving state: %s", exc)
            if previous is not None:
                return previous  # keep serving what we had; try again next refresh
            return ServingState(None, None, None, problems=("database unreachable",))

        problems: list[str] = []
        if model is None:
            problems.append("no active model in model_versions (run `make register`)")
        if len(versions) != 1:
            problems.append(f"player_current_features holds {len(versions)} data versions; expected exactly 1")
        data_version = versions[0] if len(versions) == 1 else None
        if model and data_version and model["conformal"].get("data_version") != data_version:
            problems.append(
                f"intervals and offset were calibrated on data {model['conformal'].get('data_version')} but serving "
                f"data is {data_version}: run `make calibrate && make register`"
            )

        predictor = None
        if model is not None:
            key = (model["model_version"], str(model["conformal"].get("created_at")))
            if previous is not None and previous.predictor is not None and key == self._predictor_key:
                predictor = previous.predictor
            else:
                artifact_dir = Path(model["artifact_path"]).parent
                try:
                    predictor = self.loader(artifact_dir, model["conformal"])
                    self._predictor_key = key
                    log.info("loaded model %s from %s", model["model_version"], artifact_dir)
                except (OSError, ValueError, KeyError, RuntimeError) as exc:  # RuntimeError: native lib missing
                    problems.append(f"cannot load model artifact from {artifact_dir}: {exc}")
        return ServingState(
            model_version=model["model_version"] if model else None,
            data_version=data_version,
            predictor=predictor,
            model_info=model or {},
            problems=tuple(problems),
        )
