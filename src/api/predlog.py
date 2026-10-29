"""`predictions_log` writes, run as a background task after the response is sent."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from src.api.repository import insert_prediction_log

log = logging.getLogger(__name__)


class PredictionLogWriter:
    """Opens its own session, as the request's session is already closed when this runs."""

    def __init__(self, session_factory: Callable[[], Session]):
        self.session_factory = session_factory

    def __call__(self, entry: dict[str, Any]) -> None:
        try:
            with self.session_factory() as db:
                insert_prediction_log(db, entry)
                db.commit()
        except SQLAlchemyError:
            # The answer was already sent; a lost audit row is logged, never turned into an error
            log.exception("could not write predictions_log row for request %s", entry.get("request_id"))
