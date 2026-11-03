"""Register a model version in `model_versions` and optionally make it the active one.

    python -m src.ml.registry v1.0.0 [--activate]

Reads `artifacts/<version>/{metadata,features,metrics,phase4_metrics,conformal}.json`. The API
resolves the active model from `model_versions.is_active` (a partial unique index allows only one
active row), so activation switches the previous model off and this one on in one transaction.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.config import settings
from src.db.models import ModelVersion
from src.db.session import SessionLocal


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def registry_record(version: str, artifacts_dir: Path | None = None, manifest_path: Path | None = None) -> dict[str, Any]:
    """The `model_versions` row for `version`, built only from its artifact JSON files."""
    out_dir = (artifacts_dir or settings.artifacts_dir) / version
    meta, spec = _read(out_dir / "metadata.json"), _read(out_dir / "features.json")
    m3, p4, conformal = _read(out_dir / "metrics.json"), _read(out_dir / "phase4_metrics.json"), _read(out_dir / "conformal.json")
    current = _read(manifest_path or settings.raw_manifest_path)["data_version"]
    if conformal["data_version"] != current:
        raise ValueError(
            f"conformal.json was built on data {conformal['data_version']} but the current data is {current}: "
            "the recentering offset is stale, rerun `make calibrate`"
        )
    if not (out_dir / "model.ubj").exists():
        raise FileNotFoundError(f"{out_dir / 'model.ubj'} missing; run `make train`")

    chosen = p4["final"]["chosen"]
    metrics = {
        "test": chosen["point_metrics"],
        "test_without_recentering": p4["final"]["all_configs_test_record_only"]["global"]["point_metrics"],
        "val_calib_without_recentering": m3["stage_b"]["xgb_tuned"]["val_calib"],
        "test_coverage": {lv: {"coverage": d["coverage"], "ci": d["coverage_ci"], "median_factor": d["median_factor"]}
                          for lv, d in chosen["levels"].items()},
        "best_baseline": {"name": m3["best_baseline"], "test": m3["stage_b"][m3["best_baseline"]]["test"]},
        "interval_config": p4["decision"]["config"],
    }
    return {
        "model_version": version,
        "model_type": meta["model_type"],
        "data_version": meta["data_version"],
        "trained_at": datetime.fromisoformat(meta["trained_at"]),
        "artifact_path": (out_dir / "model.ubj").as_posix(),
        "feature_list": spec["features"],
        "metrics": metrics,
        "conformal_quantiles": conformal,
    }


def register_model(version: str, activate: bool, session: Session | None = None, artifacts_dir: Path | None = None) -> dict[str, Any]:
    """Upsert the row; with `activate`, make it the only active model. Commits only its own session."""
    record = registry_record(version, artifacts_dir)
    owns_session = session is None
    db = session or SessionLocal()
    try:
        stmt = pg_insert(ModelVersion).values(record)
        stmt = stmt.on_conflict_do_update(
            index_elements=["model_version"],
            set_={k: stmt.excluded[k] for k in record if k != "model_version"},
        )
        db.execute(stmt)
        previous: list[str] = []
        if activate:
            previous = list(db.scalars(
                select(ModelVersion.model_version).where(ModelVersion.is_active.is_(True), ModelVersion.model_version != version)
            ))
            # Deactivate first: the partial unique index is checked per statement
            db.execute(update(ModelVersion).where(ModelVersion.model_version.in_(previous)).values(is_active=False))
            db.execute(update(ModelVersion).where(ModelVersion.model_version == version).values(is_active=True))
        db.flush()
        is_active = db.scalar(select(ModelVersion.is_active).where(ModelVersion.model_version == version))
        if owns_session:
            db.commit()
        return {"model_version": version, "is_active": bool(is_active), "deactivated": previous}
    except Exception:
        if owns_session:
            db.rollback()
        raise
    finally:
        if owns_session:
            db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_version")
    parser.add_argument("--activate", action="store_true", help="make this the only active model")
    args = parser.parse_args()
    print(json.dumps(register_model(args.model_version, args.activate), indent=2))


if __name__ == "__main__":
    main()
