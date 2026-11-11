# fair-football-fee — common commands (macOS/Linux). Uses the project virtualenv.
PY ?= .venv/bin/python

.PHONY: doctor help db-up db-down migrate load-db db ingest clean-data features pipeline test test-unit lint train train-retune report calibrate register

doctor:
	@$(PY) -c "import xgboost, lightgbm; print('xgboost', xgboost.__version__, '| lightgbm', lightgbm.__version__, '| native libs OK')" \
		|| echo "Native ML libraries failed to load. On macOS run: brew install libomp"
	@docker info >/dev/null 2>&1 && echo "docker OK" || echo "Docker isn't running: open -a Docker"

help:
	@echo "doctor       check native ML libraries (libomp) and Docker"
	@echo "db           start postgres+redis, migrate, load clean tables and serving features"
	@echo "pipeline     ingest -> clean -> features (Parquet only, no database)"
	@echo "train        Phase 3: baselines + XGBoost + report (reuses committed tuning.json)"
	@echo "train-retune Phase 3 with a fresh 80-trial Optuna search"
	@echo "calibrate    Phase 4: TreeSHAP checks, conformal intervals, recentering, report + model card"
	@echo "register     write the model to model_versions and mark it active (needs make db)"
	@echo "test / lint  full pytest suite / ruff"

db-up:
	@docker info >/dev/null 2>&1 || { echo "Docker isn't running. Start Docker Desktop (open -a Docker), wait for it, then retry."; exit 1; }
	docker compose up -d --wait postgres redis

db-down:
	docker compose down

migrate:
	$(PY) -m alembic upgrade head

load-db:
	$(PY) -m src.data.clean --load-db
	$(PY) -m src.features.pipeline --load-db

db: db-up migrate load-db

ingest:
	$(PY) -m src.data.ingest

clean-data:
	$(PY) -m src.data.clean

features:
	$(PY) -m src.features.pipeline

pipeline: ingest clean-data features

train:
	$(PY) -m src.ml.train --reuse-tuning

train-retune:
	rm -f artifacts/v1.0.0/optuna.db artifacts/v1.0.0/optuna.db-journal artifacts/v1.0.0/tuning.json
	$(PY) -m src.ml.train --trials 80

report:
	$(PY) -m src.ml.report artifacts/v1.0.0
	$(PY) -m src.ml.report_phase4 artifacts/v1.0.0

calibrate:
	$(PY) -m src.ml.calibrate --model-version v1.0.0

register:
	$(PY) -m src.ml.registry v1.0.0 --activate

test:
	$(PY) -m pytest -q

test-unit:
	$(PY) -m pytest -q -m "not data and not db"

lint:
	$(PY) -m ruff check src tests migrations
