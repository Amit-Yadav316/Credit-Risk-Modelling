.PHONY: setup data mlflow train test lint api ui docker-up docker-down

setup:
	pip install -r requirements.txt && pip install -e .

data:
	python scripts/make_synthetic_data.py

mlflow:
	mlflow server --host 127.0.0.1 --port 5000 \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns

train:
	python -m credit_risk.pipelines.train_pipeline

test:
	pytest -q

lint:
	ruff check src api app tests scripts

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
	streamlit run app/streamlit_app.py

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v
