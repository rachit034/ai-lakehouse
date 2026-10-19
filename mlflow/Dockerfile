FROM python:3.11-slim

RUN pip install --no-cache-dir mlflow==3.14.0

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--default-artifact-root", "/mlflow-data/artifacts", \
     "--allowed-hosts", "lh-mlflow:5000,mlflow:5000,localhost:5050,localhost:5000"]
