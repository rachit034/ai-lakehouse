FROM python:3.11-slim

WORKDIR /app

# System deps for fastembed (ONNX runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# fastembed downloads the model on first use — pre-download at build time
# so the first request isn't slow
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

# DuckDB extensions (iceberg, httpfs) download from DuckDB's extension repo on
# first LOAD — pre-install at build time so the container doesn't need
# runtime network access to it and the first request isn't slow.
RUN python -c "import duckdb; c = duckdb.connect(); c.execute('INSTALL iceberg'); c.execute('INSTALL httpfs')"

EXPOSE 8088

# --reload for hot reload during dev (volume-mounted source)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8088", "--reload"]
