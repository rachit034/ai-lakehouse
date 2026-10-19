#!/bin/sh
set -e

echo "Waiting for MinIO..."
until mc alias set local http://minio:9000 minioadmin minioadmin 2>/dev/null; do
  sleep 2
done

echo "Creating buckets..."
mc mb local/lakehouse   --ignore-existing   # Iceberg table data (mirrors S3 in prod)
mc mb local/lance       --ignore-existing   # LanceDB vector datasets
mc mb local/athena-results --ignore-existing # DuckDB/Athena query results

echo "Loading sample data..."
if [ -f /sample_data/products.csv ]; then
  mc cp /sample_data/products.csv local/lakehouse/raw/products/products.csv
  echo "  Uploaded products.csv"
fi

echo "MinIO init done."
