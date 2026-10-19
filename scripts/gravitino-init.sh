#!/bin/sh
set -e

GRAVITINO_URL="http://gravitino:8090"
METALAKE="lakehouse"
CATALOG="iceberg"

echo "Waiting for Gravitino..."
until curl -sf "${GRAVITINO_URL}/api/metalakes" >/dev/null 2>&1; do
  sleep 2
done

echo "Creating metalake '${METALAKE}'..."
curl -s -X POST \
  -H "Accept: application/vnd.gravitino.v1+json" \
  -H "Content-Type: application/json" \
  -d "{\"name\": \"${METALAKE}\", \"comment\": \"AI Lakehouse prototype\"}" \
  "${GRAVITINO_URL}/api/metalakes" \
  | grep -q '"code":0\|already exists' && echo "  ok" || echo "  (may already exist, continuing)"

echo "Creating catalog '${CATALOG}' (Iceberg, JDBC-backend, MinIO warehouse)..."
curl -s -X POST \
  -H "Accept: application/vnd.gravitino.v1+json" \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"${CATALOG}\",
    \"type\": \"RELATIONAL\",
    \"comment\": \"Iceberg tables on MinIO\",
    \"provider\": \"lakehouse-iceberg\",
    \"properties\": {
      \"catalog-backend\": \"jdbc\",
      \"uri\": \"jdbc:postgresql://lh-iceberg-pg:5432/iceberg\",
      \"jdbc-user\": \"iceberg\",
      \"jdbc-password\": \"iceberg\",
      \"jdbc-driver\": \"org.postgresql.Driver\",
      \"warehouse\": \"s3a://lakehouse/\",
      \"io-impl\": \"org.apache.iceberg.aws.s3.S3FileIO\",
      \"s3-endpoint\": \"http://lh-minio:9000\",
      \"s3-access-key-id\": \"minioadmin\",
      \"s3-secret-access-key\": \"minioadmin\",
      \"s3-region\": \"us-east-1\",
      \"s3-path-style-access\": \"true\"
    }
  }" \
  "${GRAVITINO_URL}/api/metalakes/${METALAKE}/catalogs" \
  | grep -q '"code":0\|already exists' && echo "  ok" || echo "  (may already exist, or failed — check gravitino logs)"

echo "Gravitino init done."
