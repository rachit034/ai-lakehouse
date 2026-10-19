#!/bin/sh
# Exercises the business-metrics layer (gateway/services/metrics.py) end to
# end against a running docker-compose stack. Re-run this any time you add
# or edit a metric — see gateway/services/metrics.py's module docstring.
#
# Requires: docker-compose stack up, API_KEY_BUSINESS set (from .env).
set -e

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8088}"
API_KEY="${API_KEY_BUSINESS:?Set API_KEY_BUSINESS (see .env) before running this script}"

echo "== 1. Schema contexts include the metrics block =="
docker-compose exec -T gateway python3 -c "
from services import catalog, graph_store
sql_ctx = catalog.build_schema_context(schema='ecommerce')
cypher_ctx = graph_store.build_schema_context()
assert 'Business metrics' in sql_ctx, 'SQL context missing metrics block'
assert 'Business metrics' in cypher_ctx, 'Cypher context missing metrics block'
print('OK - both schema contexts include the metrics block')
"

echo
echo "== 2. /query/metrics =="
curl -sf -H "X-API-Key: $API_KEY" "$GATEWAY_URL/query/metrics" | python3 -m json.tool

echo
echo "== 3. NL -> SQL uses the registered revenue expression =="
# Flush the semantic cache first — a pre-existing cached translation from
# before you edited metrics.py/llm.py would otherwise mask whether the new
# prompt rule actually took effect.
docker-compose exec -T redis redis-cli FLUSHDB >/dev/null
RESP=$(curl -sf -X POST "$GATEWAY_URL/query/nl" \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"question": "What is total revenue?", "schema": "ecommerce", "explain": true}')
echo "$RESP" | python3 -m json.tool
echo "$RESP" | python3 -c "
import json, sys
r = json.load(sys.stdin)
sql = r.get('generated_sql', '') or ''
ok = 'total_amount' in sql and 'status' in sql and 'completed' in sql
print('MATCHES REGISTERED METRIC' if ok else 'DOES NOT obviously match the registered metric -- inspect generated_sql above')
"

echo
echo "== 4/5. Cross-surface consistency (SQL vs. graph) =="
SQL_VALUE=$(echo "$RESP" | python3 -c "
import json, sys
r = json.load(sys.stdin)
print(r['rows'][0][0] if r.get('rows') else 'N/A')
")
echo "SQL surface revenue: $SQL_VALUE"
echo
echo "Now run the graph surface and compare by hand:"
echo "  cd agents && python3 graph_agent.py \"What's our total revenue?\""
echo "Check MLflow at http://localhost:5050 for the exact Cypher it ran, and"
echo "GET $GATEWAY_URL/graph/freshness?schema=ecommerce first -- a stale graph"
echo "will disagree with the SQL number for reasons that have nothing to do"
echo "with the metric definition."
