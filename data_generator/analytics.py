"""
DuckDB analytics demo — query the generated Parquet files directly.
No database server required.

Mirrors: https://www.kdnuggets.com/building-your-modern-data-analytics-stack-with-python-parquet-and-duckdb

Run:
    python analytics.py
    python analytics.py --query revenue_by_category
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import duckdb

PARQUET_DIR = Path(__file__).parent / "parquet"


def _db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    # Register Parquet files as virtual tables — DuckDB reads only needed columns
    con.execute(f"""
        CREATE VIEW customers   AS SELECT * FROM read_parquet('{PARQUET_DIR}/customers.parquet');
        CREATE VIEW products    AS SELECT * FROM read_parquet('{PARQUET_DIR}/products.parquet');
        CREATE VIEW orders      AS SELECT * FROM read_parquet('{PARQUET_DIR}/orders.parquet');
        CREATE VIEW order_items AS SELECT * FROM read_parquet('{PARQUET_DIR}/order_items.parquet');
    """)
    return con


def run(label: str, sql: str, con: duckdb.DuckDBPyConnection):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    t0 = time.perf_counter()
    result = con.execute(sql).df()
    ms = (time.perf_counter() - t0) * 1000
    print(result.to_string(index=False))
    print(f"\n  ({len(result)} rows, {ms:.1f} ms)")


QUERIES = {

    "overview": (
        "Dataset overview",
        """
        SELECT
            'customers'   AS table_name, COUNT(*) AS row_count FROM customers
        UNION ALL SELECT 'products',   COUNT(*) FROM products
        UNION ALL SELECT 'orders',     COUNT(*) FROM orders
        UNION ALL SELECT 'order_items',COUNT(*) FROM order_items
        ORDER BY row_count DESC
        """
    ),

    "revenue_by_category": (
        "Revenue by product category (completed orders only)",
        """
        SELECT
            p.category,
            COUNT(DISTINCT o.order_id)          AS total_orders,
            SUM(oi.quantity)                    AS units_sold,
            ROUND(SUM(oi.line_total), 2)        AS gross_revenue,
            ROUND(AVG(oi.unit_price), 2)        AS avg_unit_price,
            ROUND(SUM(oi.line_total)
                  / SUM(SUM(oi.line_total)) OVER () * 100, 1) AS revenue_share_pct
        FROM order_items oi
        JOIN orders   o  ON oi.order_id  = o.order_id
        JOIN products p  ON oi.product_id = p.product_id
        WHERE o.status = 'completed'
        GROUP BY p.category
        ORDER BY gross_revenue DESC
        """
    ),

    "monthly_trend": (
        "Monthly revenue trend (2022–2024)",
        """
        SELECT
            order_year,
            order_month,
            COUNT(DISTINCT order_id)            AS orders,
            ROUND(SUM(total_amount), 2)         AS revenue,
            ROUND(AVG(total_amount), 2)         AS avg_order_value,
            ROUND(SUM(SUM(total_amount)) OVER (
                PARTITION BY order_year
                ORDER BY order_month
            ), 2)                               AS cumulative_revenue_ytd
        FROM orders
        WHERE status != 'cancelled'
        GROUP BY order_year, order_month
        ORDER BY order_year, order_month
        """
    ),

    "top_products": (
        "Top 10 products by revenue",
        """
        SELECT
            p.product_id,
            p.name,
            p.category,
            p.subcategory,
            ROUND(p.price, 2)                   AS price,
            SUM(oi.quantity)                    AS units_sold,
            ROUND(SUM(oi.line_total), 2)        AS total_revenue,
            ROUND(AVG(oi.discount_pct), 1)      AS avg_discount_pct
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        JOIN orders   o ON oi.order_id   = o.order_id
        WHERE o.status = 'completed'
        GROUP BY p.product_id, p.name, p.category, p.subcategory, p.price
        ORDER BY total_revenue DESC
        LIMIT 10
        """
    ),

    "customer_segments": (
        "Revenue and order count by customer segment + country",
        """
        SELECT
            c.segment,
            c.country,
            COUNT(DISTINCT c.customer_id)       AS customers,
            COUNT(DISTINCT o.order_id)          AS orders,
            ROUND(SUM(o.total_amount), 2)       AS revenue,
            ROUND(AVG(o.total_amount), 2)       AS avg_order_value,
            ROUND(SUM(o.total_amount)
                  / COUNT(DISTINCT c.customer_id), 2) AS revenue_per_customer
        FROM customers c
        JOIN orders    o ON c.customer_id = o.customer_id
        WHERE o.status NOT IN ('cancelled', 'refunded')
        GROUP BY c.segment, c.country
        ORDER BY revenue DESC
        LIMIT 15
        """
    ),

    "cohort": (
        "Customer cohort — first purchase year vs lifetime revenue",
        """
        WITH first_order AS (
            SELECT customer_id, MIN(order_date) AS first_date
            FROM orders
            GROUP BY customer_id
        ),
        cohorts AS (
            SELECT
                c.customer_id,
                YEAR(CAST(fo.first_date AS DATE)) AS cohort_year,
                SUM(o.total_amount)               AS ltv
            FROM customers   c
            JOIN first_order fo ON c.customer_id = fo.customer_id
            JOIN orders       o ON c.customer_id = o.customer_id
            WHERE o.status NOT IN ('cancelled', 'refunded')
            GROUP BY c.customer_id, cohort_year
        )
        SELECT
            cohort_year,
            COUNT(DISTINCT customer_id)       AS customers,
            ROUND(SUM(ltv), 2)                AS total_revenue,
            ROUND(AVG(ltv), 2)                AS avg_ltv,
            ROUND(MEDIAN(ltv), 2)             AS median_ltv
        FROM cohorts
        GROUP BY cohort_year
        ORDER BY cohort_year
        """
    ),

    "payment_methods": (
        "Payment method breakdown and avg order value",
        """
        SELECT
            payment_method,
            COUNT(*)                            AS orders,
            ROUND(SUM(total_amount), 2)         AS revenue,
            ROUND(AVG(total_amount), 2)         AS avg_order_value,
            ROUND(COUNT(*) * 100.0
                  / SUM(COUNT(*)) OVER (), 1)   AS order_share_pct
        FROM orders
        WHERE status = 'completed'
        GROUP BY payment_method
        ORDER BY revenue DESC
        """
    ),

    "supplier_margin": (
        "Supplier profitability — revenue vs cost vs margin",
        """
        SELECT
            p.supplier,
            COUNT(DISTINCT p.product_id)        AS products,
            SUM(oi.quantity)                    AS units_sold,
            ROUND(SUM(oi.line_total), 2)        AS gross_revenue,
            ROUND(SUM(oi.quantity * p.cost), 2) AS total_cost,
            ROUND(SUM(oi.line_total)
                  - SUM(oi.quantity * p.cost), 2) AS gross_profit,
            ROUND((SUM(oi.line_total)
                   - SUM(oi.quantity * p.cost))
                  / NULLIF(SUM(oi.line_total), 0) * 100, 1) AS margin_pct
        FROM order_items oi
        JOIN products p ON oi.product_id = p.product_id
        JOIN orders   o ON oi.order_id   = o.order_id
        WHERE o.status = 'completed'
        GROUP BY p.supplier
        ORDER BY gross_profit DESC
        """
    ),

    "cancellation_rate": (
        "Cancellation and refund rate by category",
        """
        SELECT
            p.category,
            COUNT(DISTINCT o.order_id)          AS total_orders,
            COUNT(DISTINCT CASE WHEN o.status = 'cancelled' THEN o.order_id END) AS cancelled,
            COUNT(DISTINCT CASE WHEN o.status = 'refunded'  THEN o.order_id END) AS refunded,
            ROUND(
                COUNT(DISTINCT CASE WHEN o.status IN ('cancelled','refunded')
                      THEN o.order_id END) * 100.0
                / COUNT(DISTINCT o.order_id), 1
            ) AS loss_rate_pct
        FROM orders o
        JOIN order_items oi ON o.order_id   = oi.order_id
        JOIN products    p  ON oi.product_id = p.product_id
        GROUP BY p.category
        ORDER BY loss_rate_pct DESC
        """
    ),

}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query", default="all",
        choices=list(QUERIES.keys()) + ["all"],
        help="Which query to run (default: all)"
    )
    args = parser.parse_args()

    if not (PARQUET_DIR / "orders.parquet").exists():
        print("ERROR: Parquet files not found. Run generate.py first.")
        return

    print("\n══ DuckDB Analytics Demo ════════════════════════════════════════")
    print(f"   Reading directly from Parquet (no database server needed)")
    print(f"   Files: {PARQUET_DIR.resolve()}\n")

    con = _db()

    if args.query == "all":
        for key, (label, sql) in QUERIES.items():
            run(label, sql, con)
    else:
        label, sql = QUERIES[args.query]
        run(label, sql, con)

    print(f"\n{'─'*60}")
    print("  Same queries run on DuckDB via the gateway:")
    print("  POST http://localhost:8088/query/sql")
    print("  POST http://localhost:8088/query/nl  (natural language)")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    main()
