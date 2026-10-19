"""
E-commerce data generator — inspired by:
https://www.kdnuggets.com/building-your-modern-data-analytics-stack-with-python-parquet-and-duckdb

Generates a realistic star schema:
  customers     → 1 000 rows
  products      → 200 rows
  orders        → 8 000 rows   (fact table)
  order_items   → ~24 000 rows (fact table, ~3 items per order)

Output
------
  Parquet files  → data_generator/parquet/
  MinIO upload   → s3://lakehouse/raw/<table>/   (if --upload flag)

The same data powers all three query modes in the gateway:
  SQL analytics  → orders + customers + products (DuckDB / Athena)
  Vector search  → product descriptions          (LanceDB)
  NL → SQL       → all tables                    (Ollama)
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from faker import Faker

OUTPUT_DIR = Path(__file__).parent / "parquet"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)


# ── Lookup tables ─────────────────────────────────────────────────────────────

CATEGORIES = {
    "Electronics": {
        "subcategories": ["Headphones", "Laptops", "Smartphones", "Cameras", "Wearables"],
        "price_range": (49, 1999),
        "margin": (0.15, 0.35),
    },
    "Furniture": {
        "subcategories": ["Chairs", "Desks", "Shelving", "Lighting", "Beds"],
        "price_range": (89, 1499),
        "margin": (0.20, 0.45),
    },
    "Sports": {
        "subcategories": ["Running", "Yoga", "Cycling", "Swimming", "Gym Equipment"],
        "price_range": (15, 499),
        "margin": (0.25, 0.55),
    },
    "Kitchen": {
        "subcategories": ["Cookware", "Appliances", "Utensils", "Coffee", "Storage"],
        "price_range": (20, 599),
        "margin": (0.22, 0.48),
    },
    "Books": {
        "subcategories": ["Programming", "Data Science", "Business", "Self Help", "Design"],
        "price_range": (12, 89),
        "margin": (0.30, 0.60),
    },
}

SUPPLIERS = [
    "TechSupply Co.", "GlobalGoods Ltd.", "FastShip Inc.", "QualityFirst GmbH",
    "AsiaSource Pte.", "EuroTrade BV", "NordVendor AS", "PrimeGoods LLC",
]

COUNTRIES = {
    "United Arab Emirates": ["Dubai", "Abu Dhabi", "Sharjah", "Al Ain"],
    "Saudi Arabia": ["Riyadh", "Jeddah", "Dammam", "Mecca"],
    "United Kingdom": ["London", "Manchester", "Birmingham", "Leeds"],
    "United States": ["New York", "Los Angeles", "Chicago", "Houston"],
    "Germany": ["Berlin", "Munich", "Hamburg", "Frankfurt"],
    "India": ["Mumbai", "Delhi", "Bangalore", "Hyderabad"],
    "Australia": ["Sydney", "Melbourne", "Brisbane", "Perth"],
    "Canada": ["Toronto", "Vancouver", "Montreal", "Calgary"],
}

SEGMENTS = ["B2C", "B2B", "Premium", "Enterprise"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "Bank Transfer", "Buy Now Pay Later"]
ORDER_STATUSES = ["completed", "completed", "completed", "shipped", "processing", "cancelled", "refunded"]
PRODUCT_ADJECTIVES = [
    "Pro", "Ultra", "Smart", "Compact", "Portable", "Premium", "Essential",
    "Advanced", "Elite", "Lite", "Plus", "Max", "Mini", "Deluxe",
]


# ── Generators ────────────────────────────────────────────────────────────────

def generate_customers(n: int = 1_000) -> pd.DataFrame:
    print(f"  Generating {n:,} customers...")
    records = []
    for i in range(1, n + 1):
        country = random.choice(list(COUNTRIES.keys()))
        city = random.choice(COUNTRIES[country])
        signup = fake.date_between(start_date=date(2020, 1, 1), end_date=date(2024, 6, 30))
        records.append({
            "customer_id": i,
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "email": fake.unique.email(),
            "country": country,
            "city": city,
            "segment": random.choice(SEGMENTS),
            "signup_date": signup.isoformat(),
            "age_group": random.choice(["18-25", "26-35", "36-45", "46-55", "55+"]),
            "lifetime_value": round(random.uniform(50, 15_000), 2),
        })
    return pd.DataFrame(records)


def _product_description(name: str, category: str, subcategory: str, price: float) -> str:
    templates = [
        f"High-quality {subcategory.lower()} product offering exceptional performance for everyday use. {name} is designed for customers who demand reliability and value at {price:.0f}.",
        f"Premium {category.lower()} solution from our {subcategory} range. {name} combines cutting-edge design with practical functionality, perfect for professionals and enthusiasts.",
        f"Introducing {name} — our best-selling {subcategory.lower()} item. Engineered for durability and ease of use with a competitive price point of {price:.0f}.",
        f"{name} redefines what you expect from a {subcategory.lower()} product. Thoughtfully designed for the modern {category.lower()} market with top-tier build quality.",
        f"Discover {name}, a leading choice in our {category} collection. This {subcategory.lower()} product delivers superior results backed by our quality guarantee.",
    ]
    return random.choice(templates)


def generate_products(n: int = 200) -> pd.DataFrame:
    print(f"  Generating {n:,} products...")
    records = []
    per_category = n // len(CATEGORIES)
    pid = 1
    for category, meta in CATEGORIES.items():
        for _ in range(per_category):
            subcategory = random.choice(meta["subcategories"])
            adj = random.choice(PRODUCT_ADJECTIVES)
            name = f"{adj} {subcategory} {random.randint(100, 999)}"
            price = round(random.uniform(*meta["price_range"]), 2)
            margin = random.uniform(*meta["margin"])
            cost = round(price * (1 - margin), 2)
            records.append({
                "product_id": pid,
                "name": name,
                "category": category,
                "subcategory": subcategory,
                "price": price,
                "cost": cost,
                "margin_pct": round(margin * 100, 1),
                "supplier": random.choice(SUPPLIERS),
                "stock_quantity": random.randint(0, 500),
                "is_active": random.random() > 0.1,
                "description": _product_description(name, category, subcategory, price),
                "launch_date": fake.date_between(
                    start_date=date(2019, 1, 1), end_date=date(2024, 1, 1)
                ).isoformat(),
            })
            pid += 1
    # Fill remainder with random category
    while len(records) < n:
        category = random.choice(list(CATEGORIES.keys()))
        meta = CATEGORIES[category]
        subcategory = random.choice(meta["subcategories"])
        adj = random.choice(PRODUCT_ADJECTIVES)
        name = f"{adj} {subcategory} {random.randint(100, 999)}"
        price = round(random.uniform(*meta["price_range"]), 2)
        margin = random.uniform(*meta["margin"])
        records.append({
            "product_id": pid,
            "name": name,
            "category": category,
            "subcategory": subcategory,
            "price": price,
            "cost": round(price * (1 - margin), 2),
            "margin_pct": round(margin * 100, 1),
            "supplier": random.choice(SUPPLIERS),
            "stock_quantity": random.randint(0, 500),
            "is_active": random.random() > 0.1,
            "description": _product_description(name, category, subcategory, price),
            "launch_date": fake.date_between(
                start_date=date(2019, 1, 1), end_date=date(2024, 1, 1)
            ).isoformat(),
        })
        pid += 1
    return pd.DataFrame(records)


def generate_orders(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    n_orders: int = 8_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"  Generating {n_orders:,} orders + order items...")

    customer_ids = customers["customer_id"].tolist()
    product_ids = products["product_id"].tolist()
    prices = dict(zip(products["product_id"], products["price"]))

    orders = []
    order_items = []
    item_id = 1

    for oid in range(1, n_orders + 1):
        cid = random.choice(customer_ids)
        order_date = fake.date_between(start_date=date(2022, 1, 1), end_date=date(2024, 9, 30))
        status = random.choice(ORDER_STATUSES)
        payment = random.choice(PAYMENT_METHODS)

        # 1–6 items per order (Poisson-ish)
        n_items = min(6, max(1, int(np.random.poisson(2.5))))
        selected = random.sample(product_ids, min(n_items, len(product_ids)))

        total = 0.0
        for pid in selected:
            qty = random.randint(1, 4)
            unit_price = round(prices[pid] * random.uniform(0.85, 1.05), 2)  # slight price variation
            discount = round(random.choice([0, 0, 0, 5, 10, 15, 20]) / 100, 2)
            line_total = round(unit_price * qty * (1 - discount), 2)
            total += line_total
            order_items.append({
                "item_id": item_id,
                "order_id": oid,
                "product_id": pid,
                "quantity": qty,
                "unit_price": unit_price,
                "discount_pct": discount * 100,
                "line_total": line_total,
            })
            item_id += 1

        orders.append({
            "order_id": oid,
            "customer_id": cid,
            "order_date": order_date.isoformat(),
            "order_year": order_date.year,
            "order_month": order_date.month,
            "order_quarter": (order_date.month - 1) // 3 + 1,
            "status": status,
            "payment_method": payment,
            "total_amount": round(total, 2),
            "item_count": n_items,
            "shipping_days": random.randint(1, 14) if status != "cancelled" else None,
        })

    return pd.DataFrame(orders), pd.DataFrame(order_items)


# ── Parquet writer ────────────────────────────────────────────────────────────

def write_parquet(df: pd.DataFrame, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.parquet"
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression="snappy",
        row_group_size=50_000,
    )
    size_kb = path.stat().st_size / 1024
    print(f"  ✓ Wrote {path.name}  ({len(df):,} rows, {size_kb:.1f} KB)")
    return path


# ── MinIO upload ──────────────────────────────────────────────────────────────

def upload_to_minio(
    path: Path,
    table_name: str,
    endpoint: str = "http://localhost:9000",
    access_key: str = "minioadmin",
    secret_key: str = "minioadmin",
):
    import boto3
    from botocore.client import Config

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    bucket = "lakehouse"
    key = f"raw/{table_name}/{path.name}"
    s3.upload_file(str(path), bucket, key)
    print(f"  ✓ Uploaded s3://{bucket}/{key}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate e-commerce sample data")
    parser.add_argument("--customers", type=int, default=1_000)
    parser.add_argument("--products",  type=int, default=200)
    parser.add_argument("--orders",    type=int, default=8_000)
    parser.add_argument("--upload",    action="store_true", help="Upload Parquet files to MinIO")
    parser.add_argument("--minio-endpoint", default="http://localhost:9000")
    parser.add_argument("--minio-access-key", default="minioadmin")
    parser.add_argument("--minio-secret-key", default="minioadmin")
    args = parser.parse_args()

    print("\n── Generating e-commerce dataset ──────────────────────────────")
    customers  = generate_customers(args.customers)
    products   = generate_products(args.products)
    orders, order_items = generate_orders(customers, products, args.orders)

    print("\n── Writing Parquet files ───────────────────────────────────────")
    paths = {
        "customers":   write_parquet(customers,   "customers"),
        "products":    write_parquet(products,    "products"),
        "orders":      write_parquet(orders,      "orders"),
        "order_items": write_parquet(order_items, "order_items"),
    }

    print(f"\n── Summary ─────────────────────────────────────────────────────")
    print(f"  customers:   {len(customers):>6,} rows")
    print(f"  products:    {len(products):>6,} rows")
    print(f"  orders:      {len(orders):>6,} rows")
    print(f"  order_items: {len(order_items):>6,} rows")
    print(f"  Output dir:  {OUTPUT_DIR.resolve()}")

    if args.upload:
        print("\n── Uploading to MinIO ──────────────────────────────────────────")
        for name, path in paths.items():
            upload_to_minio(
                path, name,
                endpoint=args.minio_endpoint,
                access_key=args.minio_access_key,
                secret_key=args.minio_secret_key,
            )

    print("\n── Next steps ──────────────────────────────────────────────────")
    print("  Load into Iceberg:  python load_to_iceberg.py")
    print("  Run DuckDB queries: python analytics.py")
    print("  Gateway API docs:   http://localhost:8088/docs\n")


if __name__ == "__main__":
    main()
