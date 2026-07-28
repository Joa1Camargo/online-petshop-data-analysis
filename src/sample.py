#!/usr/bin/env python3
"""Generate a deterministic, fully synthetic pet-retail dataset.

The generated records do not derive from or reproduce source rows. Aggregate
weights approximate the original case only to keep the demo analytically useful.
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path


CUSTOMER_FIELDS = ["CUSTOMER_ID", "SEX", "BIRTH_DATE", "FIRST_PURCHASE_DATE"]
PRODUCT_FIELDS = ["PRODUCT_ID", "UNIT", "LEVEL_1", "LEVEL_2", "LEVEL_3", "LEVEL_4"]
SALE_FIELDS = [
    "SALE_ID",
    "SALE_DATE",
    "CUSTOMER_ID",
    "PRODUCT_ID",
    "QUANTITY",
    "SALE_CATEGORY",
    "BRANCH",
]

SEX_VALUES = ["", "F", "M", "O"]
SEX_WEIGHTS = [59_401, 26_809 + 575, 17_384 + 448, 1_307]

SALE_CATEGORIES = ["STORE", "GROOMING", "CLINIC", "E-COMMERCE", "DIGITAL"]
SALE_CATEGORY_WEIGHTS = [1_698_909, 91_187, 37_179, 18_101, 14_207]

BRANCHES = [101, 102, 104, 103, 106, 107, 105, 110, 108, 109, 113]
BRANCH_WEIGHTS = [
    462_792,
    379_464,
    280_065,
    233_542,
    151_519,
    141_019,
    139_958,
    30_835,
    21_422,
    18_359,
    608,
]

PRODUCT_TAXONOMY = [
    ("DOGS", "FOOD", "PET FOOD", "DRY PET FOOD"),
    ("DOGS", "FOOD", "PET FOOD", "WET PET FOOD (POUCHES & CANS)"),
    ("DOGS", "FOOD", "PET FOOD", "PRESCRIPTION DIET FOOD"),
    ("DOGS", "FOOD", "TREATS & CHEWS", "BISCUITS - MEAT STRIPS - SNACKS"),
    ("DOGS", "FOOD", "TREATS & CHEWS", "BONES & CHEW STICKS"),
    ("CATS", "FOOD", "PET FOOD", "DRY PET FOOD"),
    ("CATS", "FOOD", "PET FOOD", "WET PET FOOD (POUCHES & CANS)"),
    ("CATS", "FOOD", "PET FOOD", "PRESCRIPTION DIET FOOD"),
    ("BIRDS", "FOOD", "PET FOOD", ""),
    ("FISH", "FOOD", "PET FOOD", ""),
    ("RODENTS", "FOOD", "PET FOOD - HAY - ALFALFA", ""),
    ("DOGS", "PHARMACY", "FLEAS - TICKS - WORMS", "FLEA & TICK TREATMENTS"),
    ("DOGS", "PHARMACY", "FLEAS - TICKS - WORMS", "DEWORMERS"),
    ("CATS", "PHARMACY", "FLEAS - TICKS - WORMS", "FLEA & TICK TREATMENTS"),
    ("DOGS", "PHARMACY", "PAIN & INFECTIONS", "ANTI-INFLAMMATORIES"),
    ("DOGS", "PHARMACY", "PAIN & INFECTIONS", "ANTIBIOTICS"),
    ("DOGS", "PHARMACY", "SKIN & COAT", "SKIN TREATMENT"),
    ("DOGS", "PHARMACY", "EYES & EARS", "EAR TREATMENT"),
    ("DOGS", "PHARMACY", "VITAMINS & SUPPLEMENTS", "VITAMINS & SUPPLEMENTS"),
    ("DOGS", "ACCESSORIES", "TOYS", "TUG & CHEW TOYS"),
    ("DOGS", "ACCESSORIES", "TOYS", "FETCH TOYS"),
    ("DOGS", "ACCESSORIES", "FEEDING ACCESSORIES", "BOWLS & WATER DISPENSERS"),
    ("CATS", "ACCESSORIES", "TOYS", "WANDS - MICE - BALLS"),
    ("CATS", "ACCESSORIES", "TOYS", "SCRATCHERS"),
    ("DOGS", "GROOMING & HYGIENE", "BATHING SUPPLIES", "SHAMPOOS & COSMETICS"),
    ("DOGS", "GROOMING & HYGIENE", "HYGIENE & CLEANING", "TRAINING PADS"),
    ("CATS", "GROOMING & HYGIENE", "HYGIENE & CLEANING", "CAT LITTER & GRANULES"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--customers", type=int, default=5_000)
    parser.add_argument("--products", type=int, default=300)
    parser.add_argument("--orders", type=int, default=60_000)
    return parser.parse_args()


def random_date(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randrange((end - start).days + 1))


def weighted_sales_dates(start: date, end: date) -> tuple[list[date], list[float]]:
    days: list[date] = []
    weights: list[float] = []
    current = start
    while current <= end:
        weight = 1.0
        if current.weekday() >= 5:
            weight *= 1.15
        if current.month == 12:
            weight *= 1.35
        elif current.month in (6, 7):
            weight *= 1.10
        days.append(current)
        weights.append(weight)
        current += timedelta(days=1)
    return days, weights


def generate_customers(
    rng: random.Random, count: int
) -> tuple[list[dict[str, object]], list[float]]:
    rows: list[dict[str, object]] = []
    activity_weights: list[float] = []
    for customer_id in range(1, count + 1):
        age_at_start = rng.randint(18, 80)
        birth_start = date(2020 - age_at_start - 1, 1, 1)
        birth_end = date(2020 - age_at_start, 12, 31)
        rows.append(
            {
                "CUSTOMER_ID": customer_id,
                "SEX": rng.choices(SEX_VALUES, weights=SEX_WEIGHTS, k=1)[0],
                "BIRTH_DATE": random_date(rng, birth_start, birth_end).strftime("%Y/%m/%d"),
                "FIRST_PURCHASE_DATE": random_date(
                    rng, date(2017, 1, 1), date(2019, 12, 31)
                ).strftime("%Y/%m/%d"),
            }
        )
        activity_weights.append(min(rng.paretovariate(1.8), 25.0))
    return rows, activity_weights


def generate_products(
    rng: random.Random, count: int
) -> tuple[list[dict[str, object]], list[float]]:
    rows: list[dict[str, object]] = []
    popularity_weights: list[float] = []
    units = ["UN", "PC", "CX", "KG", "DS"]
    unit_weights = [88, 5, 3, 3, 1]
    for product_id in range(1, count + 1):
        if rng.random() < 0.10:
            hierarchy = ("", "", "", "")
        else:
            hierarchy = rng.choice(PRODUCT_TAXONOMY)
        rows.append(
            {
                "PRODUCT_ID": product_id,
                "UNIT": rng.choices(units, weights=unit_weights, k=1)[0],
                "LEVEL_1": hierarchy[0],
                "LEVEL_2": hierarchy[1],
                "LEVEL_3": hierarchy[2],
                "LEVEL_4": hierarchy[3],
            }
        )
        popularity_weights.append(1.0 / (product_id**0.75))
    return rows, popularity_weights


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_sales(
    rng: random.Random,
    path: Path,
    order_count: int,
    customer_count: int,
    product_count: int,
    customer_weights: list[float],
    product_weights: list[float],
) -> int:
    sale_days, date_weights = weighted_sales_dates(
        date(2020, 1, 2), date(2021, 10, 26)
    )
    order_dates = rng.choices(sale_days, weights=date_weights, k=order_count)
    order_dates.sort()

    line_count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SALE_FIELDS)
        writer.writeheader()
        for sale_id, sale_date in enumerate(order_dates, start=1):
            customer_id = rng.choices(
                range(1, customer_count + 1), weights=customer_weights, k=1
            )[0]
            sale_category = rng.choices(
                SALE_CATEGORIES, weights=SALE_CATEGORY_WEIGHTS, k=1
            )[0]
            branch = rng.choices(BRANCHES, weights=BRANCH_WEIGHTS, k=1)[0]
            item_count = rng.choices([1, 2, 3, 4], weights=[25, 45, 20, 10], k=1)[0]
            product_ids = rng.choices(
                range(1, product_count + 1), weights=product_weights, k=item_count
            )
            for product_id in product_ids:
                writer.writerow(
                    {
                        "SALE_ID": sale_id,
                        "SALE_DATE": sale_date.strftime("%Y/%m/%d"),
                        "CUSTOMER_ID": customer_id,
                        "PRODUCT_ID": product_id,
                        "QUANTITY": rng.choices([1, 2, 3, 4], weights=[84, 11, 4, 1], k=1)[0],
                        "SALE_CATEGORY": sale_category,
                        "BRANCH": branch,
                    }
                )
                line_count += 1
    return line_count


def validate(output_dir: Path) -> None:
    with (output_dir / "customers.csv").open(newline="", encoding="utf-8") as handle:
        customers = list(csv.DictReader(handle))
    with (output_dir / "products.csv").open(newline="", encoding="utf-8") as handle:
        products = list(csv.DictReader(handle))

    customer_ids = {row["CUSTOMER_ID"] for row in customers}
    product_ids = {row["PRODUCT_ID"] for row in products}
    seen_sales = 0

    with (output_dir / "sales.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SALE_FIELDS:
            raise ValueError(f"Unexpected sales schema: {reader.fieldnames}")
        for row in reader:
            if row["CUSTOMER_ID"] not in customer_ids:
                raise ValueError("Broken CUSTOMER_ID foreign key")
            if row["PRODUCT_ID"] not in product_ids:
                raise ValueError("Broken PRODUCT_ID foreign key")
            if not row["QUANTITY"].isdigit() or int(row["QUANTITY"]) < 1:
                raise ValueError("QUANTITY must be a positive integer")
            seen_sales += 1

    if not customers or not products or not seen_sales:
        raise ValueError("Synthetic dataset cannot be empty")


def main() -> None:
    args = parse_args()
    if min(args.customers, args.products, args.orders) < 1:
        raise ValueError("customers, products, and orders must all be positive")

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    customers, customer_weights = generate_customers(rng, args.customers)
    products, product_weights = generate_products(rng, args.products)
    write_csv(output_dir / "customers.csv", CUSTOMER_FIELDS, customers)
    write_csv(output_dir / "products.csv", PRODUCT_FIELDS, products)
    sale_lines = generate_sales(
        rng,
        output_dir / "sales.csv",
        args.orders,
        args.customers,
        args.products,
        customer_weights,
        product_weights,
    )
    validate(output_dir)
    print(
        f"Generated {len(customers):,} customers, {len(products):,} products, "
        f"{args.orders:,} orders, and {sale_lines:,} sale lines in {output_dir}"
    )


if __name__ == "__main__":
    main()
