import numpy as np
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_raw_tables(base_path: str):
    """Load the 3 raw tables from data/raw/."""
    customers_df = pd.read_csv(f"{base_path}/customer.csv")
    products_df = pd.read_csv(f"{base_path}/product.csv")
    sales_df = pd.read_csv(f"{base_path}/sales.csv", decimal=".", thousands=",")
    return customers_df, products_df, sales_df


def _first_word(s):
    try:
        return s.split(" ")[0]
    except AttributeError:
        return np.nan


def clean_tables(customers_df, products_df, sales_df):
    """Parse dates, clean blank strings, and normalize LEVEL_1."""
    customers_df = customers_df.copy()
    products_df = products_df.copy()
    sales_df = sales_df.copy()

    customers_df["BIRTH_DATE"] = pd.to_datetime(customers_df["BIRTH_DATE"], errors="coerce")
    customers_df["FIRST_PURCHASE_DATE"] = pd.to_datetime(customers_df["FIRST_PURCHASE_DATE"], errors="coerce")
    sales_df["SALE_DATE"] = pd.to_datetime(sales_df["SALE_DATE"], format="%Y/%m/%d", errors="coerce")

    for df in (customers_df, products_df, sales_df):
        df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

    products_df["LEVEL_1"] = products_df["LEVEL_1"].apply(_first_word)
    return customers_df, products_df, sales_df


def merge_sales_products(sales_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    return sales_df.merge(products_df, on="PRODUCT_ID", how="left")