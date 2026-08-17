from pathlib import Path
from typing import Set, Union

import pandas as pd


CUSTOMER_COLUMNS = {
    "CUSTOMER_ID",
    "SEX",
    "BIRTH_DATE",
    "FIRST_PURCHASE_DATE",
}
PRODUCT_COLUMNS = {
    "PRODUCT_ID",
    "UNIT",
    "LEVEL_1",
    "LEVEL_2",
    "LEVEL_3",
    "LEVEL_4",
}
SALE_COLUMNS = {
    "SALE_ID",
    "SALE_DATE",
    "CUSTOMER_ID",
    "PRODUCT_ID",
    "QUANTITY",
    "SALE_CATEGORY",
    "BRANCH",
}


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(path)


def _require_columns(
    df: pd.DataFrame,
    required: Set[str],
    table_name: str,
) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {sorted(missing)}"
        )


def load_raw_tables(base_path: Union[str, Path]):
    """Load the three project tables from a directory."""
    base_path = Path(base_path)

    customers_df = pd.read_csv(
        base_path / "customers.csv",
        dtype={"SEX": "string"},
    )
    products_df = pd.read_csv(base_path / "products.csv")
    sales_df = pd.read_csv(
        base_path / "sales.csv",
        dtype={"QUANTITY": "Int64"},
        low_memory=False,
    )

    _require_columns(customers_df, CUSTOMER_COLUMNS, "customers.csv")
    _require_columns(products_df, PRODUCT_COLUMNS, "products.csv")
    _require_columns(sales_df, SALE_COLUMNS, "sales.csv")

    return customers_df, products_df, sales_df


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    string_columns = df.select_dtypes(include=["object", "string"]).columns
    for column in string_columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )
    return df


def clean_tables(
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    sales_df: pd.DataFrame,
):
    """Normalize strings, identifiers, dates, and quantities."""
    customers_df = _strip_string_columns(customers_df.copy())
    products_df = _strip_string_columns(products_df.copy())
    sales_df = _strip_string_columns(sales_df.copy())

    customers_df["CUSTOMER_ID"] = pd.to_numeric(
        customers_df["CUSTOMER_ID"], errors="raise"
    ).astype("Int64")
    products_df["PRODUCT_ID"] = pd.to_numeric(
        products_df["PRODUCT_ID"], errors="raise"
    ).astype("Int64")
    sales_df["SALE_ID"] = pd.to_numeric(
        sales_df["SALE_ID"], errors="raise"
    ).astype("Int64")
    sales_df["CUSTOMER_ID"] = pd.to_numeric(
        sales_df["CUSTOMER_ID"], errors="raise"
    ).astype("Int64")
    sales_df["PRODUCT_ID"] = pd.to_numeric(
        sales_df["PRODUCT_ID"], errors="raise"
    ).astype("Int64")

    customers_df["SEX"] = customers_df["SEX"].str.upper()
    customers_df["BIRTH_DATE"] = pd.to_datetime(
        customers_df["BIRTH_DATE"],
        format="%Y/%m/%d",
        errors="coerce",
    )
    customers_df["FIRST_PURCHASE_DATE"] = pd.to_datetime(
        customers_df["FIRST_PURCHASE_DATE"],
        format="%Y/%m/%d",
        errors="coerce",
    )
    sales_df["SALE_DATE"] = pd.to_datetime(
        sales_df["SALE_DATE"],
        format="%Y/%m/%d",
        errors="coerce",
    )
    sales_df["QUANTITY"] = pd.to_numeric(
        sales_df["QUANTITY"],
        errors="coerce",
    ).astype("Int64")

    return customers_df, products_df, sales_df


def merge_sales_products(
    sales_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> pd.DataFrame:
    return sales_df.merge(
        products_df,
        on="PRODUCT_ID",
        how="left",
        validate="many_to_one",
    )

def _first_word(s):
    try:
        return s.split(" ")[0]
    except AttributeError:
        return np.nan
