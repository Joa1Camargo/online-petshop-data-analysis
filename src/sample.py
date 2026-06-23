import pandas as pd
import os

# 1. Path of the raw data files
raw_sales = "data/raw/sales.csv"
raw_customers = "data/raw/customer.csv"
raw_products = "data/raw/product.csv"

# 2. Paths where we will save the samples (LIGHT and COHERENT)
samples_sample = "data/sample/sales.csv"
customers_sample = "data/sample/customer.csv"
products_sample = "data/sample/product.csv"

def create_samples():
    print("⏳ Step 1: Creating sample of sales data...")
    df_sales_sample = pd.read_csv(raw_sales, nrows=5000)
    
    # Aqui nós descobrimos quais IDs únicos de clientes e produtos existem nessa amostra de vendas
    customers_on_sample = df_sales_sample["CUSTOMER_ID"].unique()
    products_on_sample = df_sales_sample["PRODUCT_ID"].unique()
    
    print(f"   -> Found {len(customers_on_sample)} customers and {len(products_on_sample)} products uniquely in the sales sample.")

    print("\n⏳ Step 2: Filtering the Customers table coherently...")
    # Reading the giant customers table
    df_customers_complete = pd.read_csv(raw_customers)
    # We filter to keep ONLY the customers that are in the 'customers_on_sample' list
    df_customers_sample = df_customers_complete[df_customers_complete["CUSTOMER_ID"].isin(customers_on_sample)]

    print("\n⏳ Step 3: Filtering the Products table coherently...")
    # Reading the giant products table
    df_products_complete = pd.read_csv(raw_products)
    # We filter to keep ONLY the products that are in the 'products_on_sample' list
    df_products_sample = df_products_complete[df_products_complete["PRODUCT_ID"].isin(products_on_sample)]

    # Garantir que a pasta de destino existe
    os.makedirs("data/sample", exist_ok=True)

    # Salvando tudo na pasta sample
    df_sales_sample.to_csv(samples_sample, index=False)
    df_customers_sample.to_csv(customers_sample, index=False)
    df_products_sample.to_csv(products_sample, index=False)

    print("\n✅ Success! All coherent samples have been saved in data/sample/")
    print(f"   - Sales: {len(df_sales_sample)} rows")
    print(f"   - Customers: {len(df_customers_sample)} rows")
    print(f"   - Products: {len(df_products_sample)} rows")

if __name__ == "__main__":
    create_samples()