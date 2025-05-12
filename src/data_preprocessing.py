"""
Data Preprocessing Module

This module handles the cleaning and transformation of raw customer transaction data.
It includes handling missing values, removing invalid records, correcting data types,
and preparing the dataset for feature engineering and modeling.
"""

from typing import Dict
import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the customer transaction dataset.

    Steps:
      1. Drop or impute missing CustomerID
      2. Remove duplicates
      3. Convert data types
      4. Remove returns and invalid transactions
      5. Remove top 1% outliers
      6. Derive new features (TotalPrice, date parts)
      7. Standardize text fields

    Args:
        df (pd.DataFrame): Raw customer transaction data

    Returns:
        pd.DataFrame: Cleaned and preprocessed data
    """
    print("\n=== DATA PREPROCESSING ===")

    # Record original dimensions
    original_rows, original_cols = df.shape
    print(f"Starting with {original_rows} rows and {original_cols} columns")

    # Copy dataframe to avoid modifying original
    df = df.copy()

    # -----------------------------
    # 1. Handle Missing Customer IDs
    # -----------------------------
    missing_cust = df['CustomerID'].isna().sum()
    missing_pct = missing_cust / len(df) * 100
    print(f"Missing CustomerID: {missing_cust} rows ({missing_pct:.2f}%)")

    if missing_pct < 25:
        df = df.dropna(subset=['CustomerID'])
        print("Dropped rows with missing CustomerID")
    else:
        df['CustomerID'] = df['CustomerID'].fillna(-1)
        print("Filled missing CustomerID with placeholder -1")

    df['CustomerID'] = df['CustomerID'].astype(int)

    # -----------------------------
    # 2. Remove Exact Duplicates
    # -----------------------------
    dup_count = df.duplicated().sum()
    print(f"Found {dup_count} duplicate rows")

    if dup_count > 0:
        df = df.drop_duplicates()
        print("Removed duplicate rows")

    # -----------------------------
    # 3. Convert Data Types
    # -----------------------------
    if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        print("Converted InvoiceDate to datetime")

    df['Quantity'] = df['Quantity'].astype(int)
    df['UnitPrice'] = df['UnitPrice'].astype(float)

    for col in ['InvoiceNo', 'StockCode', 'Country']:
        df[col] = df[col].astype('category')
    print("Converted types for numeric and categorical features")

    # -----------------------------
    # 4. Filter Invalid Transactions
    # -----------------------------
    returns_count = (df['Quantity'] < 0).sum()
    print(f"Return transactions (negative quantity): {returns_count}")
    df = df[df['Quantity'] > 0]

    invalid_price_count = (df['UnitPrice'] <= 0).sum()
    print(f"Invalid prices (<= 0): {invalid_price_count}")
    df = df[df['UnitPrice'] > 0]

    # -----------------------------
    # 5. Remove Top 1% Outliers
    # -----------------------------
    qty_high = df['Quantity'].quantile(0.99)
    price_high = df['UnitPrice'].quantile(0.99)
    out_qty = (df['Quantity'] > qty_high).sum()
    out_price = (df['UnitPrice'] > price_high).sum()

    print(f"Quantity outliers (>99th percentile): {out_qty}")
    print(f"Price outliers (>99th percentile): {out_price}")
    df = df[(df['Quantity'] <= qty_high) & (df['UnitPrice'] <= price_high)]

    print("Removed top 1% outliers for Quantity and UnitPrice")

    # -----------------------------
    # 6. Derive Basic Features
    # -----------------------------
    if 'TotalPrice' not in df.columns:
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        print("Created 'TotalPrice' column")

    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.day
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    df['Hour'] = df['InvoiceDate'].dt.hour

    print("Extracted date/time parts: Year, Month, Day, DayOfWeek, Hour")

    # -----------------------------
    # 7. Standardize Text Fields
    # -----------------------------
    if 'Description' in df.columns:
        df['Description'] = df['Description'].str.strip()
        print("Trimmed whitespace from Description field")

    # -----------------------------
    # Summary of Changes
    # -----------------------------
    final_rows, final_cols = df.shape
    print("\nPreprocessing Summary:")
    print(f"Rows before: {original_rows}, after: {final_rows} (removed {original_rows - final_rows})")
    print(f"Columns before: {original_cols}, after: {final_cols}")
    print(f"New columns added: {final_cols - original_cols}")

    return df
