from typing import Any, Dict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time


def explore_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive EDA on the customer transaction dataset.

    Args:
        df (pd.DataFrame): Raw customer transaction data.

    Returns:
        Dict[str, Any]: Key insights and summary statistics.
    """
    start = time.time()
    df = df.copy()

    # 1. Ensure datetime format for InvoiceDate
    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    print("\n=== EXPLORATORY DATA ANALYSIS ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    print(f"Total transactions: {df['InvoiceNo'].nunique()}")
    print(f"Total products: {df['StockCode'].nunique()}")
    print(f"Total customers: {df['CustomerID'].nunique()}")

    # 2. Check missing values & data types
    print("\n-- Missing values & data types --")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print(missing)
    print(df.dtypes)

    # 3. Summary statistics
    print("\n-- Statistical summary --")
    print(df.describe())

    # 4. Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate rows: {duplicates} ({duplicates / len(df):.2%})")

    # 5. TotalPrice column
    if "TotalPrice" not in df.columns:
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
        print("\nCreated 'TotalPrice' column.")

    # 6. Numeric distributions (sample if too large)
    dist_cols = [
        ("Quantity", "Order Quantities"),
        ("UnitPrice", "Unit Prices (£)"),
        ("TotalPrice", "Transaction Values (£)")
    ]
    for col, title in dist_cols:
        cutoff = df[col].quantile(0.99)
        subset = df[df[col] <= cutoff]
        if len(subset) > 5000:
            subset = subset.sample(5000, random_state=42)

        plt.figure(figsize=(8, 4))
        sns.histplot(subset[col], kde=True)
        plt.title(f"Distribution of {title}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show(block=False)

    # 7. Top 10 countries
    top_countries = (
        df["Country"]
        .value_counts()
        .head(10)
        .rename_axis("Country")
        .reset_index(name="Transactions")
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=top_countries, x="Transactions", y="Country")
    plt.title("Top 10 Countries by Number of Transactions")
    plt.tight_layout()
    plt.show(block=False)

    # 8. Monthly trends
    df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    monthly = df.groupby("Month").size().reset_index(name="Transactions")
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=monthly, x="Month", y="Transactions", marker="o")
    plt.xticks(rotation=45)
    plt.title("Monthly Transactions Over Time")
    plt.tight_layout()
    plt.show(block=False)

    # 9. Correlation heatmap
    corr = df[["Quantity", "UnitPrice", "TotalPrice"]].corr()
    plt.figure(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Between Numerical Features")
    plt.tight_layout()
    plt.show(block=False)

    # 10. Anomalies
    neg_qty = (df["Quantity"] < 0).sum()
    zero_price = (df["UnitPrice"] <= 0).sum()
    high_qty = (df["Quantity"] > df["Quantity"].quantile(0.99)).sum()
    high_price = (df["UnitPrice"] > df["UnitPrice"].quantile(0.99)).sum()
    missing_cust = df["CustomerID"].isna().sum()

    print("\n-- Anomaly & missing data counts --")
    print(f"Returns (negative quantity): {neg_qty}")
    print(f"Zero or negative prices: {zero_price}")
    print(f"Extreme quantities (top 1%): {high_qty}")
    print(f"Extreme prices (top 1%): {high_price}")
    print(f"Missing CustomerIDs: {missing_cust}")

    # 11. Limitations
    top_country = df["Country"].value_counts().idxmax()
    top_pct = df["Country"].value_counts(normalize=True).max() * 100
    days_span = (df["InvoiceDate"].max() - df["InvoiceDate"].min()).days

    print("\n-- Dataset limitations & biases --")
    print(f"• {top_country} accounts for {top_pct:.1f}% of transactions.")
    print(f"• Time span covers {days_span} days.")
    print("• No demographics or product categories available.")
    print(f"• {duplicates} duplicates.")
    print(f"• {missing_cust} missing CustomerIDs.")

    # 12. Insights
    total_revenue = df["TotalPrice"].sum()
    avg_transaction = df.groupby("InvoiceNo")["TotalPrice"].sum().mean()
    items_per_txn = df.groupby("InvoiceNo").size().mean()
    return_rate = neg_qty / len(df) * 100

    print("\n-- Key insights --")
    print(f"Total revenue: £{total_revenue:,.2f}")
    print(f"Average transaction value: £{avg_transaction:,.2f}")
    print(f"Average items per transaction: {items_per_txn:.2f}")
    print(f"Return rate: {return_rate:.2f}%")

    end = time.time()
    print(f"\n[Timing] Data exploration completed in {end - start:.2f} seconds.")

    return {
        "shape": df.shape,
        "date_range": (df["InvoiceDate"].min(), df["InvoiceDate"].max()),
        "missing_values": missing.to_dict(),
        "duplicate_rows": duplicates,
        "total_revenue": total_revenue,
        "avg_transaction": avg_transaction,
        "return_rate_pct": return_rate,
        "top_country": top_country,
        "top_country_pct": top_pct,
    }
