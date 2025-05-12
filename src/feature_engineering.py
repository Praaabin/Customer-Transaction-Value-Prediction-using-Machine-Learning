import os
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def engineer_features(data, cache_path='data/processed/featured_data.csv', force_recompute=False):
    """
    Perform advanced feature engineering or load cached output if available.

    Args:
        data (pd.DataFrame): Preprocessed transaction-level data.
        cache_path (str): Path to save/load engineered dataset.
        force_recompute (bool): If True, ignore cache and recompute features.

    Returns:
        pd.DataFrame: Feature-enhanced dataset.
    """

    # -------------------------------------------------------------------------------------
    # 0. Use cached result if available and not forced to recompute
    # -------------------------------------------------------------------------------------
    if os.path.exists(cache_path) and not force_recompute:
        print(f"✅ Using cached featured dataset: {cache_path}")
        return pd.read_csv(cache_path)

    print("\n=== FEATURE ENGINEERING ===")
    df = data.copy()
    print(f"Initial shape: {df.shape}")

    if 'IsReturn' not in df.columns:
        df['IsReturn'] = df['Quantity'] < 0
        print("Created 'IsReturn' column from negative Quantity values.")

    # -------------------------------------------------------------------------------------
    # 1. Transaction-Level Features
    # -------------------------------------------------------------------------------------
    print("\n1. Transaction-Level Features")

    df = df.merge(df.groupby('InvoiceNo')['StockCode'].nunique().rename("BasketSize"), on='InvoiceNo', how='left')
    df = df.merge(df.groupby('InvoiceNo')['TotalPrice'].sum().rename("BasketValue"), on='InvoiceNo', how='left')
    df['AvgItemValue'] = df['BasketValue'] / df['BasketSize']
    df['IsSingleItem'] = df['BasketSize'] == 1
    df['TransactionDiversity'] = df['BasketSize'] / df.groupby('InvoiceNo')['Quantity'].transform('sum')

    # -------------------------------------------------------------------------------------
    # 2. Customer-Level Features
    # -------------------------------------------------------------------------------------
    print("\n2. Customer-Level Features")

    reference_date = df['InvoiceDate'].max() + timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'TotalSpend'})

    customer_metrics = df.groupby('CustomerID').agg({
        'BasketSize': 'mean',
        'BasketValue': 'mean',
        'StockCode': 'nunique',
        'IsReturn': 'sum'
    }).rename(columns={
        'BasketSize': 'AvgBasketSize',
        'BasketValue': 'AvgTransactionValue',
        'StockCode': 'ProductVariety',
        'IsReturn': 'ReturnCount'
    })

    customer_metrics['ReturnRate'] = customer_metrics['ReturnCount'] / df.groupby('CustomerID')['InvoiceNo'].nunique()
    customer_features = pd.concat([rfm, customer_metrics], axis=1)

    purchase_dates = df.groupby(['CustomerID', 'InvoiceNo'])['InvoiceDate'].max().reset_index()
    purchase_dates.sort_values(['CustomerID', 'InvoiceDate'], inplace=True)
    purchase_dates['PrevDate'] = purchase_dates.groupby('CustomerID')['InvoiceDate'].shift(1)
    purchase_dates['DaysBetween'] = (purchase_dates['InvoiceDate'] - purchase_dates['PrevDate']).dt.days
    avg_days = purchase_dates.groupby('CustomerID')['DaysBetween'].mean().fillna(0)
    customer_features = customer_features.merge(avg_days.rename("AvgDaysBetweenPurchases"), on='CustomerID', how='left')
    customer_features['PurchaseFrequency30Days'] = 30 / customer_features['AvgDaysBetweenPurchases'].replace(0, 30)

    df = df.merge(customer_features.reset_index(), on='CustomerID', how='left')

    # -------------------------------------------------------------------------------------
    # 3. Temporal Features
    # -------------------------------------------------------------------------------------
    print("\n3. Temporal Features")

    df['Quarter'] = df['Month'].apply(lambda x: (x - 1) // 3 + 1)
    df['Season'] = df['Month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6])
    df['TimeOfDay'] = df['Hour'].map({
        **dict.fromkeys(range(0, 6), 'Night'),
        **dict.fromkeys(range(6, 12), 'Morning'),
        **dict.fromkeys(range(12, 18), 'Afternoon'),
        **dict.fromkeys(range(18, 24), 'Evening')
    })
    df['IsHolidayPeriod'] = df['Month'] == 12
    df['DayOfMonthSegment'] = df['Day'].apply(lambda x: 'Beginning' if x <= 10 else ('Middle' if x <= 20 else 'End'))

    # -------------------------------------------------------------------------------------
    # 4. Geographic Features
    # -------------------------------------------------------------------------------------
    print("\n4. Geographic Features")

    uk = ['United Kingdom']
    europe = ['France', 'Germany', 'Spain', 'Belgium', 'Switzerland', 'Portugal',
              'Netherlands', 'Ireland', 'Italy', 'Norway', 'Finland', 'Denmark', 'Sweden']
    df['CountryGroup'] = df['Country'].apply(lambda x: 'UK' if x in uk else ('Europe' if x in europe else 'Other'))
    df['IsDomestic'] = df['Country'] == 'United Kingdom'

    # -------------------------------------------------------------------------------------
    # 5. Product Features
    # -------------------------------------------------------------------------------------
    print("\n5. Product Features")

    df = df.merge(df.groupby('StockCode').size().rank(ascending=False).rename("PopularityRank"), on='StockCode', how='left')
    df['PriceCategory'] = pd.qcut(df['UnitPrice'], 4, labels=['Budget', 'Standard', 'Premium', 'Luxury'])
    df = df.merge(df.groupby('StockCode')['IsReturn'].mean().rename("ProductReturnRate"), on='StockCode', how='left')

    cust_prod = df.groupby(['CustomerID', 'StockCode']).size().reset_index(name='PurchaseCount')
    total_purchases = df.groupby('CustomerID').size().reset_index(name='TotalPurchases')
    cust_prod = cust_prod.merge(total_purchases, on='CustomerID', how='left')
    cust_prod['ProductAffinity'] = cust_prod['PurchaseCount'] / cust_prod['TotalPurchases']
    df = df.merge(cust_prod[['CustomerID', 'StockCode', 'ProductAffinity']], on=['CustomerID', 'StockCode'], how='left')

    # -------------------------------------------------------------------------------------
    # 6. Interaction Features
    # -------------------------------------------------------------------------------------
    print("\n6. Interaction Features")

    spender_threshold = df.groupby('CustomerID')['TotalPrice'].sum().quantile(0.8)
    df['IsHighSpender'] = df['CustomerID'].map(
        lambda cid: df[df['CustomerID'] == cid]['TotalPrice'].sum() > spender_threshold)

    budget_ratio = df.groupby('CustomerID')['PriceCategory'].apply(
        lambda x: (x == 'Budget').mean()).reset_index(name='BudgetItemRatio')
    df = df.merge(budget_ratio, on='CustomerID', how='left')

    df['WeekendDomestic'] = df['IsWeekend'] & df['IsDomestic']
    df['HolidayLuxury'] = df['IsHolidayPeriod'] & (df['PriceCategory'] == 'Luxury')

    # -------------------------------------------------------------------------------------
    # 7. PCA on Sampled Data (Optional Speed-Up)
    # -------------------------------------------------------------------------------------
    print("\n7. Dimensionality Reduction with PCA")

    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(['BasketValue', 'TotalPrice'])

    if len(num_cols) >= 5:
        scaler = StandardScaler()
        sample = df[num_cols].sample(n=20000, random_state=42) if len(df) > 20000 else df[num_cols]
        scaled = scaler.fit_transform(sample.fillna(sample.median()))
        pca = PCA(n_components=0.95)
        pcs = pca.fit_transform(scaled)
        for i, comp in enumerate(pca.transform(df[num_cols].fillna(df[num_cols].median())).T):
            df[f'PC{i+1}'] = comp
        print(f"Added {pcs.shape[1]} principal components")

    # -------------------------------------------------------------------------------------
    # 8. Missing Value Imputation
    # -------------------------------------------------------------------------------------
    print("\n8. Handling Missing Values")
    for col in df.columns[df.isnull().any()]:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # -------------------------------------------------------------------------------------
    # 9. Summary & Save to Cache
    # -------------------------------------------------------------------------------------
    print("\n9. Feature Engineering Summary")
    print(f"Final shape: {df.shape}")
    df.to_csv(cache_path, index=False)
    print(f"✅ Featured data saved to: {cache_path}")

    return df
