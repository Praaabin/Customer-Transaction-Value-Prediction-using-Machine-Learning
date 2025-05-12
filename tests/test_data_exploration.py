import os
import pandas as pd
import pytest
from unittest.mock import patch
from src.data_exploration import explore_data

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'InvoiceNo': ['1001', '1002', '1003'],
        'StockCode': ['A', 'B', 'C'],
        'Quantity': [10, -5, 7],
        'InvoiceDate': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-02 15:00:00', '2023-01-03 20:00:00']),
        'UnitPrice': [2.5, 5.0, 10.0],
        'CustomerID': [12345, 12346, None],
        'Country': ['United Kingdom', 'France', 'Germany']
    })

def test_explore_data_returns_expected_keys(sample_data):
    with patch("matplotlib.pyplot.show"):  # suppress plots
        summary = explore_data(sample_data)

    expected_keys = {
        'shape',
        'date_range',
        'missing_values',
        'duplicate_rows',
        'total_revenue',
        'avg_transaction',
        'return_rate_pct',
        'top_country',
        'top_country_pct'
    }

    assert isinstance(summary, dict)
    assert expected_keys.issubset(summary.keys())

def test_explore_data_handles_missing_totalprice(sample_data):
    df = sample_data.drop(columns=['Quantity', 'UnitPrice'], errors='ignore')
    df['Quantity'] = [1, 2, 3]
    df['UnitPrice'] = [1.0, 2.0, 3.0]
    df.drop(columns=['InvoiceDate'], inplace=True)
    df['InvoiceDate'] = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03'])

    with patch("matplotlib.pyplot.show"):
        summary = explore_data(df)

    assert "total_revenue" in summary
    assert summary["total_revenue"] > 0
