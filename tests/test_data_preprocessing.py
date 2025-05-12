import pandas as pd
import numpy as np
import pytest
from src.data_preprocessing import preprocess_data

# Sample raw data fixture
@pytest.fixture
def raw_data():
    return pd.DataFrame({
        'InvoiceNo': ['1001', '1002', '1003', '1003'],
        'StockCode': ['A1', 'A2', 'A3', 'A3'],
        'Description': ['  Widget ', 'Gadget', 'Tool', 'Tool'],
        'Quantity': [1, -1, 1000, 3],
        'InvoiceDate': pd.to_datetime(['2021-01-01 10:00', '2021-01-02 12:00',
                                       '2021-01-03 13:00', '2021-01-03 13:00']),
        'UnitPrice': [2.5, 0.0, 1.2, 1.2],
        'CustomerID': [12345.0, np.nan, 12346.0, 12346.0],
        'Country': ['United Kingdom', 'France', 'Germany', 'Germany']
    })


def test_preprocessing_output_shape(raw_data):
    cleaned = preprocess_data(raw_data)
    assert isinstance(cleaned, pd.DataFrame)
    assert not cleaned.empty
    assert 'TotalPrice' in cleaned.columns
    assert 'Year' in cleaned.columns
    assert 'Month' in cleaned.columns
    assert 'Hour' in cleaned.columns


def test_no_negative_or_zero_values(raw_data):
    cleaned = preprocess_data(raw_data)
    assert (cleaned['Quantity'] > 0).all()
    assert (cleaned['UnitPrice'] > 0).all()


def test_duplicates_removed(raw_data):
    cleaned = preprocess_data(raw_data)
    assert cleaned.duplicated().sum() == 0


def test_description_trimmed(raw_data):
    cleaned = preprocess_data(raw_data)
    if 'Description' in cleaned.columns:
        assert cleaned['Description'].str.startswith(' ').sum() == 0
        assert cleaned['Description'].str.endswith(' ').sum() == 0


def test_missing_customer_id_handling(raw_data):
    cleaned = preprocess_data(raw_data)
    assert cleaned['CustomerID'].isna().sum() == 0
    assert cleaned['CustomerID'].dtype == int
