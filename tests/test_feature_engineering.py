import os
import tempfile
import pandas as pd
import numpy as np
import pytest
from src.feature_engineering import engineer_features

from pandas.testing import assert_frame_equal
from unittest.mock import patch

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'InvoiceNo': ['1001', '1002', '1003'],
        'StockCode': ['A', 'B', 'C'],
        'Quantity': [10, -5, 7],
        'InvoiceDate': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-02 15:00:00', '2023-01-03 20:00:00']),
        'UnitPrice': [2.5, 5.0, 10.0],
        'CustomerID': [12345, 12346, 12345],
        'Country': ['United Kingdom', 'France', 'Germany'],
        'TotalPrice': [25.0, -25.0, 70.0],
        'Month': [1, 1, 1],
        'DayOfWeek': [6, 0, 1],
        'Hour': [10, 15, 20],
        'Day': [1, 2, 3]
    })

def test_engineer_features_output_shape_and_columns(sample_data):
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "featured_test.csv")
        result = engineer_features(sample_data, cache_path=cache_path, force_recompute=True)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)
        assert 'IsReturn' in result.columns
        assert 'BasketSize' in result.columns
        assert 'AvgBasketSize' in result.columns
        assert 'Quarter' in result.columns
        assert 'IsDomestic' in result.columns
        assert 'PriceCategory' in result.columns
        assert 'IsHighSpender' in result.columns

def test_pca_columns_exist_if_triggered(sample_data):
    big_sample = pd.concat([sample_data] * 10000, ignore_index=True)
    big_sample['UnitPrice'] = big_sample['UnitPrice'] + np.random.normal(0, 0.5, size=len(big_sample))

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "featured_test.csv")
        df = engineer_features(big_sample, cache_path=cache_path, force_recompute=True)

        pca_cols = [col for col in df.columns if col.startswith('PC')]
        assert len(pca_cols) > 0
