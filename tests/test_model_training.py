import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from train_models import train_models

@pytest.fixture
def sample_data():
    np.random.seed(42)
    size = 500

    return pd.DataFrame({
        'BasketValue': np.random.exponential(scale=100, size=size),
        'TotalPrice': np.random.exponential(scale=100, size=size),
        'Quantity': np.random.randint(1, 10, size=size),
        'UnitPrice': np.random.uniform(1, 50, size=size),
        'Country': np.random.choice(['UK', 'France', 'Germany'], size=size),
        'Hour': np.random.randint(0, 24, size=size),
        'Day': np.random.randint(1, 31, size=size),
        'Month': np.random.randint(1, 13, size=size),
        'DayOfWeek': np.random.randint(0, 7, size=size),
        'CustomerID': np.random.randint(10000, 11000, size=size),
        'InvoiceNo': [f'INV{i}' for i in range(size)],
        'StockCode': [f'P{i%50}' for i in range(size)],
        'Description': ['Product'] * size,
        'InvoiceDate': pd.date_range(start='2021-01-01', periods=size, freq='h')  # 'h' to avoid warning
    })

def test_train_models_returns_expected_structure(sample_data):
    # Mock: skip model file checks, avoid joblib save/load, and suppress plots
    with patch("matplotlib.pyplot.show"), \
         patch("os.path.exists", return_value=False), \
         patch("joblib.dump"):
        results = train_models(sample_data)

    # Validate structure
    assert isinstance(results, dict)
    assert all(model in results for model in ['Logistic Regression', 'Random Forest', 'XGBoost'])

    for model_name, model_result in results.items():
        assert isinstance(model_result, dict)
        for metric in ['model', 'f1', 'accuracy', 'precision', 'recall', 'roc_auc', 'avg_precision']:
            assert metric in model_result
        assert isinstance(model_result['f1'], float)
