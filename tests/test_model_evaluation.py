import pytest
import pandas as pd
from src.model_evaluation import evaluate_models


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'BasketValue': [50, 200, 150, 80, 300],
        'TotalPrice': [50, 200, 150, 80, 300],
        'IsHighValue': [0, 1, 1, 0, 1]
    })


@pytest.fixture
def sample_model_results():
    return {
        'Logistic Regression': {
            'accuracy': 0.85,
            'precision': 0.75,
            'recall': 0.70,
            'f1': 0.72,
            'roc_auc': 0.82,
            'avg_precision': 0.74
        },
        'Random Forest': {
            'accuracy': 0.88,
            'precision': 0.78,
            'recall': 0.72,
            'f1': 0.75,
            'roc_auc': 0.85,
            'avg_precision': 0.77
        },
        'XGBoost': {
            'accuracy': 0.89,
            'precision': 0.80,
            'recall': 0.75,
            'f1': 0.77,
            'roc_auc': 0.87,
            'avg_precision': 0.79
        }
    }


def test_evaluate_models_output_structure(sample_data, sample_model_results):
    results = evaluate_models(sample_data, sample_model_results)

    assert isinstance(results, dict)
    assert 'model_performance' in results
    assert 'business_interpretation' in results
    assert 'model_comparison' in results
    assert 'improvement_suggestions' in results
    assert 'ethical_considerations' in results

    assert isinstance(results['model_performance'], dict)
    assert 'metrics' in results['model_performance']
    assert 'overall_interpretation' in results['model_performance']

    assert isinstance(results['business_interpretation'], dict)
    assert 'model_profit' in results['business_interpretation']
    assert 'random_profit' in results['business_interpretation']

    assert isinstance(results['model_comparison'], dict)
    assert 'comparison_df' in results['model_comparison']

    assert isinstance(results['improvement_suggestions'], dict)
    assert 'top_recommendations' in results['improvement_suggestions']

    assert isinstance(results['ethical_considerations'], dict)
    assert 'privacy_concerns' in results['ethical_considerations']
