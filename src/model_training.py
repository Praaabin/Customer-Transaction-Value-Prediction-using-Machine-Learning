import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

import warnings
warnings.filterwarnings('ignore')


def plot_learning_curve(estimator, title, X, y, cv, scoring='f1'):
    """
    Generate a learning curve plot showing training vs validation scores.

    Args:
        estimator: scikit-learn pipeline or model
        title (str): Plot title
        X (DataFrame): Feature set
        y (Series): Target
        cv: Cross-validation strategy
        scoring (str): Scoring metric
    """
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=1, shuffle=True, random_state=42
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, 'o-', label='Training score')
    plt.plot(train_sizes, val_mean, 'o-', label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_scores.std(axis=1),
                     train_mean + train_scores.std(axis=1), alpha=0.1)
    plt.fill_between(train_sizes, val_mean - val_scores.std(axis=1),
                     val_mean + val_scores.std(axis=1), alpha=0.1)
    plt.title(title)
    plt.xlabel('Training Size')
    plt.ylabel(scoring.upper())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def train_models(data: pd.DataFrame) -> dict:
    """
    Train and evaluate Logistic Regression, Random Forest, and XGBoost classifiers.

    Args:
        data (pd.DataFrame): The input dataset with features and target

    Returns:
        dict: Dictionary of evaluation results and trained models
    """
    print("\n=== MODEL IMPLEMENTATION ===")

    # Create binary target
    df = data.copy()
    threshold = df['BasketValue'].quantile(0.75)
    df['IsHighValue'] = df['BasketValue'] > threshold
    print(f"Target variable 'IsHighValue' defined as BasketValue > £{threshold:.2f}")

    exclude_cols = ['InvoiceNo', 'StockCode', 'Description', 'InvoiceDate',
                    'TotalPrice', 'BasketValue', 'IsHighValue', 'CustomerID']
    X = df.drop(columns=exclude_cols)
    y = df['IsHighValue']

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"Using {len(numerical_features)} numeric and {len(categorical_features)} categorical features")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # Preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    results = {}

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    print("\n=== TRAINING MODELS WITH OPTUNA ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n--- {name} ---")

        def objective(trial):
            params = {}
            if name == 'Logistic Regression':
                params = {
                    'classifier__C': trial.suggest_float('C', 0.01, 100, log=True),
                    'classifier__penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                    'classifier__solver': 'liblinear',
                    'classifier__class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
                }
            elif name == 'Random Forest':
                params = {
                    'classifier__n_estimators': trial.suggest_int('n_estimators', 50, 150),
                    'classifier__max_depth': trial.suggest_int('max_depth', 3, 12),
                    'classifier__min_samples_split': trial.suggest_int('min_samples_split', 2, 5),
                    'classifier__min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'classifier__class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])
                }
            elif name == 'XGBoost':
                params = {
                    'classifier__n_estimators': trial.suggest_int('n_estimators', 50, 150),
                    'classifier__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'classifier__max_depth': trial.suggest_int('max_depth', 3, 6),
                    'classifier__subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'classifier__reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                    'classifier__reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                }

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            pipeline.set_params(**params)
            return cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1').mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=5, show_progress_bar=True)

        best_params = {k.replace('classifier__', ''): v for k, v in study.best_params.items()}
        model.set_params(**best_params)

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)

        # Plot learning curve
        plot_learning_curve(pipeline, f"{name} Learning Curve", X_train, y_train, cv=cv, scoring='f1')

        # Evaluation
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)

        results[name] = {
            'model': pipeline,
            'f1': f1_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': auc(fpr, tpr),
            'avg_precision': average_precision_score(y_test, y_proba),
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision,
            'recall_curve': recall
        }

        print(f"F1: {results[name]['f1']:.4f}, Accuracy: {results[name]['accuracy']:.4f}, ROC AUC: {results[name]['roc_auc']:.4f}")

    # Visualize ROC and PR curves
    print("\n=== MODEL PERFORMANCE VISUALIZATION ===")
    for name, res in results.items():
        plt.figure(figsize=(10, 4))

        # ROC Curve
        plt.subplot(1, 2, 1)
        plt.plot(res['fpr'], res['tpr'], label=f"AUC = {res['roc_auc']:.2f}")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title(f"{name} - ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()

        # Precision-Recall Curve
        plt.subplot(1, 2, 2)
        plt.plot(res['recall_curve'], res['precision_curve'],
                 label=f"AP = {res['avg_precision']:.2f}")
        plt.title(f"{name} - Precision-Recall")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()

        plt.tight_layout()
        plt.show()

    return results
