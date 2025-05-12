## Customer Transaction Value Prediction using Machine Learning

A machine learning solution to identify and predict high-value customer transactions using real-world online retail data. This project incorporates industry-standard practices in data preprocessing, advanced feature engineering, model optimization, performance visualization, and ethical evaluation, all aimed at enabling targeted marketing strategies and business insights.

---

## 🚀 Project Overview

This project predicts whether a customer transaction is "high-value" based on historical purchase data. By applying supervised machine learning models, we aim to help businesses:

- Identify high-spending customers,
- Enhance targeted marketing campaigns,
- Optimize resource allocation for promotions,
- Maximize ROI through data-driven decision-making.

---

## 📊 Key Results

- **Best Performing Model**: XGBoost
- **F1 Score**: 0.9998
- **ROC AUC**: 1.0000
- **Avg Precision**: 1.0000
- **Business Insight**: Model-driven targeting improves campaign ROI by **over 100%** vs random targeting.

---

## 📁 Project Structure

```plaintext
customer_transaction_ml_project/
│
├── data/
│   ├── raw/                         # Raw transaction CSV
│   └── processed/                   # Cleaned and feature datasets
│
├── models/                          # Saved models (.joblib)
│
├── src/
│   ├── data_preprocessing.py       # Cleans raw data
│   ├── feature_engineering.py      # Feature creation (RFM, PCA, etc.)
│   ├── model_training.py           # Model training, tuning, caching
│   ├── model_evaluation.py         # Metrics & business insights
│   ├── data_exploration.py         # Optional EDA/visualization
│   ├── utils.py                    # Helper functions
│   └── __init__.py
│
├── tests/                          # Unit test files
│   ├── test_data_preprocessing.py
│   ├── test_feature_engineering.py      
│   ├── test_model_training.py           
│   ├── test_model_evaluation.py         
│   └── test_data_exploration.py  
│
├── .github/workflows/              # GitHub Actions config
│   └── python-ci.yml
│
├── main.py                         # Pipeline entry point
├── requirements.txt                # Python dependencies
├── .gitignore                      # Files to ignore
└── README.md                       # This file

```
## 🧪 Machine Learning Pipeline

````
📦 Raw Data
     ⬇
🧹 Preprocessing (Missing, Duplicates, Outliers)
     ⬇
🛠 Feature Engineering (RFM, PCA, etc.)
     ⬇
📊 Model Training (Logistic, RF, XGBoost)
     ⬇
⚙️ Hyperparameter Tuning (Optuna)
     ⬇
📈 Evaluation & Visualizations
     ⬇
💼 Business & Ethical Interpretation

````

## 🚀 How to Run the Project

```
1. Clone the Repo
    git clone https://github.com/Praaabin/Assessment-cps5010.git
    cd ML_project

2. Create Virtual Environment
    python -m venv .venv
    source .venv/bin/activate  # or .venv\Scripts\activate on Windows
    
3. Install Dependencies
    pip install -r requirements.txt

4. Run Main Pipeline
    python main.py

5. Run Unit Tests
   pytest tests/(file-path)  #file you want to test
```

## Methodology

1. Data Cleaning: Removed outliers, corrected datatypes, dropped invalid entries.
2. Feature Engineering:
   - RFM, product, time, geographic, behavioral, and PCA features.
3. Modeling:
   - Logistic Regression, Random Forest, XGBoost (Optuna-tuned)
   - Caching to prevent retraining, pipelines for reproducibility
4. Evaluation:
   - Metrics: Accuracy, F1, ROC AUC, PR AUC
   - Plots: ROC, Precision-Recall, Learning Curves
5. Business Impact Analysis:
   - Campaign ROI simulation
   - High-value thresholds
   - Strategic recommendations

---

## 📈 Visualizations

- Exploratory Analysis: Product demand, sales trends, return patterns
- Model Evaluation:
  - ROC Curve
  - Precision-Recall Curve
  - Learning Curve
- Business Insights:
  - Average order value comparisons
  - Campaign profit simulations

---

## ⚙️ Technologies Used

- Language: Python 3.13
- Libraries: 
  - Data: `pandas`, `numpy`
  - ML: `scikit-learn`, `xgboost`, `optuna`
  - Visualization: `matplotlib`, `seaborn`, `plotly`
- Environment: 
  - IDE: PyCharm
  - Version Control: Git + GitHub
  - Diagrams: Draw.io

## 📌 Ethical Considerations
- Transparency: Results explained in business-friendly terms
- Fairness: Avoided proxy variables for income/demographics
- Privacy: Customer IDs anonymized
- Consent: Simulated dataset used (no real personal data)

## 📌 Limitations & Improvements
- Data is limited to transactional logs (no demographics or feedback).
- Could improve with A/B testing, multi-model ensembles, real-time scoring.
- Feature drift monitoring and continuous learning can be added in production.

## 📄 License

 This project is licensed under the General Public License.

## 👤 Author
- Prabin Pokhrel
- BSc (Hons) Computer Science – Data Science
- St Mary's University, Twickenham, London
- Git ID: @Praaabin