"""
Model Evaluation and Discussion Module

This module provides in-depth evaluation of machine learning models for
high-value purchase prediction, including performance analysis, business
implications, improvement suggestions, and ethical considerations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc


def evaluate_models(data, model_results):
    """
    Evaluate machine learning models and discuss results.

    Args:
        data (pd.DataFrame): Feature-engineered transaction data
        model_results (dict): Results from model training

    Returns:
        dict: Evaluation results and discussion
    """
    print("\n=== MODEL EVALUATION & DISCUSSION ===")

    if 'IsHighValue' not in data.columns:
        if 'BasketValue' in data.columns:
            threshold = data['BasketValue'].quantile(0.75)
            data['IsHighValue'] = data['BasketValue'] > threshold
            print(f"Target column 'IsHighValue' created using 75th percentile threshold: £{threshold:.2f}")
        else:
            raise ValueError("'BasketValue' column is required to compute 'IsHighValue' target.")

    # Store evaluation results
    evaluation_results = {
        'model_performance': {},
        'business_interpretation': {},
        'model_comparison': {},
        'improvement_suggestions': [],
        'ethical_considerations': []
    }

    # 1. Evaluate Model Performance
    print("\n1. Model Performance Evaluation:")
    model_performance = evaluate_model_performance(model_results)
    evaluation_results['model_performance'] = model_performance

    # 2. Business Interpretation
    print("\n2. Business Interpretation:")
    business_interpretation = interpret_business_impact(model_results, data)
    evaluation_results['business_interpretation'] = business_interpretation

    # 3. Compare Model Results
    print("\n3. Model Comparison:")
    model_comparison = compare_model_results(model_results)
    evaluation_results['model_comparison'] = model_comparison

    # 4. Suggest Improvements
    print("\n4. Suggested Improvements:")
    improvement_suggestions = suggest_improvements(model_results)
    evaluation_results['improvement_suggestions'] = improvement_suggestions

    # 5. Ethical Considerations
    print("\n5. Ethical Considerations:")
    ethical_considerations = discuss_ethical_considerations()
    evaluation_results['ethical_considerations'] = ethical_considerations

    return evaluation_results


def evaluate_model_performance(model_results):
    """
    Evaluate model performance based on various metrics.

    Args:
        model_results (dict): Results from model training

    Returns:
        dict: Performance evaluation results
    """
    # Identify the best model
    best_model_name = max(model_results, key=lambda k: model_results[k]['f1'])
    best_model_metrics = model_results[best_model_name]

    print(f"\n1.1 Best Model: {best_model_name}")
    print(f"   - Accuracy:  {best_model_metrics['accuracy']:.4f}")
    print(f"   - Precision: {best_model_metrics['precision']:.4f}")
    print(f"   - Recall:    {best_model_metrics['recall']:.4f}")
    print(f"   - F1 Score:  {best_model_metrics['f1']:.4f}")
    print(f"   - ROC AUC:   {best_model_metrics['roc_auc']:.4f}")
    print(f"   - Avg Precision: {best_model_metrics['avg_precision']:.4f}")

    # Interpret metrics in business context
    print("\n1.2 Metric Interpretation:")

    # Precision interpretation
    if best_model_metrics['precision'] > 0.8:
        precision_interpretation = (
            f"High precision ({best_model_metrics['precision']:.2f}) indicates that when the model "
            f"predicts a high-value purchase, it is correct about {best_model_metrics['precision'] * 100:.1f}% "
            f"of the time. This enables confident targeting of customers for premium promotions "
            f"with minimal wasted marketing spend."
        )
    elif best_model_metrics['precision'] > 0.6:
        precision_interpretation = (
            f"Moderate precision ({best_model_metrics['precision']:.2f}) means that when the model "
            f"identifies high-value purchases, it is correct about {best_model_metrics['precision'] * 100:.1f}% "
            f"of the time. This is useful for targeted marketing, but expect some wasted resources "
            f"on false positives."
        )
    else:
        precision_interpretation = (
            f"Low precision ({best_model_metrics['precision']:.2f}) suggests that many predicted high-value "
            f"purchases are false positives. This could lead to inefficient allocation of marketing "
            f"resources if used for targeting."
        )

    print(precision_interpretation)

    # Recall interpretation
    if best_model_metrics['recall'] > 0.8:
        recall_interpretation = (
            f"High recall ({best_model_metrics['recall']:.2f}) shows that the model successfully "
            f"identifies {best_model_metrics['recall'] * 100:.1f}% of all actual high-value purchases. "
            f"This minimizes missed opportunities and ensures most valuable customers are targeted."
        )
    elif best_model_metrics['recall'] > 0.6:
        recall_interpretation = (
            f"Moderate recall ({best_model_metrics['recall']:.2f}) indicates the model captures "
            f"{best_model_metrics['recall'] * 100:.1f}% of all high-value purchases, but misses some "
            f"opportunities. Consider whether these missed opportunities are acceptable."
        )
    else:
        recall_interpretation = (
            f"Low recall ({best_model_metrics['recall']:.2f}) means the model misses many actual "
            f"high-value purchases. This results in significant missed opportunities for targeting "
            f"valuable customers."
        )

    print(recall_interpretation)

    # ROC AUC interpretation
    if best_model_metrics['roc_auc'] > 0.9:
        auc_interpretation = (
            f"Excellent discriminative ability (AUC = {best_model_metrics['roc_auc']:.2f}) indicates "
            f"the model can effectively distinguish between high-value and standard-value purchases "
            f"across different threshold settings."
        )
    elif best_model_metrics['roc_auc'] > 0.8:
        auc_interpretation = (
            f"Good discriminative ability (AUC = {best_model_metrics['roc_auc']:.2f}) shows the model "
            f"is generally effective at distinguishing between high-value and standard-value purchases."
        )
    elif best_model_metrics['roc_auc'] > 0.7:
        auc_interpretation = (
            f"Acceptable discriminative ability (AUC = {best_model_metrics['roc_auc']:.2f}), but there "
            f"is room for improvement in distinguishing between purchase types."
        )
    else:
        auc_interpretation = (
            f"Limited discriminative ability (AUC = {best_model_metrics['roc_auc']:.2f}) suggests the "
            f"model struggles to reliably distinguish between high-value and standard-value purchases."
        )

    print(auc_interpretation)

    # Overall model quality interpretation
    if best_model_metrics['f1'] > 0.8:
        overall_interpretation = (
            f"The {best_model_name} model shows excellent performance in predicting high-value purchases, "
            f"with an F1 score of {best_model_metrics['f1']:.2f}. This model can be confidently deployed "
            f"to support marketing decision-making and customer targeting."
        )
    elif best_model_metrics['f1'] > 0.7:
        overall_interpretation = (
            f"The {best_model_name} model shows good performance in predicting high-value purchases, "
            f"with an F1 score of {best_model_metrics['f1']:.2f}. It provides valuable insights for "
            f"marketing strategies, though there is some room for improvement."
        )
    elif best_model_metrics['f1'] > 0.6:
        overall_interpretation = (
            f"The {best_model_name} model shows moderate performance in predicting high-value purchases, "
            f"with an F1 score of {best_model_metrics['f1']:.2f}. It offers useful guidance for marketing "
            f"decisions, but should be used with some caution and human oversight."
        )
    else:
        overall_interpretation = (
            f"The {best_model_name} model shows limited performance in predicting high-value purchases, "
            f"with an F1 score of {best_model_metrics['f1']:.2f}. While it provides some value over random "
            f"targeting, significant improvements are needed before full deployment."
        )

    print("\n1.3 Overall Model Performance:")
    print(overall_interpretation)

    # Return performance evaluation
    performance_evaluation = {
        'best_model': best_model_name,
        'metrics': best_model_metrics,
        'precision_interpretation': precision_interpretation,
        'recall_interpretation': recall_interpretation,
        'auc_interpretation': auc_interpretation,
        'overall_interpretation': overall_interpretation
    }

    return performance_evaluation


def interpret_business_impact(model_results, data):
    """
    Interpret the business impact of the model results.

    Args:
        model_results (dict): Results from model training
        data (pd.DataFrame): Feature-engineered transaction data

    Returns:
        dict: Business interpretation results
    """
    # Identify the best model
    best_model_name = max(model_results, key=lambda k: model_results[k]['f1'])
    best_model_metrics = model_results[best_model_name]

    # Calculate relevant business metrics
    avg_transaction_value = data['TotalPrice'].mean()
    high_value_threshold = data['BasketValue'].quantile(0.75)
    avg_high_value = data[data['BasketValue'] > high_value_threshold]['BasketValue'].mean()
    avg_standard_value = data[data['BasketValue'] <= high_value_threshold]['BasketValue'].mean()
    value_difference = avg_high_value - avg_standard_value

    print(f"\n2.1 Key Business Metrics:")
    print(f"   - Average transaction value: £{avg_transaction_value:.2f}")
    print(f"   - High-value threshold: £{high_value_threshold:.2f}")
    print(f"   - Average high-value purchase: £{avg_high_value:.2f}")
    print(f"   - Average standard-value purchase: £{avg_standard_value:.2f}")
    print(f"   - Value difference: £{value_difference:.2f}")

    # Calculate potential financial impact
    print("\n2.2 Financial Impact Analysis:")

    # Scenario: Marketing campaign targeting 1,000 customers
    campaign_size = 1000
    marketing_cost_per_customer = 5  # Assumed cost in £

    # Baseline approach (random targeting)
    random_high_value_rate = data['IsHighValue'].mean()
    random_expected_conversions = campaign_size * random_high_value_rate
    random_expected_revenue = random_expected_conversions * avg_high_value
    random_marketing_cost = campaign_size * marketing_cost_per_customer
    random_profit = random_expected_revenue - random_marketing_cost

    print(f"   Baseline (Random Targeting):")
    print(f"   - Campaign size: {campaign_size} customers")
    print(f"   - Expected high-value rate: {random_high_value_rate:.2f}")
    print(f"   - Expected conversions: {random_expected_conversions:.1f}")
    print(f"   - Expected revenue: £{random_expected_revenue:.2f}")
    print(f"   - Marketing cost: £{random_marketing_cost:.2f}")
    print(f"   - Expected profit: £{random_profit:.2f}")

    # Model-based targeting
    # Assuming we target customers predicted as high-value
    # Precision: Proportion of true high-value among predicted high-value
    model_target_size = campaign_size
    model_precision = best_model_metrics['precision']
    model_expected_conversions = model_target_size * model_precision
    model_expected_revenue = model_expected_conversions * avg_high_value
    model_marketing_cost = model_target_size * marketing_cost_per_customer
    model_profit = model_expected_revenue - model_marketing_cost

    print(f"\n   Model-Based Targeting:")
    print(f"   - Campaign size: {model_target_size} customers")
    print(f"   - Model precision: {model_precision:.2f}")
    print(f"   - Expected conversions: {model_expected_conversions:.1f}")
    print(f"   - Expected revenue: £{model_expected_revenue:.2f}")
    print(f"   - Marketing cost: £{model_marketing_cost:.2f}")
    print(f"   - Expected profit: £{model_profit:.2f}")

    # Calculate improvement
    profit_increase = model_profit - random_profit
    profit_increase_pct = (profit_increase / random_profit) * 100 if random_profit > 0 else float('inf')
    roi = profit_increase / model_marketing_cost * 100

    print(f"\n   Improvement:")
    print(f"   - Profit increase: £{profit_increase:.2f} ({profit_increase_pct:.1f}%)")
    print(f"   - ROI: {roi:.1f}%")
    print(f"   - Additional conversions: {model_expected_conversions - random_expected_conversions:.1f}")

    # Strategic recommendations
    print("\n2.3 Strategic Recommendations:")

    print("   1. Implement targeted marketing campaigns using the model's predictions")
    print("   2. Focus on customers with features identified as important by the model")
    print("   3. Develop personalized offers based on predicted purchase value")
    print("   4. Test and refine the model through A/B testing with actual campaigns")
    print("   5. Monitor performance over time and retrain the model as customer behavior evolves")

    # Long-term business value
    print("\n2.4 Long-Term Business Value:")

    print("   - Improved customer experience through more relevant offers")
    print("   - Reduced marketing waste and improved sustainability")
    print("   - Enhanced customer loyalty through appropriate targeting")
    print("   - More efficient allocation of marketing resources")
    print("   - Data-driven decision-making culture")

    # Return business interpretation
    business_interpretation = {
        'avg_transaction_value': avg_transaction_value,
        'high_value_threshold': high_value_threshold,
        'avg_high_value': avg_high_value,
        'avg_standard_value': avg_standard_value,
        'value_difference': value_difference,
        'random_profit': random_profit,
        'model_profit': model_profit,
        'profit_increase': profit_increase,
        'profit_increase_pct': profit_increase_pct,
        'roi': roi
    }

    return business_interpretation


def compare_model_results(model_results):
    """
    Compare different model results and discuss limitations.

    Args:
        model_results (dict): Results from model training

    Returns:
        dict: Model comparison and limitations
    """
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'F1 Score': [model_results[m]['f1'] for m in model_results],
        'Accuracy': [model_results[m]['accuracy'] for m in model_results],
        'Precision': [model_results[m]['precision'] for m in model_results],
        'Recall': [model_results[m]['recall'] for m in model_results],
        'ROC AUC': [model_results[m]['roc_auc'] for m in model_results]
    }).sort_values('F1 Score', ascending=False)

    print("\n3.1 Model Performance Comparison:")
    print(comparison_df)

    # Performance gap analysis
    best_model = comparison_df.iloc[0]['Model']
    worst_model = comparison_df.iloc[-1]['Model']
    performance_gap = comparison_df.iloc[0]['F1 Score'] - comparison_df.iloc[-1]['F1 Score']

    print(f"\n3.2 Performance Gap Analysis:")
    print(f"   - Best performing model: {best_model}")
    print(f"   - Worst performing model: {worst_model}")
    print(f"   - Performance gap (F1 Score): {performance_gap:.4f}")

    if performance_gap > 0.1:
        gap_assessment = (
            f"The substantial performance gap between models suggests that model selection is critical "
            f"for this task, with {best_model} significantly outperforming other approaches."
        )
    elif performance_gap > 0.05:
        gap_assessment = (
            f"The moderate performance gap between models indicates that while {best_model} performs best, "
            f"other models also provide reasonable results and might be suitable alternatives."
        )
    else:
        gap_assessment = (
            f"The small performance gap between models suggests that multiple approaches work similarly well "
            f"for this task, and factors like interpretability or deployment efficiency might guide selection."
        )

    print(gap_assessment)

    # Model-specific strengths and limitations
    print("\n3.3 Model-Specific Analysis:")

    model_analysis = {}

    for model_name in model_results.keys():
        print(f"\n   {model_name}:")

        if model_name == 'Logistic Regression':
            strengths = [
                "Highly interpretable with clear feature coefficients",
                "Efficient training and prediction",
                "Works well with linearly separable data",
                "Provides probability estimates for predictions"
            ]
            limitations = [
                "Limited capacity to capture non-linear relationships",
                "May underperform for complex patterns",
                "Sensitive to feature scaling",
                "Requires careful handling of categorical features"
            ]
        elif model_name == 'Random Forest':
            strengths = [
                "Captures non-linear patterns and feature interactions",
                "Robust to outliers and noisy data",
                "Provides feature importance measures",
                "Less prone to overfitting than single decision trees",
                "Handles high-dimensional data well"
            ]
            limitations = [
                "Less interpretable than linear models",
                "Feature importance can be biased toward high cardinality features",
                "May be computationally expensive for large datasets",
                "Requires tuning of multiple hyperparameters"
            ]
        elif model_name == 'XGBoost':
            strengths = [
                "Often achieves state-of-the-art performance on structured data",
                "Handles complex patterns effectively",
                "Built-in regularization to prevent overfitting",
                "Efficient handling of missing values",
                "Scales well to large datasets with parallel processing"
            ]
            limitations = [
                "Less interpretable than simpler models",
                "Sensitive to hyperparameter settings",
                "Can overfit if not properly tuned",
                "May be more complex to implement and maintain",
                "Black-box nature may limit transparency in some contexts"
            ]
        else:
            strengths = ["Good general-purpose model"]
            limitations = ["Performance depends on data characteristics"]

        print("     Strengths:")
        for strength in strengths:
            print(f"     - {strength}")

        print("     Limitations:")
        for limitation in limitations:
            print(f"     - {limitation}")

        model_analysis[model_name] = {
            'strengths': strengths,
            'limitations': limitations
        }

    # General limitations of the modeling approach
    print("\n3.4 General Limitations:")

    general_limitations = [
        "Models are limited to patterns present in historical data",
        "Assumes future customer behavior will follow similar patterns",
        "Limited to available transaction data; missing potentially valuable demographic information",
        "Temporal effects like seasonality may not be fully captured",
        "No visibility into competitors' activities that might influence customer behavior",
        "Models do not account for external market factors like economic conditions",
        "Predictions are probabilistic and subject to uncertainty",
        "Limited to the specific business context of the training data"
    ]

    for limitation in general_limitations:
        print(f"   - {limitation}")

    # Return comparison results
    comparison_results = {
        'comparison_df': comparison_df.to_dict(),
        'best_model': best_model,
        'worst_model': worst_model,
        'performance_gap': performance_gap,
        'gap_assessment': gap_assessment,
        'model_analysis': model_analysis,
        'general_limitations': general_limitations
    }

    return comparison_results


def suggest_improvements(model_results):
    """
    Suggest potential improvements and future work.

    Args:
        model_results (dict): Results from model training

    Returns:
        dict: Suggested improvements
    """
    # Data improvements
    print("\n4.1 Data Improvements:")
    data_improvements = [
        "Collect customer demographic data (age, gender, income) to enhance predictions",
        "Include product category information for more detailed analysis",
        "Gather customer feedback and satisfaction scores to correlate with purchase behavior",
        "Incorporate longer time periods to capture seasonal patterns and trends",
        "Collect marketing campaign exposure data to measure impact on high-value purchases",
        "Include competitor pricing information to understand price sensitivity",
        "Gather web browsing behavior prior to purchase for early prediction"
    ]

    for improvement in data_improvements:
        print(f"   - {improvement}")

    # Feature engineering improvements
    print("\n4.2 Feature Engineering Improvements:")
    feature_improvements = [
        "Develop more sophisticated temporal features to capture seasonal effects",
        "Create customer lifecycle stage indicators (new, growing, mature, at-risk)",
        "Implement more granular price sensitivity metrics at customer level",
        "Generate features representing the customer journey before purchase",
        "Create more interaction features between customer and product attributes",
        "Implement time-based features showing changes in purchase behavior",
        "Develop more sophisticated RFM segmentation with weighted components"
    ]

    for improvement in feature_improvements:
        print(f"   - {improvement}")

    # Model improvements
    print("\n4.3 Model Improvements:")
    model_improvements = [
        "Implement ensemble methods combining predictions from multiple models",
        "Conduct more extensive hyperparameter optimization with Optuna",
        "Develop separate models for different customer segments",
        "Implement cost-sensitive learning to account for different misclassification costs",
        "Try deep learning approaches for capturing complex patterns",
        "Implement calibrated probability estimates for better threshold selection",
        "Develop temporal models to capture evolving purchase patterns"
    ]

    for improvement in model_improvements:
        print(f"   - {improvement}")

    # Evaluation improvements
    print("\n4.4 Evaluation Improvements:")
    evaluation_improvements = [
        "Implement k-fold cross-validation for more robust performance estimates",
        "Evaluate model performance across different customer segments",
        "Test model performance across different time periods to assess temporal stability",
        "Develop custom evaluation metrics aligned with specific business objectives",
        "Calculate financial impact metrics (ROI, profit lift) for different threshold settings",
        "Conduct A/B testing to validate model effectiveness in real scenarios",
        "Implement a monitoring system to track model performance over time"
    ]

    for improvement in evaluation_improvements:
        print(f"   - {improvement}")

    # Implementation improvements
    print("\n4.5 Implementation Improvements:")
    implementation_improvements = [
        "Develop an automated retraining pipeline to keep the model up-to-date",
        "Implement real-time scoring for immediate purchase prediction",
        "Create an interactive dashboard for marketing teams to explore predictions",
        "Develop a system for capturing feedback on prediction accuracy",
        "Implement A/B testing framework to compare model versions",
        "Develop API endpoints for seamless integration with marketing systems",
        "Create a model explanation interface for business users"
    ]

    for improvement in implementation_improvements:
        print(f"   - {improvement}")

    # Prioritized recommendations
    print("\n4.6 Top 5 Prioritized Recommendations:")
    top_recommendations = [
        "Collect customer demographic data to enhance the predictive power of the model",
        "Implement an ensemble approach combining the strengths of multiple models",
        "Develop separate models for different customer segments for more targeted predictions",
        "Conduct A/B testing with marketing campaigns to validate the model's effectiveness",
        "Implement a model monitoring system to track performance and trigger retraining when needed"
    ]

    for i, recommendation in enumerate(top_recommendations, 1):
        print(f"   {i}. {recommendation}")

    # Return improvement suggestions
    improvement_suggestions = {
        'data_improvements': data_improvements,
        'feature_improvements': feature_improvements,
        'model_improvements': model_improvements,
        'evaluation_improvements': evaluation_improvements,
        'implementation_improvements': implementation_improvements,
        'top_recommendations': top_recommendations
    }

    return improvement_suggestions


def discuss_ethical_considerations():
    """
    Discuss ethical considerations related to the dataset and model predictions.

    Returns:
        dict: Ethical considerations
    """
    # Privacy concerns
    print("\n5.1 Privacy Considerations:")
    privacy_concerns = [
        "Transaction data contains sensitive information about customer spending habits",
        "Long-term transaction history enables detailed profiling of individual behavior",
        "Combining transaction data with other sources could lead to de-anonymization",
        "Model predictions about spending behavior may constitute financial profiling",
        "Storage and access to individual-level purchase predictions raises privacy concerns",
        "Customer may not be aware their data is being used for predictive modeling"
    ]

    for concern in privacy_concerns:
        print(f"   - {concern}")

    # Bias and fairness
    print("\n5.2 Bias and Fairness:")
    bias_concerns = [
        "The dataset may over-represent certain customer demographics, leading to biased predictions",
        "Models might penalize customers with irregular purchasing patterns due to income variability",
        "High-value predictions could systematically favor higher-income customer segments",
        "Geographic biases in the data may lead to unequal treatment of different regions",
        "Models trained on historical data may perpetuate existing inequalities in marketing attention",
        "Different cultural spending patterns may not be equally recognized by the model"
    ]

    for concern in bias_concerns:
        print(f"   - {concern}")

    # Transparency
    print("\n5.3 Transparency and Explainability:")
    transparency_issues = [
        "Complex models like XGBoost lack inherent transparency in their decision-making",
        "Customers may not understand how they are being profiled or why they receive certain offers",
        "Feature importances may not fully explain specific predictions for individual customers",
        "Marketing teams may not understand the limitations of the model's predictions",
        "Difficulty in explaining prediction errors could erode trust in the system",
        "Black-box nature of complex models may conflict with regulatory requirements"
    ]

    for issue in transparency_issues:
        print(f"   - {issue}")

    # Consent and autonomy
    print("\n5.4 Consent and Autonomy:")
    consent_issues = [
        "Customers may not have explicitly consented to predictive modeling of their behavior",
        "Opt-out options for data use in modeling may be unclear or difficult to exercise",
        "Targeted marketing based on predictions may feel intrusive to some customers",
        "Automated decision-making about marketing approach limits human judgment",
        "Persistent profiling may undermine customer autonomy in purchasing decisions",
        "Lack of awareness about data usage may compromise informed consent"
    ]

    for issue in consent_issues:
        print(f"   - {issue}")

    # Recommendations for ethical implementation
    print("\n5.5 Ethical Implementation Recommendations:")
    ethical_recommendations = [
        "Implement a comprehensive data governance framework with clear usage guidelines",
        "Conduct regular algorithmic fairness audits across different customer segments",
        "Provide transparent opt-out mechanisms for predictive modeling and targeted marketing",
        "Develop simplified model explanations for both customers and staff",
        "Establish an ethics committee to review new applications of customer data",
        "Create clear privacy notices about how purchase data is used for modeling",
        "Implement differential privacy techniques to protect individual transaction data",
        "Establish maximum retention periods for transaction data and derived models",
        "Develop guidelines for appropriate use of high-value purchase predictions",
        "Include diverse perspectives in the design and evaluation of customer models"
    ]

    for recommendation in ethical_recommendations:
        print(f"   - {recommendation}")

    # Return ethical considerations
    ethical_considerations = {
        'privacy_concerns': privacy_concerns,
        'bias_concerns': bias_concerns,
        'transparency_issues': transparency_issues,
        'consent_issues': consent_issues,
        'ethical_recommendations': ethical_recommendations
    }

    return ethical_considerations