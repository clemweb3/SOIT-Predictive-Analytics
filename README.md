Sparse Data Predictive Framework for Educational Service Quality
Project Overview

This repository implements a multimodal machine learning pipeline designed to predict student satisfaction within the School of Information Technology (SOIT) at Mapúa University.

The primary technical challenge addressed is the N=81 Data Sparsity constraint. This project demonstrates how to bridge the gap between small-scale institutional surveys and "Big Data" requirements through synthetic data generation and NLP-enhanced feature engineering.
The Problem

Educational institutions often collect detailed survey data (29+ variables) but suffer from low participation rates. Standard statistical methods provide descriptive averages but fail to:

    Identify Predictive Weights: Determining which non-academic factor (e.g., facilities vs. IT) is the strongest leading indicator of dissatisfaction.

    Generalize: Small datasets lead to overfitted models that cannot predict the sentiment of future student cohorts.

Technical Pipeline & Reasoning
1. Preprocessing & Multimodal Fusion

    Likert Normalization: Mapping 1-5 scales to standardized floats.

    NLP Sentiment Analysis: Using a pre-trained Transformer model to convert Taglish (Tagalog-English) qualitative comments into a quantitative "Sentiment Score" feature.

    Reasoning: Raw text contains high-variance indicators (e.g., specific complaints about bidets or Wi-Fi) that numerical scales often miss.

2. Tabular Data Augmentation

    Implementation: CTGAN (Conditional Tabular GAN) or SMOTE-NC.

    Reasoning: To satisfy the requirements of high-dimensional predictive modeling, we generate a synthetic population that mirrors the statistical distribution of the original 81 responses. This allows for more robust gradient-boosting performance.

3. Predictive Modeling

    Model: XGBoost (Extreme Gradient Boosting).

    Validation: Stratified K-Fold Cross-Validation to ensure the model performs consistently across different segments of the small real-world dataset.

    Reasoning: XGBoost handles non-linear relationships and missing values more effectively than traditional Logistic Regression.

4. Explainable AI (XAI)

    Implementation: SHAP (SHapley Additive exPlanations).

    Reasoning: Modern institutional decision-making requires transparency. SHAP values allow us to decompose the model's logic, showing exactly how much each variable (e.g., "Bathroom Cleanliness") contributed to a "Dissatisfied" prediction.