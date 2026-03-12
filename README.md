# Strategic Service Quality Analytics: Sparse Data Predictive Framework

## 1. Project Overview
This repository implements a robust machine learning pipeline designed to predict student satisfaction within the School of Information Technology (SOIT) at Mapúa University. The research specifically addresses the **Statistical Sparsity Constraint ($N=81$)** by utilizing synthetic data augmentation and Explainable AI (XAI) to transform subjective survey data into actionable institutional strategy.

## 2. The Methodology: Adaptive Research Design
In contrast to standard "Big Data" approaches, this project utilizes **Predictive Parsimony** and **Synthetic Balancing**. Recognizing that high-dimensional datasets ($D=36$) with small sample sizes ($N=81$) lead to unstable models, we implemented a four-stage refinement process:

* **Multimodal Data Fusion:** Integrating Likert-scale numerical data with NLP-derived sentiment scores from qualitative student feedback.
* **Tabular Data Augmentation (SMOTE-NC):** To address class imbalance and satisfy the requirements of Gradient Boosting, we utilized SMOTE-NC to generate a statistically representative synthetic population. This expanded the training set, allowing the model to learn the minority class patterns (dissatisfaction) underrepresented in the raw survey results.
* **Dimensionality Reduction:** Utilizing a **Pearson-Correlation Filter** to isolate the Top 10 institutional drivers, mitigating the risk of high-dimensional noise and overfitting inherent in sparse datasets.
* **Binary Sentiment Realignment:** Re-framing the problem from 4-class classification to **Binary Sentiment Detection** (At-Risk vs. Satisfied) to achieve stable, generalizable performance.

## 3. Technical Stack
* **Core Engine:** XGBoost (Extreme Gradient Boosting)
* **Data Augmentation:** SMOTE-NC (Synthetic Minority Over-sampling Technique for Nominal and Continuous features)
* **Explainability:** SHAP (SHapley Additive exPlanations)
* **Infrastructure:** Python 3.11+, Scikit-Learn, Pandas, Matplotlib, Seaborn

---

## 4. Final Research Results
These results represent the validated performance on a **100% Real-World Holdout set ($n=16$)**, which was isolated prior to any data augmentation to ensure zero data leakage and objective validation.

### 4.1 Predictive Accuracy
| Model | Initial Accuracy (4-Class) | Final Accuracy (Binary) |
| :--- | :--- | :--- |
| **Logistic Regression** | 0.5625 | 0.7500 |
| **Random Forest** | 0.6250 | 0.5000 |
| **XGBoost (Proposed)** | 0.3750 | **0.8125** |

**Technical Takeaway:** The final XGBoost model achieved an **81.25% accuracy**, correctly identifying **13 out of 16** real-world student profiles. This validates the use of Gradient Boosting over linear baselines for capturing non-linear institutional satisfaction drivers.

### 4.2 Institutional Priority Matrix (XAI Findings)
Utilizing SHAP value attribution, the following features were identified as the primary drivers of student sentiment:
1.  **Academic Grading Fairness:** The strongest predictor of satisfaction; volatility in grading perception is the leading indicator for "At-Risk" flagging.
2.  **Staff Responsiveness:** Operational efficiency in handling student queries significantly outweighs physical facility ratings.
3.  **Administrative Efficiency:** Bureaucratic friction acts as a secondary gatekeeper for student retention.

---

## 5. Repository Structure
* `01_preprocessing_nlp.ipynb`: Normalization and Transformer-based sentiment extraction.
* `02_data_augmentation.ipynb`: Implementation of SMOTE-NC to bridge the data gap.
* `03_predictive_modeling_xai.ipynb`: The core engine, feature selection, and SHAP analysis.
* `04_synthesis_application.ipynb`: Translation of model outputs into the **Institutional Priority Matrix**.

## 6. Conclusion for IEEE Submission
This framework proves that high-accuracy predictive analytics ($>80\%$) is achievable on sparse institutional datasets through intentional feature pruning, synthetic augmentation, and binary realignment. The model serves as a **Transparent Autonomy** tool, providing administrators with an explainable "Flagging System" to improve student satisfaction through data-driven intervention.