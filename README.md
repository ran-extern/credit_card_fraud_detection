# Credit Card Fraud Detection

This project aims to predict fraudulent use of credit card based on the usage pattern of the usage pattern of the credit card,understand the factors contributing to fraudulent activities, and explore effective methods for detection and prevention using *binary classification*.

---
## Problem Statement

- Fraud detection using credit card usage pattern.
- Handling of highly imbalanced dataset, typically 1% fraud case with 99% legitimate case.
- Maximizing F1 score.
---

## Project Structure


```plaintext
credit_card_fraud_detection/
├── assets/                          # Visualizations, charts, exported models
│   └── figures/
├── data/                            # Raw and processed data
│   ├── processed_data/
│   │   └── credit_card_fraud_processed_dataset.csv
│   └── raw_data/
│       └── credit_card_fraud_dataset.csv
├── src/                       # Step-by-step development notebooks
│   ├── preprocessing.ipynb
│   └──  model_train.ipynb
│   
├── README.md                        # Project documentation
├── LICENSE                          # MIT License
└── requirements.txt                 # Project dependencies
```

---

## Dataset Overview

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/bhadramohit/credit-card-fraud-detection)
- **Total Rows**: ~2,240
- **Columns**:
  - Demographics: `Age`, `Income`, `Education`, `Marital_Status`, `Kidhome`, `Teenhome`
  - Campaign History: `Response`, `AcceptedCmp1` to `AcceptedCmp5`, `Complain`, `Recency`
  - Purchasing Behavior: `MntWines`, `MntMeatProducts`, `MntFruits`, `MntFishProducts`, `MntGoldProds`, `MntSweetProducts`
  - Temporal: `Dt_Customer` (date joined)
  - Others: `NumDealsPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`

---

## Notebook Walkthrough

### 01_data_loading_and_exploration.ipynb
- Load and inspect the raw dataset
- Identify missing values, data types, and categorical distributions

### 02_preprocessing_and_feature_engineering.ipynb
- Handle nulls, convert dates, clean categorical values
- Create age group, year joined, log-transform skewed features

### 03_eda_and_visualization.ipynb
- Visualize purchase behavior, demographics
- Correlation analysis, target variable distribution

### 04_classification_models.ipynb
- Predict `Response` using:
  - Logistic Regression
  - Random Forest
  - Decision Trees
- Evaluate with accuracy, precision, recall, F1-score

### 05_hyperparameter_tuning_with_gridsearchcv.ipynb
- Use `GridSearchCV` to tune model hyperparameters
- Visualize validation results

### 06_clustering_with_kmeans_and_hierarchical.ipynb
- KMeans and Hierarchical clustering
- Use `Elbow Method`, `Silhouette Score` for optimal k
- Visualize clusters with PCA or t-SNE

### 07_model_evaluation_and_summary.ipynb
- ROC curves, confusion matrix, final metrics
- Summary of best model and cluster characteristics

### 08_final_insights_and_export.ipynb
- Save models using `joblib` or `pickle`
- Export CSV with cluster labels
- Final strategic recommendations

---

## Key Skills Demonstrated

- Feature Engineering & Encoding
- Data Cleaning & Transformation
- Model Building (Classification & Clustering)
- Model Evaluation & Optimization
- Data Visualization and Insights
- End-to-End Machine Learning Pipeline Structuring

---

## Tools & Libraries Used

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`: models, preprocessing, metrics
- `joblib` for model serialization
- `jupyter`, `notebook`, `openpyxl`

---

## Setup Instructions

```bash
git clone https://gitlab.com/<your-username>/customer_segmentation_project.git
cd customer_segmentation_project
pip install -r requirements.txt
```

---

## Author

**Name:** Md Farmanul Haque  
**GitLab:** [https://gitlab.com/Md-Farmanul-Haque](https://gitlab.com/Md-Farmanul-Haque)

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for full details.
