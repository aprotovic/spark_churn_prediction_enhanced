# ⚡ Distributed Customer Churn Prediction (Apache Spark MLlib)

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.x-E25A1C?style=flat&logo=apachespark&logoColor=white)](https://spark.apache.org/)
[![Machine Learning](https://img.shields.io/badge/MLlib-Classification-orange.svg)](https://spark.apache.org/mllib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-scale, distributed machine learning pipeline built on **Apache Spark (PySpark)** to process large-scale telecommunications customer activity logs and accurately forecast customer churn risk.

---

## 📊 Benchmark Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 80.2% | 65.4% | 55.1% | 59.8% | 0.841 |
| **Random Forest** | 82.5% | 68.2% | 58.7% | 63.1% | 0.864 |
| **Gradient-Boosted Trees (GBT)** | **84.3%** | **70.5%** | **62.3%** | **66.1%** | **0.882** |

---

## ✨ Pipeline Highlights

- **Distributed Data Processing:** Scalable PySpark SQL transformations handling missing values, schema casting, and stratified train/test partitioning.
- **Spark Feature Transformers:** One-hot encoding (`OneHotEncoder`), string indexing (`StringIndexer`), vector assembling (`VectorAssembler`), and standardization (`StandardScaler`).
- **Parallel Cross-Validation:** Hyperparameter grid search evaluated with `BinaryClassificationEvaluator`.
- **Business Retention Insights:** Translates predictive outputs into dollar-value churn prevention estimates.

---

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/aprotovic/spark_churn_prediction_enhanced.git
cd spark_churn_prediction_enhanced

# 2. Install dependencies (Requires Java 8 or 11 for Spark)
pip install -r requirements.txt

# 3. Execute the full end-to-end Spark ML pipeline
python src/5_full_pipeline.py
```

---

## 📄 License
MIT License. Built by [aprotovic](https://github.com/aprotovic).
