# Customer Churn Prediction using Apache Spark MLlib

## 📋 Project Overview

This project implements a **Customer Churn Prediction System** using Apache Spark and MLlib. It predicts whether a customer will leave (churn) a telecommunications service based on their usage patterns and demographics.

### Why This Project?

- **Real-world Application**: Customer retention is critical for businesses
- **Big Data Ready**: Uses Spark for scalability to millions of records
- **End-to-End Pipeline**: Covers data preprocessing, feature engineering, model training, and evaluation
- **Multiple Algorithms**: Compares Logistic Regression, Random Forest, and Gradient-Boosted Trees

## 🎯 Learning Objectives

- Master Apache Spark DataFrame operations
- Implement ML pipelines with Spark MLlib
- Handle imbalanced datasets
- Perform feature engineering and selection
- Compare multiple classification algorithms
- Evaluate models using industry-standard metrics

## 📊 Dataset

### Option 1: Kaggle Telco Customer Churn Dataset (Recommended)
- **Source**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Size**: 7,043 customers with 21 features
- **Download**: 
  1. Visit the Kaggle link
  2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
  3. Place it in the `data/` folder

### Option 2: IBM Sample Dataset
- **Source**: https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113
- Direct download link provided in the dataset documentation

### Option 3: Use Synthetic Data Generator
- Run `python generate_sample_data.py` to create a sample dataset
- This generates 10,000 synthetic customer records for testing

## 🏗️ Project Structure

```
spark_churn_prediction/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/
│   └── telco_churn.csv               # Dataset (you need to download)
├── src/
│   ├── 1_data_preprocessing.py       # Data loading and cleaning
│   ├── 2_feature_engineering.py      # Feature creation and transformation
│   ├── 3_model_training.py           # Train multiple ML models
│   ├── 4_model_evaluation.py         # Comprehensive evaluation
│   └── 5_full_pipeline.py            # Complete end-to-end pipeline
├── notebooks/
│   └── churn_analysis.ipynb          # Jupyter notebook version
├── models/
│   └── (trained models saved here)
├── outputs/
│   └── (evaluation results saved here)
└── utils/
    ├── config.py                      # Configuration settings
    ├── data_generator.py              # Synthetic data generator
    └── visualizations.py              # Result visualization tools
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Java 8 or 11 (required for Spark)
- 4GB+ RAM recommended

### Installation

1. **Clone or extract this project**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download dataset**:
   - Download from Kaggle (see Dataset section above)
   - Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the `data/` folder
   - Rename to `telco_churn.csv`

   OR generate sample data:
```bash
python utils/data_generator.py
```

4. **Verify Spark installation**:
```bash
python -c "from pyspark.sql import SparkSession; print('Spark OK')"
```

### Running the Project

#### Option A: Run Complete Pipeline (Recommended for Demo)
```bash
python src/5_full_pipeline.py
```

#### Option B: Run Step-by-Step
```bash
# Step 1: Data Preprocessing
python src/1_data_preprocessing.py

# Step 2: Feature Engineering
python src/2_feature_engineering.py

# Step 3: Model Training
python src/3_model_training.py

# Step 4: Model Evaluation
python src/4_model_evaluation.py
```

#### Option C: Use Jupyter Notebook
```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

## 📈 Expected Results

### Model Performance Metrics

Based on the Telco dataset, you should expect:

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~80% | ~65% | ~55% | ~60% | ~84% |
| Random Forest | ~82% | ~68% | ~58% | ~62% | ~86% |
| GBT Classifier | **~84%** | **~70%** | **~62%** | **~66%** | **~88%** |

### Key Insights

The models identify these top churn predictors:
1. **Contract Type**: Month-to-month contracts have higher churn
2. **Tenure**: Customers with shorter tenure are more likely to churn
3. **Monthly Charges**: Higher charges correlate with churn
4. **Internet Service**: Fiber optic customers churn more
5. **Payment Method**: Electronic check users churn more

## 🔍 Code Highlights

### 1. Data Preprocessing
- Handle missing values intelligently
- Convert categorical variables to numeric
- Detect and handle data type inconsistencies
- Create stratified train/test splits

### 2. Feature Engineering
- Create interaction features (e.g., tenure_per_charge)
- Bin continuous variables (tenure groups)
- One-hot encode categorical variables
- Scale numerical features using StandardScaler
- Handle class imbalance with class weights

### 3. Model Training
- Implement Spark ML Pipelines
- Use cross-validation for hyperparameter tuning
- Train multiple models in parallel
- Save best models for deployment

### 4. Model Evaluation
- Confusion matrices
- ROC curves and AUC scores
- Precision-Recall curves
- Feature importance analysis
- Business impact calculations (revenue saved)

## 💡 Future Improvements

### 1. Model Enhancements
- **Deep Learning**: Implement neural networks using Spark with TensorFlow
- **Ensemble Methods**: Stack multiple models for better performance
- **AutoML**: Use H2O.ai or MLflow for automated hyperparameter tuning
- **Time-Series Features**: Incorporate customer behavior trends over time

### 2. Feature Engineering
- **RFM Analysis**: Recency, Frequency, Monetary value features
- **Customer Segmentation**: K-means clustering for customer groups
- **Network Effects**: Social network analysis if customer referral data available
- **Text Analysis**: Sentiment analysis from customer service interactions

### 3. Production Deployment

#### Option A: Batch Prediction System
```python
# Deploy on Apache Spark cluster
spark-submit \
  --master spark://cluster:7077 \
  --deploy-mode cluster \
  --executor-memory 4G \
  --num-executors 10 \
  src/5_full_pipeline.py
```

#### Option B: Real-Time Prediction API
- Use Flask/FastAPI to create REST API
- Load trained Spark model
- Serve predictions in real-time
- Example endpoint: `POST /predict`

#### Option C: Cloud Deployment
- **AWS EMR**: Deploy on Elastic MapReduce
- **Azure Databricks**: Use managed Spark clusters
- **Google Dataproc**: Deploy on GCP
- **MLflow**: Track experiments and deploy models

### 4. Monitoring and Maintenance
- **Model Drift Detection**: Monitor prediction accuracy over time
- **A/B Testing**: Compare model versions in production
- **Automated Retraining**: Schedule periodic model updates
- **Alerting**: Set up alerts for performance degradation

### 5. Business Integration
- **Customer Dashboard**: Build visualization dashboards
- **Intervention System**: Automated retention campaigns for high-risk customers
- **ROI Calculator**: Measure retention campaign effectiveness
- **Explainable AI**: Use SHAP values to explain predictions to business users

## 🎓 Academic Considerations

### For Your Final Year Report

1. **Problem Statement**: Clearly define customer churn and its business impact
2. **Literature Review**: Review existing churn prediction approaches
3. **Methodology**: Explain Spark architecture and MLlib algorithms
4. **Implementation**: Detail your feature engineering decisions
5. **Results**: Compare models with statistical significance tests
6. **Discussion**: Analyze why certain features matter most
7. **Conclusion**: Summarize findings and business recommendations

### Presentation Tips
- Demonstrate scalability by showing time comparisons (small vs large datasets)
- Visualize feature importance to show business insights
- Calculate revenue impact (e.g., "Retaining 100 customers saves $X")
- Show ROC curves and confusion matrices
- Discuss ethical considerations (privacy, fairness)

## 🐛 Troubleshooting

### Common Issues

**1. "Java not found" error**
```bash
# Install Java 11
sudo apt-get install openjdk-11-jdk  # Ubuntu/Debian
brew install openjdk@11              # macOS

# Set JAVA_HOME
export JAVA_HOME=/path/to/java
```

**2. "Py4JJavaError" during Spark initialization**
- Check Java version: `java -version` (should be 8 or 11)
- Reduce memory if limited: Edit `utils/config.py` → reduce `spark.executor.memory`

**3. "Dataset not found" error**
- Ensure `telco_churn.csv` is in the `data/` folder
- Or run: `python utils/data_generator.py`

**4. "Out of Memory" error**
- Reduce dataset size for testing
- Increase Spark memory in `config.py`
- Close other applications

## 📚 Additional Resources

### Learning Spark
- [Official Spark Documentation](https://spark.apache.org/docs/latest/)
- [Spark MLlib Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [Learning Spark Book](https://www.oreilly.com/library/view/learning-spark-2nd/9781492050032/)

### Machine Learning
- [Hands-On Machine Learning](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

### Big Data
- [Hadoop: The Definitive Guide](https://www.oreilly.com/library/view/hadoop-the-definitive/9781491901687/)
- [Designing Data-Intensive Applications](https://www.oreilly.com/library/view/designing-data-intensive-applications/9781491903063/)

## 📝 License

This project is created for educational purposes. Feel free to use and modify for your academic work.

## 🤝 Contributing

This is an educational project. Suggestions for improvements are welcome!

## 📧 Support

If you encounter any issues:
1. Check the Troubleshooting section
2. Review the code comments
3. Consult Spark documentation
4. Search Stack Overflow for Spark-specific questions

## 🌟 Project Highlights for Resume

- Built end-to-end ML pipeline using Apache Spark and MLlib
- Processed and analyzed 7K+ customer records with distributed computing
- Implemented and compared 3 classification algorithms achieving 84% accuracy
- Engineered 15+ features including interaction terms and behavioral indicators
- Deployed scalable solution capable of handling millions of records
- Applied best practices: cross-validation, hyperparameter tuning, class balancing

---

**Good luck with your final year project! 🚀**

*Remember: The key to a successful project is not just running the code, but understanding WHY each decision was made and being able to explain it to your evaluators.*
