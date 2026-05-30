# 🚀 Quick Start Guide

This is a 5-minute guide to get you started with the Customer Churn Prediction project.

## ⚡ Fast Setup

### 1. Install Dependencies (2 minutes)

```bash
# Install Python packages
pip install -r requirements.txt

# Verify Java is installed (required for Spark)
java -version

# If Java is not installed:
# Ubuntu/Debian: sudo apt-get install openjdk-11-jdk
# macOS: brew install openjdk@11
# Windows: Download from https://adoptium.net/
```

### 2. Get Dataset (1 minute)

**Option A: Generate Sample Data (Fastest)**
```bash
python utils/data_generator.py
```

**Option B: Download Real Data**
- Visit: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Save to `data/telco_churn.csv`

### 3. Run the Project (2 minutes)

```bash
# Run the complete pipeline
python src/5_full_pipeline.py
```

That's it! The pipeline will:
1. ✅ Preprocess data
2. ✅ Engineer features
3. ✅ Train 3 ML models
4. ✅ Evaluate performance
5. ✅ Generate results

## 📊 Expected Output

You'll see:
- Real-time progress updates
- Model training metrics
- Confusion matrices
- ROC-AUC scores
- Business impact analysis (ROI)

Results are saved to:
- `models/` - Trained models
- `outputs/` - Evaluation reports

## 🎯 What You Get

### Performance Metrics
- **Accuracy**: ~80-84%
- **AUC-ROC**: ~84-88%
- **F1-Score**: ~60-66%

### Business Insights
- Top churn predictors
- Revenue impact calculation
- Customer retention recommendations

## 📝 For Your Report

Key points to include:
1. **Problem**: Customer churn costs businesses millions
2. **Solution**: ML-based prediction using Apache Spark
3. **Methods**: Logistic Regression, Random Forest, GBT
4. **Results**: 84% accuracy, identified top 5 churn factors
5. **Impact**: Can save $X per year in retention costs

## 🐛 Troubleshooting

**Error: "Java not found"**
```bash
# Install Java 11
sudo apt-get install openjdk-11-jdk  # Linux
brew install openjdk@11              # macOS
```

**Error: "Dataset not found"**
```bash
# Generate synthetic data
python utils/data_generator.py
```

**Error: "Out of memory"**
- Edit `utils/config.py`
- Reduce `spark.executor.memory` to `2g`

## 💡 Next Steps

1. **Understand the Code**: Read through each module with comments
2. **Experiment**: Try different hyperparameters in `config.py`
3. **Visualize**: Create charts from the evaluation results
4. **Deploy**: Consider deploying the best model

## 🎓 Academic Tips

For your final year presentation:
1. **Demo Live**: Run the pipeline during presentation
2. **Show ROC Curve**: Visual impact is powerful
3. **Explain Features**: Which factors matter most for churn?
4. **Business Value**: Translate accuracy to dollar savings
5. **Future Work**: Deep learning, real-time predictions, etc.

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Setup | 2-3 minutes |
| Data preparation | 1 minute |
| Full pipeline execution | 2-5 minutes |
| Total | **5-10 minutes** |

*Times are for the sample 10K dataset. Real dataset may take longer.*

## 📚 Learn More

See the main `README.md` for:
- Detailed explanations
- Advanced features
- Deployment options
- Future improvements

---

**Good luck with your project! 🎓✨**
