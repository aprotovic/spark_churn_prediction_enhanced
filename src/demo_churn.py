import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.functions import array_to_vector
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
from utils.config import *

def main():
    print("="*80)
    print("DEMO: SPARK CHURN PREDICTION + VISUALIZATION")
    print("="*80)

    # 1. Initialize Spark
    spark = SparkSession.builder \
        .appName("ChurnDemo") \
        .master("local[*]") \
        .config("spark.executor.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print("✓ Spark initialized")

    # 2. Load Data (Patched for Windows)
    print("\nLoading data...")
    train_path = os.path.join(DATA_DIR, 'train_data.parquet')
    test_path = os.path.join(DATA_DIR, 'test_data.parquet')

    if not os.path.exists(train_path):
        print(f"ERROR: Data not found at {train_path}")
        return

    # Load and convert Array->Vector
    train_df = spark.read.parquet(train_path).withColumn("features", array_to_vector("features"))
    test_df = spark.read.parquet(test_path).withColumn("features", array_to_vector("features"))
    
    print(f"✓ Data loaded (Train: {train_df.count()}, Test: {test_df.count()})")

    # 3. Train Model
    print("\nTraining Logistic Regression model...")
    lr = LogisticRegression(labelCol='label', featuresCol='features', maxIter=10)
    model = lr.fit(train_df)
    print("✓ Model trained")

    # 4. Predict
    print("\ngenerating predictions...")
    predictions = model.transform(test_df)
    
    # Extract probability for positive class
    # Probability is a vector [prob_0, prob_1], we need prob_1
    extract_prob = udf(lambda v: float(v[1]), DoubleType())
    preds_df = predictions.select('label', extract_prob('probability').alias('score')).toPandas()

    # 5. Visualize
    print("\nGenerating ROC Curve...")
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(preds_df['label'], preds_df['score'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Churn Prediction')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    output_plot = os.path.join(OUTPUTS_DIR, 'demo_roc_curve.png')
    plt.savefig(output_plot)
    print(f"✓ Plot saved to: {output_plot}")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    spark.stop()

if __name__ == "__main__":
    main()
