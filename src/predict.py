"""
PREDICT: Single-customer & batch churn prediction CLI
=======================================================
Usage:
  # Single customer (interactive prompts)
  python src/predict.py

  # Batch CSV
  python src/predict.py --batch path/to/customers.csv

Author: Customer Churn Prediction Project
"""

import sys
import os
import argparse
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import *
from utils.spark import create_spark_session


def batch_predict(spark, batch_csv_path):
    """Run batch prediction on a CSV file and output results to outputs/batch_predictions.csv"""
    best_model_name, auc = load_best_model_meta()
    if best_model_name is None:
        print("✗ No trained model found. Run the full pipeline first.")
        return

    print(f"\nUsing best model: {best_model_name}  (AUC-ROC={auc:.4f})")

    if not os.path.exists(batch_csv_path):
        print(f"✗ Batch file not found: {batch_csv_path}")
        return

    print(f"\nLoading batch data from: {batch_csv_path}...")
    df = spark.read.csv(batch_csv_path, header=True, inferSchema=True)
    n_records = df.count()
    print(f"Loaded {n_records:,} customer records.")

    # Apply data preprocessing steps
    from src import data_preprocessing as dp
    df = dp.handle_missing_values(df)
    df = dp.convert_data_types(df)

    # Feature engineering + prediction
    from src import feature_engineering as fe
    from pyspark.ml.functions import array_to_vector
    df = fe.create_derived_features(df)
    cat_cols, num_cols = fe.identify_column_types(df)
    df, _ = fe.encode_categorical_features(df, cat_cols)
    df, _, _ = fe.assemble_features(df, cat_cols, num_cols)
    df, _ = fe.scale_features(df)
    df, _ = fe.encode_target(df)
    df = df.withColumn('features', array_to_vector('features'))

    # Load and run model
    from pyspark.ml.classification import (
        LogisticRegressionModel, RandomForestClassificationModel, GBTClassificationModel
    )
    loaders = {
        'logistic_regression': LogisticRegressionModel,
        'random_forest'      : RandomForestClassificationModel,
        'gbt_classifier'     : GBTClassificationModel,
    }
    model_path = os.path.join(MODELS_DIR, f'{best_model_name}_model')
    if not os.path.exists(model_path):
        print(f"✗ Model file not found at {model_path}. Save models to disk first.")
        return

    model = loaders[best_model_name].load(model_path)
    preds = model.transform(df)

    # Extract probability for positive class
    from pyspark.sql.functions import udf, col
    from pyspark.sql.types import DoubleType
    extract_prob = udf(lambda v: float(v[1]), DoubleType())

    # Select original customer ID (if exists), probability, and prediction
    select_cols = []
    if 'customerID' in preds.columns:
        select_cols.append('customerID')
    if TARGET_COLUMN in preds.columns:
        select_cols.append(TARGET_COLUMN)

    results_df = preds.select(
        *select_cols,
        extract_prob('probability').alias('ChurnProbability'),
        col('prediction').alias('PredictedChurn')
    )

    out_dir = os.path.join(OUTPUTS_DIR, 'batch_predictions_dir')
    out_csv = os.path.join(OUTPUTS_DIR, 'batch_predictions.csv')

    # Coalesce to 1 partition and write to CSV
    results_df.coalesce(1).write.csv(out_dir, header=True, mode='overwrite')

    # Rename/move single partition file to target CSV path
    import glob
    import shutil
    parts = glob.glob(os.path.join(out_dir, 'part-*.csv'))
    if parts:
        if os.path.exists(out_csv):
            os.remove(out_csv)
        shutil.copy(parts[0], out_csv)
        shutil.rmtree(out_dir)
        print(f"\n✓ Batch predictions saved to: {out_csv}")
    else:
        print("✗ Failed to save batch prediction output.")


def load_best_model_meta():
    """Return the model name with the highest AUC-ROC from saved metadata."""
    best_name, best_auc = None, -1
    for name in MODELS_TO_TRAIN:
        meta_path = os.path.join(MODELS_DIR, f'{name}_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                m = json.load(f)
            if m.get('auc', 0) > best_auc:
                best_auc = m['auc']
                best_name = name
    return best_name, best_auc


def get_churn_risk_label(probability):
    """Convert churn probability to human-readable risk label."""
    if probability >= 0.7:
        return "🔴 HIGH RISK"
    elif probability >= 0.4:
        return "🟡 MEDIUM RISK"
    else:
        return "🟢 LOW RISK"


def interactive_predict(spark):
    """Walk through an interactive single-customer prediction session."""
    from pyspark.ml.functions import array_to_vector

    best_model_name, auc = load_best_model_meta()
    if best_model_name is None:
        print("✗ No trained model found. Run the full pipeline first.")
        return

    print(f"\nUsing best model: {best_model_name}  (AUC-ROC={auc:.4f})")

    # Gather inputs
    print("\n── Enter Customer Details ───────────────────────")
    try:
        tenure          = int(input("  Tenure (months): "))
        monthly_charges = float(input("  Monthly Charges ($): "))
        total_charges   = float(input("  Total Charges ($, or press Enter to estimate): ") or tenure * monthly_charges)
        contract        = input("  Contract [Month-to-month / One year / Two year]: ").strip() or "Month-to-month"
        internet        = input("  Internet Service [DSL / Fiber optic / No]: ").strip() or "DSL"
        senior          = int(input("  Senior Citizen? [0/1]: ") or 0)
    except (ValueError, KeyboardInterrupt):
        print("\n  Cancelled.")
        return

    # Build a minimal Spark row
    from pyspark.sql import Row
    row = Row(
        tenure=tenure,
        MonthlyCharges=monthly_charges,
        TotalCharges=total_charges,
        Contract=contract,
        InternetService=internet,
        SeniorCitizen=senior,
        Churn='No',   # placeholder
        customerID='DEMO-001',
        gender='Male',
        Partner='No', Dependents='No',
        PhoneService='Yes', MultipleLines='No',
        OnlineSecurity='No', OnlineBackup='No',
        DeviceProtection='No', TechSupport='No',
        StreamingTV='No', StreamingMovies='No',
        PaperlessBilling='Yes', PaymentMethod='Electronic check',
    )
    df = spark.createDataFrame([row])

    # Feature engineering + prediction (import inline to avoid circular deps)
    from src import feature_engineering as fe
    df = fe.create_derived_features(df)
    cat_cols, num_cols = fe.identify_column_types(df)
    df, _ = fe.encode_categorical_features(df, cat_cols)
    df, _, _ = fe.assemble_features(df, cat_cols, num_cols)
    df, _ = fe.scale_features(df)
    df, _ = fe.encode_target(df)
    df = df.withColumn('features', array_to_vector('features'))

    # Load and run model
    from pyspark.ml.classification import (
        LogisticRegressionModel, RandomForestClassificationModel, GBTClassificationModel
    )
    loaders = {
        'logistic_regression': LogisticRegressionModel,
        'random_forest'      : RandomForestClassificationModel,
        'gbt_classifier'     : GBTClassificationModel,
    }
    model_path = os.path.join(MODELS_DIR, f'{best_model_name}_model')
    if not os.path.exists(model_path):
        print(f"✗ Model file not found at {model_path}. Save models to disk first.")
        return

    model = loaders[best_model_name].load(model_path)
    preds = model.transform(df)
    row_result = preds.select('prediction', 'probability').collect()[0]

    prob_churn = float(row_result['probability'][1])
    predicted  = int(row_result['prediction'])

    print(f"\n{'═'*50}")
    print(f"  CHURN PREDICTION RESULT")
    print(f"{'═'*50}")
    print(f"  Churn Probability : {prob_churn:.1%}")
    print(f"  Risk Level        : {get_churn_risk_label(prob_churn)}")
    print(f"  Prediction        : {'Will CHURN' if predicted == 1 else 'Will NOT churn'}")
    print(f"{'═'*50}")

    if prob_churn >= 0.5:
        print("\n  Recommended Actions:")
        print("  • Offer loyalty discount or contract upgrade")
        print("  • Assign dedicated customer success manager")
        print("  • Proactively resolve any open support tickets")


def main():
    parser = argparse.ArgumentParser(description='Churn Prediction Inference')
    parser.add_argument('--batch', type=str, default=None, help='Path to batch CSV file')
    args = parser.parse_args()

    spark = create_spark_session()

    try:
        if args.batch:
            batch_predict(spark, args.batch)
        else:
            interactive_predict(spark)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
