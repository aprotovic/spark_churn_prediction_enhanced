"""
STEP 5: FULL END-TO-END PIPELINE (Enhanced)
============================================
Runs all 4 steps with unified Spark session, structured error handling,
and a rich summary exported to JSON.

Author: Customer Churn Prediction Project
"""

import sys
import os
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession
from utils.config import *
from utils.spark import create_spark_session
from src import (
    data_preprocessing as step1,
    feature_engineering as step2,
    model_training as step3,
    model_evaluation as step4,
)


def run_full_pipeline():
    start = time.time()
    summary = {}
    print("=" * 80)
    print("  CUSTOMER CHURN PREDICTION  –  FULL PIPELINE")
    print("=" * 80)
    print(f"  Dataset : {DATASET_PATH}")
    print(f"  Models  : {', '.join(MODELS_TO_TRAIN)}")
    print(f"  Seed    : {RANDOM_SEED}")

    spark = create_spark_session()
    print(f"\n✓ Spark {spark.version} ready\n")

    try:
        # ── Step 1 ────────────────────────────────────────────────────────────
        t = time.time()
        print("─" * 80)
        print("  [1/4]  DATA PREPROCESSING")
        print("─" * 80)
        df = step1.load_data(spark, DATASET_PATH)
        step1.explore_data(df)
        step1.check_missing_values(df)
        df = step1.handle_missing_values(df)
        df = step1.remove_duplicates(df)
        df = step1.convert_data_types(df)
        step1.validate_data(df)
        cleaned_path = os.path.join(DATA_DIR, 'cleaned_data.csv')
        step1.save_cleaned_data(df, cleaned_path)
        summary['step1_time'] = round(time.time() - t, 2)
        summary['cleaned_records'] = df.count()
        print(f"\n  ✓ Step 1 done in {summary['step1_time']}s")

        # ── Step 2 ────────────────────────────────────────────────────────────
        t = time.time()
        print("\n" + "─" * 80)
        print("  [2/4]  FEATURE ENGINEERING")
        print("─" * 80)
        df = step2.load_cleaned_data(spark)
        df = step2.create_derived_features(df)
        cat_cols, num_cols = step2.identify_column_types(df)
        df, _ = step2.encode_categorical_features(df, cat_cols)
        df, _, feature_names = step2.assemble_features(df, cat_cols, num_cols)
        df, _ = step2.scale_features(df)
        df, _ = step2.encode_target(df)
        train_df, test_df = step2.split_data(df)
        step2.save_processed_data(train_df, test_df, feature_names)
        summary['step2_time'] = round(time.time() - t, 2)
        summary['train_records'] = train_df.count()
        summary['test_records']  = test_df.count()
        summary['num_features']  = len(feature_names)
        print(f"\n  ✓ Step 2 done in {summary['step2_time']}s")

        # ── Step 3 ────────────────────────────────────────────────────────────
        t = time.time()
        print("\n" + "─" * 80)
        print("  [3/4]  MODEL TRAINING")
        print("─" * 80)
        train_df2, test_df2 = step3.load_training_data(spark)
        trained = {}

        if 'logistic_regression' in MODELS_TO_TRAIN:
            trained['logistic_regression'] = step3.train_logistic_regression(train_df2, test_df2)
        if 'random_forest' in MODELS_TO_TRAIN:
            trained['random_forest'] = step3.train_random_forest(train_df2, test_df2)
        if 'gbt_classifier' in MODELS_TO_TRAIN:
            trained['gbt_classifier'] = step3.train_gbt_classifier(train_df2, test_df2)

        summary['step3_time'] = round(time.time() - t, 2)
        summary['models_trained'] = len(trained)
        print(f"\n  ✓ Step 3 done in {summary['step3_time']}s")

        # ── Step 4 ────────────────────────────────────────────────────────────
        t = time.time()
        print("\n" + "─" * 80)
        print("  [4/4]  MODEL EVALUATION")
        print("─" * 80)
        test_df3 = step4.load_test_data(spark)
        all_metrics = {}

        for name in MODELS_TO_TRAIN:
            model = trained.get(name)
            if model:
                preds = model.transform(test_df3)
                all_metrics[name] = step4.calculate_metrics(preds, name)
                step4.analyze_feature_importance(model, name)

        if len(all_metrics) > 1:
            step4.compare_models(all_metrics)

        step4.save_results(all_metrics)
        summary['step4_time'] = round(time.time() - t, 2)
        summary['all_metrics'] = all_metrics

        print(f"\n  ✓ Step 4 done in {summary['step4_time']}s")

    except Exception as e:
        print(f"\n✗ PIPELINE ERROR: {e}")
        import traceback; traceback.print_exc()
        summary['error'] = str(e)
    finally:
        spark.stop()
        print("\n✓ Spark session stopped")

    summary['total_time'] = round(time.time() - start, 2)
    return summary


def print_pipeline_summary(summary):
    print("\n" + "=" * 80)
    print("  PIPELINE SUMMARY")
    print("=" * 80)

    if 'error' in summary:
        print(f"\n  ✗ Failed: {summary['error']}")
        return

    print(f"\n  Timing:")
    for step, label in [
        ('step1_time', 'Data Preprocessing'),
        ('step2_time', 'Feature Engineering'),
        ('step3_time', 'Model Training'),
        ('step4_time', 'Model Evaluation'),
        ('total_time', 'TOTAL'),
    ]:
        print(f"    {label:<25} {summary.get(step,0):>7.2f}s")

    print(f"\n  Data:")
    print(f"    Cleaned records    : {summary.get('cleaned_records',0):,}")
    print(f"    Training records   : {summary.get('train_records',0):,}")
    print(f"    Testing records    : {summary.get('test_records',0):,}")
    print(f"    Features           : {summary.get('num_features',0)}")

    if 'all_metrics' in summary:
        print(f"\n  Model Performance:")
        for name, m in summary['all_metrics'].items():
            print(f"\n    {name.replace('_',' ').title()}")
            print(f"      Accuracy  {m['accuracy']:.4f}   F1 {m['f1']:.4f}   AUC-ROC {m['auc_roc']:.4f}")
            print(f"      ROI       {m['roi']['roi_percentage']:.1f}%")

    # Persist summary JSON
    out = os.path.join(OUTPUTS_DIR, 'pipeline_summary.json')
    serialisable = {k: v for k, v in summary.items() if k != 'all_metrics'}
    if 'all_metrics' in summary:
        serialisable['all_metrics'] = {
            mn: {k: (round(v,6) if isinstance(v,float) else v)
                 for k, v in mm.items() if k != 'confusion_matrix'}
            for mn, mm in summary['all_metrics'].items()
        }
    with open(out, 'w') as f:
        json.dump(serialisable, f, indent=2, default=str)
    print(f"\n  ✓ Summary saved to {out}")
    print("\n" + "=" * 80)
    print("  PIPELINE COMPLETE 🎉")
    print("=" * 80)


def main():
    if not os.path.exists(DATASET_PATH):
        print(f"\n✗ Dataset not found: {DATASET_PATH}")
        answer = input("Generate synthetic data? [yes/no]: ").strip().lower()
        if answer in ('yes', 'y'):
            from utils.data_generator import generate_churn_dataset, save_dataset
            print("Generating synthetic data (10,000 rows)...")
            df = generate_churn_dataset(n_samples=10000, churn_rate=0.27)
            save_dataset(df)
            print("✓ Synthetic data ready\n")
        else:
            sys.exit(1)

    summary = run_full_pipeline()
    print_pipeline_summary(summary)


if __name__ == '__main__':
    main()
