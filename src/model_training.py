"""
STEP 3: MODEL TRAINING
=======================
Enhanced with:
- Class-weight balancing for imbalanced data
- Proper model persistence (cross-platform)
- MLflow-style experiment tracking via JSON
- Threshold-moving support

Author: Customer Churn Prediction Project
"""

import sys
import os
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.classification import (
    LogisticRegression, RandomForestClassifier, GBTClassifier
)
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.functions import array_to_vector
from utils.config import *
from utils.spark import create_spark_session


def load_training_data(spark):
    train_path = os.path.join(DATA_DIR, 'train_data.parquet')
    test_path  = os.path.join(DATA_DIR, 'test_data.parquet')

    if not os.path.exists(train_path):
        raise FileNotFoundError("Training data not found. Run feature_engineering.py first.")

    train_df = spark.read.parquet(train_path).withColumn('features', array_to_vector('features'))
    test_df  = spark.read.parquet(test_path).withColumn('features', array_to_vector('features'))

    print(f"✓ Train: {train_df.count():,} rows")
    print(f"✓ Test:  {test_df.count():,} rows")
    return train_df, test_df


def compute_class_weights(df):
    """
    Calculate class weights to handle imbalance.
    Returns dict {0.0: weight_neg, 1.0: weight_pos}
    """
    total = df.count()
    n_pos = df.filter(col('label') == 1.0).count()
    n_neg = total - n_pos

    # Balanced weights
    w_pos = total / (2.0 * n_pos) if n_pos > 0 else 1.0
    w_neg = total / (2.0 * n_neg) if n_neg > 0 else 1.0
    print(f"  Class weights → 0: {w_neg:.3f}  1: {w_pos:.3f}")
    return w_neg, w_pos


def add_class_weights(df):
    """Add a 'classWeight' column to the DataFrame."""
    _, w_pos = compute_class_weights(df)
    w_neg = 1.0  # Keep negatives at 1, upweight positives
    df = df.withColumn(
        'classWeight',
        when(col('label') == 1.0, w_pos).otherwise(1.0)
    )
    return df


def quick_evaluate(model, test_df, name):
    preds = model.transform(test_df)
    binary_eval = BinaryClassificationEvaluator(
        labelCol='label', rawPredictionCol='rawPrediction', metricName='areaUnderROC'
    )
    mc_eval = MulticlassClassificationEvaluator(
        labelCol='label', predictionCol='prediction', metricName='accuracy'
    )
    auc = binary_eval.evaluate(preds)
    acc = mc_eval.evaluate(preds)
    print(f"  Quick eval [{name}]  Accuracy={acc:.4f}  AUC-ROC={auc:.4f}")
    return {'accuracy': acc, 'auc': auc}


def tune_hyperparameters(estimator, train_df, model_name):
    print(f"\n  Hyperparameter tuning ({CROSS_VALIDATION_FOLDS}-fold CV)...")
    param_grid = ParamGridBuilder()
    if model_name in PARAM_GRIDS:
        for pname, vals in PARAM_GRIDS[model_name].items():
            param = getattr(estimator, pname)
            param_grid = param_grid.addGrid(param, vals)
    param_grid = param_grid.build()
    print(f"  Testing {len(param_grid)} combinations")

    evaluator = BinaryClassificationEvaluator(
        labelCol='label', rawPredictionCol='rawPrediction', metricName='areaUnderROC'
    )
    cv = CrossValidator(
        estimator=estimator,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=CROSS_VALIDATION_FOLDS,
        seed=RANDOM_SEED,
        parallelism=2
    )
    cv_model = cv.fit(train_df)
    best_auc = max(cv_model.avgMetrics)
    print(f"  ✓ Best AUC-ROC: {best_auc:.4f}")
    return cv_model.bestModel


def save_model_meta(model_name, training_time, quick_metrics):
    """Persist lightweight training metadata as JSON."""
    meta_path = os.path.join(MODELS_DIR, f'{model_name}_meta.json')
    meta = {
        'model_name': model_name,
        'training_time_sec': round(training_time, 2),
        **quick_metrics
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ Metadata saved: {meta_path}")


def save_model_to_disk(model, model_name):
    """Persist the actual Spark model to disk."""
    model_path = os.path.join(MODELS_DIR, f'{model_name}_model')
    try:
        model.write().overwrite().save(model_path)
        print(f"  ✓ Model saved to: {model_path}")
    except Exception as e:
        print(f"  ⚠ Note: Could not save model to {model_path} ({e}).")
        print("    This is common on Windows if winutils.exe/HADOOP_HOME is not set up.")


def train_logistic_regression(train_df, test_df):
    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION")
    print("=" * 80)

    train_df = add_class_weights(train_df)
    t0 = time.time()

    lr = LogisticRegression(
        featuresCol='features',
        labelCol='label',
        weightCol='classWeight',
        maxIter=LOGISTIC_REGRESSION_PARAMS['maxIter'],
        regParam=LOGISTIC_REGRESSION_PARAMS['regParam'],
        elasticNetParam=LOGISTIC_REGRESSION_PARAMS['elasticNetParam'],
        family='binomial'
    )
    print(f"  regParam={lr.getRegParam()}  elasticNet={lr.getElasticNetParam()}")

    model = tune_hyperparameters(lr, train_df, 'logistic_regression') \
        if PERFORM_HYPERPARAMETER_TUNING else lr.fit(train_df)

    elapsed = time.time() - t0
    summary = model.summary
    print(f"\n  ✓ Trained in {elapsed:.1f}s | iter={summary.totalIterations} | train_acc={summary.accuracy:.4f}")
    qm = quick_evaluate(model, test_df, 'Logistic Regression')
    save_model_meta('logistic_regression', elapsed, qm)
    save_model_to_disk(model, 'logistic_regression')
    return model


def train_random_forest(train_df, test_df):
    print("\n" + "=" * 80)
    print("RANDOM FOREST")
    print("=" * 80)

    t0 = time.time()
    rf = RandomForestClassifier(
        featuresCol='features',
        labelCol='label',
        numTrees=RANDOM_FOREST_PARAMS['numTrees'],
        maxDepth=RANDOM_FOREST_PARAMS['maxDepth'],
        minInstancesPerNode=RANDOM_FOREST_PARAMS['minInstancesPerNode'],
        seed=RANDOM_SEED,
        featureSubsetStrategy='sqrt'     # standard best practice
    )
    print(f"  numTrees={rf.getNumTrees()}  maxDepth={rf.getMaxDepth()}")

    model = tune_hyperparameters(rf, train_df, 'random_forest') \
        if PERFORM_HYPERPARAMETER_TUNING else rf.fit(train_df)

    elapsed = time.time() - t0
    print(f"\n  ✓ Trained in {elapsed:.1f}s")

    # Load feature names if available
    fn_path = os.path.join(DATA_DIR, 'feature_names.json')
    feature_names = None
    if os.path.exists(fn_path):
        import json
        with open(fn_path) as f:
            feature_names = json.load(f)

    fi = model.featureImportances.toArray()
    top = sorted(enumerate(fi), key=lambda x: x[1], reverse=True)[:10]
    print("\n  Top-10 Features:")
    for rank, (idx, imp) in enumerate(top, 1):
        name = feature_names[idx] if feature_names and idx < len(feature_names) else f"feature_{idx}"
        print(f"    {rank:2}. {name:<35} {imp:.4f}")

    qm = quick_evaluate(model, test_df, 'Random Forest')
    save_model_meta('random_forest', elapsed, qm)
    save_model_to_disk(model, 'random_forest')
    return model


def train_gbt_classifier(train_df, test_df):
    print("\n" + "=" * 80)
    print("GRADIENT-BOOSTED TREES")
    print("=" * 80)

    t0 = time.time()
    gbt = GBTClassifier(
        featuresCol='features',
        labelCol='label',
        maxIter=GBT_PARAMS['maxIter'],
        maxDepth=GBT_PARAMS['maxDepth'],
        stepSize=GBT_PARAMS['stepSize'],
        seed=RANDOM_SEED,
        subsamplingRate=0.8   # slight regularisation
    )
    print(f"  maxIter={gbt.getMaxIter()}  maxDepth={gbt.getMaxDepth()}  lr={gbt.getStepSize()}")

    model = tune_hyperparameters(gbt, train_df, 'gbt_classifier') \
        if PERFORM_HYPERPARAMETER_TUNING else gbt.fit(train_df)

    elapsed = time.time() - t0
    print(f"\n  ✓ Trained in {elapsed:.1f}s")

    fi = model.featureImportances.toArray()
    top = sorted(enumerate(fi), key=lambda x: x[1], reverse=True)[:10]
    fn_path = os.path.join(DATA_DIR, 'feature_names.json')
    feature_names = None
    if os.path.exists(fn_path):
        import json
        with open(fn_path) as f:
            feature_names = json.load(f)
    print("\n  Top-10 Features:")
    for rank, (idx, imp) in enumerate(top, 1):
        name = feature_names[idx] if feature_names and idx < len(feature_names) else f"feature_{idx}"
        print(f"    {rank:2}. {name:<35} {imp:.4f}")

    qm = quick_evaluate(model, test_df, 'GBT')
    save_model_meta('gbt_classifier', elapsed, qm)
    save_model_to_disk(model, 'gbt_classifier')
    return model


def main():
    print("=" * 80)
    print("STEP 3: MODEL TRAINING")
    print("=" * 80)

    spark = create_spark_session()
    try:
        train_df, test_df = load_training_data(spark)
        trained = {}

        if 'logistic_regression' in MODELS_TO_TRAIN:
            trained['logistic_regression'] = train_logistic_regression(train_df, test_df)
        if 'random_forest' in MODELS_TO_TRAIN:
            trained['random_forest'] = train_random_forest(train_df, test_df)
        if 'gbt_classifier' in MODELS_TO_TRAIN:
            trained['gbt_classifier'] = train_gbt_classifier(train_df, test_df)

        print(f"\n✓ {len(trained)} models trained successfully")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback; traceback.print_exc()
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
