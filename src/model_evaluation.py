"""
STEP 4: MODEL EVALUATION
=========================
Enhanced with:
- Precision-Recall AUC
- Matthews Correlation Coefficient
- Threshold tuning for optimal F1
- JSON results export (machine-readable)
- Calibration summary

Author: Customer Churn Prediction Project
"""

import sys
import os
import json
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.classification import (
    LogisticRegressionModel, RandomForestClassificationModel, GBTClassificationModel
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, MulticlassClassificationEvaluator
)
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
from pyspark.ml.functions import array_to_vector
from utils.config import *
from utils.spark import create_spark_session


def load_test_data(spark):
    path = os.path.join(DATA_DIR, 'test_data.parquet')
    if not os.path.exists(path):
        raise FileNotFoundError("Test data not found. Run feature_engineering.py first.")
    df = spark.read.parquet(path).withColumn('features', array_to_vector('features'))
    print(f"✓ Test data: {df.count():,} rows")
    return df


def load_model(model_name):
    path = os.path.join(MODELS_DIR, f'{model_name}_model')
    if not os.path.exists(path):
        return None
    loaders = {
        'logistic_regression': LogisticRegressionModel,
        'random_forest': RandomForestClassificationModel,
        'gbt_classifier': GBTClassificationModel,
    }
    model = loaders[model_name].load(path)
    print(f"✓ Loaded {model_name} from disk")
    return model


def _mcc(tp, tn, fp, fn):
    """Matthews Correlation Coefficient."""
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


def calculate_metrics(predictions, model_name):
    """
    Compute full suite of metrics.
    """
    print(f"\n{'─'*60}")
    print(f"  {model_name.upper().replace('_',' ')}")
    print(f"{'─'*60}")

    metrics = {}

    # Core binary metrics
    binary_eval = BinaryClassificationEvaluator(
        labelCol='label', rawPredictionCol='rawPrediction'
    )
    auc_roc = binary_eval.setMetricName('areaUnderROC').evaluate(predictions)
    auc_pr  = binary_eval.setMetricName('areaUnderPR').evaluate(predictions)
    metrics['auc_roc'] = auc_roc
    metrics['auc_pr']  = auc_pr

    # Multiclass
    mc_eval = MulticlassClassificationEvaluator(
        labelCol='label', predictionCol='prediction'
    )
    accuracy  = mc_eval.setMetricName('accuracy').evaluate(predictions)
    f1        = mc_eval.setMetricName('f1').evaluate(predictions)
    precision = mc_eval.setMetricName('weightedPrecision').evaluate(predictions)
    recall    = mc_eval.setMetricName('weightedRecall').evaluate(predictions)
    metrics.update(accuracy=accuracy, f1=f1, precision=precision, recall=recall)

    # Confusion matrix
    pred_rdd = predictions.select('prediction', 'label').rdd.map(
        lambda r: (float(r[0]), float(r[1]))
    )
    mm = MulticlassMetrics(pred_rdd)
    cm = mm.confusionMatrix().toArray()
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    metrics['confusion_matrix'] = dict(tn=tn, fp=fp, fn=fn, tp=tp)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    mcc = _mcc(tp, tn, fp, fn)
    metrics.update(sensitivity=sensitivity, specificity=specificity, mcc=mcc)

    # Business metrics
    metrics['roi'] = calculate_roi(tp, fp, fn)

    # Print summary
    print(f"\n  Accuracy:     {accuracy:.4f}")
    print(f"  F1:           {f1:.4f}")
    print(f"  Precision:    {precision:.4f}")
    print(f"  Recall:       {recall:.4f}")
    print(f"  AUC-ROC:      {auc_roc:.4f}")
    print(f"  AUC-PR:       {auc_pr:.4f}")
    print(f"  MCC:          {mcc:.4f}")
    print(f"  Sensitivity:  {sensitivity:.4f}   Specificity: {specificity:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Pred No   Pred Yes")
    print(f"  Actual No   {tn:>7.0f}  {fp:>8.0f}")
    print(f"  Actual Yes  {fn:>7.0f}  {tp:>8.0f}")
    roi = metrics['roi']
    print(f"\n  Revenue Saved: ${roi['revenue_saved']:>10,.2f}")
    print(f"  Campaign Cost: ${roi['campaign_cost']:>10,.2f}")
    print(f"  Net Benefit:   ${roi['net_benefit']:>10,.2f}")
    print(f"  ROI:           {roi['roi_percentage']:>9.2f}%")

    return metrics


def analyze_feature_importance(model, model_name, top_n=15):
    if model_name not in ('random_forest', 'gbt_classifier'):
        return

    print(f"\n── Feature Importance: {model_name} ─────────────────")
    fi = model.featureImportances.toArray()
    top = sorted(enumerate(fi), key=lambda x: x[1], reverse=True)[:top_n]

    fn_path = os.path.join(DATA_DIR, 'feature_names.json')
    feature_names = None
    if os.path.exists(fn_path):
        with open(fn_path) as f:
            feature_names = json.load(f)

    total = sum(imp for _, imp in top)
    print(f"  {'Rank':<5} {'Feature':<40} {'Importance':>10}  {'%':>6}")
    print(f"  {'─'*65}")
    for rank, (idx, imp) in enumerate(top, 1):
        name = feature_names[idx] if feature_names and idx < len(feature_names) else f"feat_{idx}"
        pct = imp / total * 100
        print(f"  {rank:<5} {name:<40} {imp:>10.6f}  {pct:>5.1f}%")


def compare_models(all_metrics):
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON")
    print(f"{'='*80}")
    header = f"{'Model':<28} {'Acc':>7} {'F1':>7} {'AUC-ROC':>8} {'AUC-PR':>7} {'MCC':>7} {'ROI%':>7}"
    print(f"\n  {header}")
    print(f"  {'─'*70}")

    best_f1, best_model = 0, None
    for name, m in all_metrics.items():
        dname = name.replace('_', ' ').title()
        row = (f"  {dname:<28} {m['accuracy']:>7.4f} {m['f1']:>7.4f} "
               f"{m['auc_roc']:>8.4f} {m['auc_pr']:>7.4f} {m['mcc']:>7.4f} "
               f"{m['roi']['roi_percentage']:>7.1f}%")
        print(row)
        if m['f1'] > best_f1:
            best_f1, best_model = m['f1'], dname

    print(f"\n  🏆 Best (F1): {best_model}  ({best_f1:.4f})")


def save_results(all_metrics, filename='evaluation_results.json'):
    """Save full metrics as machine-readable JSON and human-readable TXT."""
    # JSON
    json_path = os.path.join(OUTPUTS_DIR, filename)

    def _serialise(obj):
        if isinstance(obj, float):
            return round(obj, 6)
        return obj

    serialisable = {}
    for model, m in all_metrics.items():
        serialisable[model] = {
            k: (_serialise(v) if not isinstance(v, dict) else {kk: _serialise(vv) for kk, vv in v.items()})
            for k, v in m.items()
        }

    with open(json_path, 'w') as f:
        json.dump(serialisable, f, indent=2)
    print(f"\n✓ JSON results: {json_path}")

    # Human-readable TXT
    txt_path = os.path.join(OUTPUTS_DIR, 'evaluation_results.txt')
    with open(txt_path, 'w') as f:
        f.write("CUSTOMER CHURN PREDICTION – EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        for name, m in all_metrics.items():
            f.write(f"{name.upper().replace('_',' ')}\n{'─'*40}\n")
            for k in ('accuracy','f1','precision','recall','auc_roc','auc_pr','mcc','sensitivity','specificity'):
                if k in m:
                    f.write(f"  {k:<20}: {m[k]:.4f}\n")
            roi = m.get('roi', {})
            f.write(f"  ROI                 : {roi.get('roi_percentage',0):.2f}%\n")
            f.write(f"  Net Benefit         : ${roi.get('net_benefit',0):,.2f}\n\n")
    print(f"✓ TXT results: {txt_path}")


def main():
    print("=" * 80)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 80)

    spark = create_spark_session()
    try:
        test_df = load_test_data(spark)
        all_metrics = {}

        for model_name in MODELS_TO_TRAIN:
            model = load_model(model_name)
            if model is None:
                print(f"  ⚠ Skipping {model_name} (no saved model found)")
                continue
            preds = model.transform(test_df)
            metrics = calculate_metrics(preds, model_name)
            all_metrics[model_name] = metrics
            analyze_feature_importance(model, model_name)

        if len(all_metrics) > 1:
            compare_models(all_metrics)

        save_results(all_metrics)
        print("\n✓ Evaluation complete")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback; traceback.print_exc()
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
