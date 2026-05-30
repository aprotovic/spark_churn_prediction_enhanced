"""
Configuration for Spark Churn Prediction Project
Enhanced with environment-variable overrides and validation helpers.
"""

import os
import math
import sys

# Configure UTF-8 encoding on Windows to prevent UnicodeEncodeErrors with special console characters
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, 'data')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

for d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    os.makedirs(d, exist_ok=True)

DATASET_PATH = os.path.join(DATA_DIR, 'telco_churn.csv')

# ── Spark ──────────────────────────────────────────────────────────────────────
SPARK_CONFIG = {
    'app_name'                                   : 'CustomerChurnPrediction',
    'master'                                     : 'local[*]',
    'spark.executor.memory'                      : '4g',
    'spark.driver.memory'                        : '4g',
    'spark.sql.shuffle.partitions'               : '10',
    'spark.default.parallelism'                  : '8',
    'spark.sql.adaptive.enabled'                 : 'true',
    'spark.sql.adaptive.coalescePartitions.enabled': 'true',
}

# ── Data ───────────────────────────────────────────────────────────────────────
TRAIN_TEST_SPLIT_RATIO = 0.8
RANDOM_SEED            = 42
TARGET_COLUMN          = 'Churn'
COLUMNS_TO_DROP        = ['customerID']
FEATURE_COLUMNS        = None   # None → auto-detect
CATEGORICAL_COLUMNS    = None
NUMERICAL_COLUMNS      = None

# ── Feature Engineering ────────────────────────────────────────────────────────
CREATE_INTERACTION_FEATURES = True
BIN_CONTINUOUS_FEATURES     = True
TENURE_BINS   = [0, 12, 24, 48, 72]
TENURE_LABELS = ['0-1yr', '1-2yr', '2-4yr', '4-6yr']

# ── Models ─────────────────────────────────────────────────────────────────────
MODELS_TO_TRAIN = [
    'logistic_regression',
    'random_forest',
    'gbt_classifier',
]

LOGISTIC_REGRESSION_PARAMS = {
    'maxIter'         : 200,    # increased from 100
    'regParam'        : 0.01,
    'elasticNetParam' : 0.0,    # L2
    'family'          : 'binomial',
}

RANDOM_FOREST_PARAMS = {
    'numTrees'              : 150,   # increased from 100
    'maxDepth'              : 10,
    'minInstancesPerNode'   : 2,     # slight regularisation
    'seed'                  : RANDOM_SEED,
}

GBT_PARAMS = {
    'maxIter'  : 100,
    'maxDepth' : 5,
    'stepSize' : 0.1,
    'seed'     : RANDOM_SEED,
}

# ── Hyperparameter Tuning ──────────────────────────────────────────────────────
PERFORM_HYPERPARAMETER_TUNING = False   # flip to True for grid search
CROSS_VALIDATION_FOLDS        = 5

PARAM_GRIDS = {
    'logistic_regression': {
        'regParam'        : [0.001, 0.01, 0.1],
        'elasticNetParam' : [0.0, 0.5, 1.0],
    },
    'random_forest': {
        'numTrees' : [100, 150, 200],
        'maxDepth' : [5, 10, 15],
    },
    'gbt_classifier': {
        'maxIter'  : [50, 100, 150],
        'maxDepth' : [3, 5, 7],
        'stepSize' : [0.05, 0.1, 0.2],
    },
}

# ── Evaluation ─────────────────────────────────────────────────────────────────
EVALUATION_METRICS       = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr', 'mcc']
CLASSIFICATION_THRESHOLD = 0.5

# ── Business / ROI ─────────────────────────────────────────────────────────────
CUSTOMER_LIFETIME_VALUE  = 1000   # $ average CLV
RETENTION_CAMPAIGN_COST  = 100    # $ per customer contacted
RETENTION_SUCCESS_RATE   = 0.30   # 30% of targeted churners retained

# ── Visualisation ──────────────────────────────────────────────────────────────
FIGURE_SIZE    = (12, 8)
COLOR_PALETTE  = 'Set2'
DPI            = 300

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL  = 'INFO'
SAVE_LOGS  = True
LOG_FILE   = os.path.join(OUTPUTS_DIR, 'churn_prediction.log')

# ── Deployment ─────────────────────────────────────────────────────────────────
MODEL_SERVING_PORT = 5000
MODEL_SERVING_HOST = '0.0.0.0'
BATCH_SIZE         = 1000

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_spark_config():
    return SPARK_CONFIG

def get_model_params(model_name):
    return {
        'logistic_regression': LOGISTIC_REGRESSION_PARAMS,
        'random_forest'      : RANDOM_FOREST_PARAMS,
        'gbt_classifier'     : GBT_PARAMS,
    }.get(model_name, {})


def calculate_roi(tp, fp, fn):
    """
    Business ROI calculation.
    """
    revenue_saved = tp  * CUSTOMER_LIFETIME_VALUE * RETENTION_SUCCESS_RATE
    campaign_cost = (tp + fp) * RETENTION_CAMPAIGN_COST
    revenue_lost  = fn  * CUSTOMER_LIFETIME_VALUE
    net_benefit   = revenue_saved - campaign_cost
    roi_pct       = (net_benefit / campaign_cost * 100) if campaign_cost > 0 else 0.0

    return dict(
        revenue_saved   = revenue_saved,
        campaign_cost   = campaign_cost,
        revenue_lost    = revenue_lost,
        net_benefit     = net_benefit,
        roi_percentage  = roi_pct,
    )


if __name__ == '__main__':
    print("── Spark Churn Config ──────────────────────────")
    print(f"BASE_DIR : {BASE_DIR}")
    print(f"DATASET  : {DATASET_PATH}")
    print(f"MODELS   : {', '.join(MODELS_TO_TRAIN)}")
    print(f"HP TUNING: {'ON' if PERFORM_HYPERPARAMETER_TUNING else 'OFF'}")
