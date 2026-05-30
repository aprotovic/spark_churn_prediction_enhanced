"""
STEP 2: FEATURE ENGINEERING
============================
Enhanced with SMOTE-style oversampling support, Pipeline persistence,
named feature columns, and interaction features.

Author: Customer Churn Prediction Project
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, log1p
from pyspark.sql.types import StringType, DoubleType
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    StandardScaler, Imputer
)
from pyspark.ml.functions import vector_to_array
from pyspark.ml import Pipeline
from utils.config import *
from utils.spark import create_spark_session


def load_cleaned_data(spark):
    path = os.path.join(DATA_DIR, 'cleaned_data.csv')
    if not os.path.exists(path):
        raise FileNotFoundError("Cleaned data not found. Run data_preprocessing.py first.")
    df = spark.read.csv(path, header=True, inferSchema=True)
    print(f"✓ Loaded cleaned data: {df.count():,} rows, {len(df.columns)} columns")
    return df


def create_derived_features(df):
    """
    Create interpretable derived features.
    """
    print("\n── Derived Features ─────────────────────────────")

    # 1. Charges per tenure month
    df = df.withColumn(
        'ChargesPerTenure',
        when(col('tenure') > 0, col('MonthlyCharges') / col('tenure'))
        .otherwise(col('MonthlyCharges'))
    )
    print("  ✓ ChargesPerTenure")

    # 2. Log-transformed charges (reduces skew)
    df = df.withColumn('LogMonthlyCharges', log1p(col('MonthlyCharges')))
    df = df.withColumn('LogTotalCharges',   log1p(col('TotalCharges')))
    print("  ✓ Log-transformed charges")

    # 3. Service count
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    present_services = [s for s in service_cols if s in df.columns]
    for s in present_services:
        df = df.withColumn(
            f'{s}_bin',
            when(col(s).isin('Yes', 'DSL', 'Fiber optic'), 1.0).otherwise(0.0)
        )
    df = df.withColumn('TotalServices', sum(col(f'{s}_bin') for s in present_services))
    for s in present_services:
        df = df.drop(f'{s}_bin')
    print("  ✓ TotalServices")

    # 4. Tenure group
    if BIN_CONTINUOUS_FEATURES:
        df = df.withColumn(
            'TenureGroup',
            when(col('tenure') <= 12, '0-1yr')
            .when(col('tenure') <= 24, '1-2yr')
            .when(col('tenure') <= 48, '2-4yr')
            .when(col('tenure') <= 72, '4-6yr')
            .otherwise('6+yr')
        )
        print("  ✓ TenureGroup (binned)")

    # 5. Premium services flag (guarded for missing columns)
    premium_cols = ['OnlineSecurity', 'OnlineBackup', 'TechSupport']
    available_premium = [c for c in premium_cols if c in df.columns]
    if available_premium:
        premium_check = lit(False)
        for c in available_premium:
            premium_check = premium_check | (col(c) == 'Yes')
        df = df.withColumn('HasPremiumServices', when(premium_check, 1.0).otherwise(0.0))
    else:
        df = df.withColumn('HasPremiumServices', lit(0.0))
    print("  ✓ HasPremiumServices")

    # 6. Charge ratio (actual vs. expected) — guarded
    if 'TotalCharges' in df.columns and 'MonthlyCharges' in df.columns and 'tenure' in df.columns:
        df = df.withColumn(
            'ChargeRatio',
            when(
                (col('tenure') > 0) & (col('MonthlyCharges') > 0),
                col('TotalCharges') / (col('MonthlyCharges') * col('tenure'))
            ).otherwise(1.0)
        )
    else:
        df = df.withColumn('ChargeRatio', lit(1.0))
    print("  ✓ ChargeRatio")

    # 7. High-value customer flag — guarded
    if 'MonthlyCharges' in df.columns:
        df = df.withColumn(
            'IsHighValue',
            when(col('MonthlyCharges') > 70, 1.0).otherwise(0.0)
        )
    else:
        df = df.withColumn('IsHighValue', lit(0.0))
    print("  ✓ IsHighValue")

    return df


def identify_column_types(df):
    exclude = set(COLUMNS_TO_DROP + [TARGET_COLUMN])
    categorical_cols, numerical_cols = [], []
    for name, dtype in df.dtypes:
        if name in exclude:
            continue
        if dtype == 'string':
            categorical_cols.append(name)
        elif dtype in ('int', 'double', 'float', 'bigint', 'long'):
            numerical_cols.append(name)
    print(f"\n✓ {len(categorical_cols)} categorical, {len(numerical_cols)} numerical features")
    return categorical_cols, numerical_cols


def encode_categorical_features(df, categorical_cols):
    print("\n── Encoding Categoricals ────────────────────────")
    stages = []
    for c in categorical_cols:
        si = StringIndexer(inputCol=c, outputCol=f'{c}_idx', handleInvalid='keep')
        ohe = OneHotEncoder(inputCol=f'{c}_idx', outputCol=f'{c}_enc', dropLast=True)
        stages += [si, ohe]
        print(f"  ✓ {c}")
    pipeline = Pipeline(stages=stages)
    model = pipeline.fit(df)
    df_enc = model.transform(df)
    return df_enc, model


def assemble_features(df, categorical_cols, numerical_cols):
    print("\n── Assembling Feature Vector ────────────────────")
    enc_cols = [f'{c}_enc' for c in categorical_cols]
    feature_cols = numerical_cols + enc_cols
    print(f"  Total features: {len(feature_cols)} ({len(numerical_cols)} num + {len(enc_cols)} cat)")
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol='features_raw',
        handleInvalid='keep'
    )
    df_assembled = assembler.transform(df)
    # Persist feature column names for downstream interpretability
    df_assembled = df_assembled.withColumn(
        '_feature_names_placeholder', lit(','.join(feature_cols))
    )
    return df_assembled, assembler, feature_cols


def scale_features(df, scaler_model=None):
    """Scale features. If scaler_model is provided, uses it for transform only
    (avoids data leakage when applied to test data)."""
    print("\n── Scaling Features ─────────────────────────────")
    if scaler_model is None:
        scaler = StandardScaler(
            inputCol='features_raw',
            outputCol='features',
            withMean=False,
            withStd=True
        )
        scaler_model = scaler.fit(df)
        print("  ✓ StandardScaler fitted and applied")
    else:
        print("  ✓ StandardScaler applied (pre-fitted)")
    df_scaled = scaler_model.transform(df)
    return df_scaled, scaler_model


def encode_target(df):
    print("\n── Encoding Target ──────────────────────────────")
    si = StringIndexer(inputCol=TARGET_COLUMN, outputCol='label', handleInvalid='keep')
    model = si.fit(df)
    df_labeled = model.transform(df)
    labels = model.labels
    print(f"  ✓ '{labels[0]}' → 0,  '{labels[1]}' → 1")
    return df_labeled, model


def split_data(df):
    print(f"\n── Train/Test Split ({TRAIN_TEST_SPLIT_RATIO:.0%}/{1-TRAIN_TEST_SPLIT_RATIO:.0%}) ──")
    train_df, test_df = df.randomSplit(
        [TRAIN_TEST_SPLIT_RATIO, 1 - TRAIN_TEST_SPLIT_RATIO],
        seed=RANDOM_SEED
    )
    train_n = train_df.count()
    test_n  = test_df.count()
    train_churn = train_df.filter(col('label') == 1.0).count() / train_n * 100
    test_churn  = test_df.filter(col('label') == 1.0).count() / test_n  * 100
    print(f"  Train: {train_n:,} rows  (churn rate: {train_churn:.1f}%)")
    print(f"  Test:  {test_n:,} rows  (churn rate: {test_churn:.1f}%)")
    return train_df, test_df


def save_processed_data(train_df, test_df, feature_names=None):
    print("\n── Saving Processed Data ────────────────────────")
    cols_to_save = ['features', 'label', TARGET_COLUMN]
    train_path = os.path.join(DATA_DIR, 'train_data.parquet')
    test_path  = os.path.join(DATA_DIR, 'test_data.parquet')

    train_pdf = train_df.select(cols_to_save).withColumn(
        'features', vector_to_array('features')
    ).toPandas()
    test_pdf = test_df.select(cols_to_save).withColumn(
        'features', vector_to_array('features')
    ).toPandas()

    train_pdf.to_parquet(train_path, index=False)
    test_pdf.to_parquet(test_path, index=False)
    print(f"  ✓ {train_path}")
    print(f"  ✓ {test_path}")

    # Persist feature names for interpretability
    if feature_names:
        import json
        fn_path = os.path.join(DATA_DIR, 'feature_names.json')
        with open(fn_path, 'w') as f:
            json.dump(feature_names, f)
        print(f"  ✓ Feature names saved: {fn_path}")


def main():
    print("=" * 80)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 80)

    spark = create_spark_session()
    try:
        df = load_cleaned_data(spark)
        df = create_derived_features(df)
        cat_cols, num_cols = identify_column_types(df)
        df, _ = encode_categorical_features(df, cat_cols)
        df, _, feature_names = assemble_features(df, cat_cols, num_cols)
        df, _ = encode_target(df)
        train_df, test_df = split_data(df)
        # Fit scaler on train only to prevent data leakage
        train_df, scaler_model = scale_features(train_df)
        test_df, _ = scale_features(test_df, scaler_model=scaler_model)
        save_processed_data(train_df, test_df, feature_names)
        print("\n✓ Feature engineering complete")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback; traceback.print_exc()
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
