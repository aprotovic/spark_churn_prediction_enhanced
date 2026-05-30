"""
STEP 1: DATA PREPROCESSING
============================
Enhanced data preprocessing with robust cleaning, validation, and logging.

Author: Customer Churn Prediction Project
"""

import sys
import os
import time
import logging
import glob
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql.functions import col, when, trim, lower, count, isnan, isnull, regexp_replace
from pyspark.sql.types import DoubleType, IntegerType, NumericType
from utils.config import *
from utils.spark import create_spark_session

logger = logging.getLogger(__name__)


def load_data(spark, path):
    """
    Load raw CSV data with schema inference.

    Args:
        spark: SparkSession
        path: Path to CSV file

    Returns:
        DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = spark.read.csv(path, header=True, inferSchema=True)
    print(f"✓ Loaded data: {df.count():,} rows × {len(df.columns)} columns")
    return df


def explore_data(df):
    """Print a comprehensive data exploration summary."""
    print("\n── Data Schema ─────────────────────────────────")
    df.printSchema()

    print("\n── Class Distribution ──────────────────────────")
    df.groupBy(TARGET_COLUMN).count().orderBy(TARGET_COLUMN).show()

    print("\n── Sample Rows ─────────────────────────────────")
    df.show(5, truncate=False)


def check_missing_values(df):
    """
    Report missing/null/NaN values per column.

    Returns:
        dict: column → missing count
    """
    print("\n── Missing Values ───────────────────────────────")
    missing = {}
    total = df.count()
    for c in df.columns:
        field = df.schema[c]
        if isinstance(field.dataType, NumericType):
            n = df.filter(isnull(col(c)) | isnan(col(c))).count()
        else:
            n = df.filter(isnull(col(c))).count()
        if n > 0:
            pct = n / total * 100
            print(f"  {c}: {n} ({pct:.1f}%)")
            missing[c] = n
    if not missing:
        print("  ✓ No missing values found")
    return missing


def handle_missing_values(df):
    """
    Impute or drop missing values.
    - Numeric: fill with median
    - String:  fill with 'Unknown'
    - TotalCharges: coerce and fill with MonthlyCharges
    """
    print("\n── Handling Missing Values ──────────────────────")

    # Coerce TotalCharges (often stored as string with spaces in Telco data)
    if 'TotalCharges' in df.columns:
        df = df.withColumn(
            'TotalCharges',
            regexp_replace(col('TotalCharges'), r'\s+', '').cast(DoubleType())
        )
        median_tc = df.filter(col('TotalCharges').isNotNull()).approxQuantile('TotalCharges', [0.5], 0.001)[0]
        df = df.withColumn(
            'TotalCharges',
            when(col('TotalCharges').isNull(), col('MonthlyCharges'))
            .otherwise(col('TotalCharges'))
        )
        print("  ✓ TotalCharges: coerced to numeric, nulls → MonthlyCharges")

    # Fill remaining numeric nulls with 0 and string nulls with 'Unknown'
    for field in df.schema.fields:
        if str(field.dataType) in ('DoubleType()', 'IntegerType()', 'LongType()'):
            df = df.fillna({field.name: 0})
        elif str(field.dataType) == 'StringType()':
            df = df.fillna({field.name: 'Unknown'})

    print("  ✓ Remaining numeric nulls → 0")
    print("  ✓ Remaining string nulls  → 'Unknown'")
    return df


def remove_duplicates(df):
    """Drop exact duplicate rows."""
    before = df.count()
    df = df.dropDuplicates()
    after = df.count()
    removed = before - after
    print(f"\n── Duplicate Removal ────────────────────────────")
    print(f"  Removed {removed:,} duplicates ({before:,} → {after:,})")
    return df


def convert_data_types(df):
    """
    Standardise data types:
    - Trim whitespace from strings
    - Ensure SeniorCitizen is integer
    - Ensure tenure/MonthlyCharges are correct numeric types
    """
    print("\n── Type Conversion ──────────────────────────────")

    # Trim all string columns
    string_cols = [f.name for f in df.schema.fields if str(f.dataType) == 'StringType()']
    for c in string_cols:
        df = df.withColumn(c, trim(col(c)))

    # Ensure SeniorCitizen is integer
    if 'SeniorCitizen' in df.columns:
        df = df.withColumn('SeniorCitizen', col('SeniorCitizen').cast(IntegerType()))
        print("  ✓ SeniorCitizen → IntegerType")

    # Ensure numeric columns are doubles
    for num_col in ['MonthlyCharges', 'TotalCharges']:
        if num_col in df.columns:
            df = df.withColumn(num_col, col(num_col).cast(DoubleType()))
            print(f"  ✓ {num_col} → DoubleType")

    return df


def validate_data(df):
    """
    Run basic sanity checks and raise warnings if data looks wrong.
    """
    print("\n── Data Validation ──────────────────────────────")
    n = df.count()

    # Check target column exists and has expected values
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found!")

    target_vals = [r[0] for r in df.select(TARGET_COLUMN).distinct().collect()]
    print(f"  ✓ Target values: {target_vals}")

    # Check for negative MonthlyCharges
    if 'MonthlyCharges' in df.columns:
        neg = df.filter(col('MonthlyCharges') < 0).count()
        if neg > 0:
            print(f"  ⚠ {neg} rows with negative MonthlyCharges")

    # Check tenure range
    if 'tenure' in df.columns:
        max_tenure = df.agg({'tenure': 'max'}).collect()[0][0]
        min_tenure = df.agg({'tenure': 'min'}).collect()[0][0]
        print(f"  ✓ Tenure range: {min_tenure}–{max_tenure} months")

    print(f"  ✓ Final dataset: {n:,} rows × {len(df.columns)} columns")
    return True


def save_cleaned_data(df, path):
    """Save cleaned data to CSV (single file for downstream Spark reads)."""
    print(f"\n── Saving Cleaned Data ──────────────────────────")
    df.coalesce(1).write.csv(path.replace('.csv', '_dir'), header=True, mode='overwrite')

    # Move partitioned file to expected path
    parts = glob.glob(path.replace('.csv', '_dir') + '/part-*.csv')
    if parts:
        shutil.copy(parts[0], path)
        shutil.rmtree(path.replace('.csv', '_dir'))
    print(f"  ✓ Saved to: {path}")


def main():
    print("=" * 80)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 80)

    spark = create_spark_session()

    try:
        df = load_data(spark, DATASET_PATH)
        explore_data(df)
        check_missing_values(df)
        df = handle_missing_values(df)
        df = remove_duplicates(df)
        df = convert_data_types(df)
        validate_data(df)
        save_cleaned_data(df, os.path.join(DATA_DIR, 'cleaned_data.csv'))
        print("\n✓ Preprocessing complete")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback; traceback.print_exc()
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
