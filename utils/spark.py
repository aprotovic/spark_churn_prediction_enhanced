"""
Centralised Spark session factory.
All modules should import create_spark_session from here
instead of defining their own.
"""

from pyspark.sql import SparkSession
from utils.config import SPARK_CONFIG


def create_spark_session(app_name=None):
    """Create and return a configured Spark session.

    Args:
        app_name: Optional custom app name. Defaults to config value.

    Returns:
        SparkSession
    """
    spark = SparkSession.builder \
        .appName(app_name or SPARK_CONFIG['app_name']) \
        .master(SPARK_CONFIG['master']) \
        .config('spark.executor.memory', SPARK_CONFIG['spark.executor.memory']) \
        .config('spark.driver.memory', SPARK_CONFIG['spark.driver.memory']) \
        .config('spark.sql.adaptive.enabled',
                SPARK_CONFIG.get('spark.sql.adaptive.enabled', 'true')) \
        .config('spark.sql.adaptive.coalescePartitions.enabled',
                SPARK_CONFIG.get('spark.sql.adaptive.coalescePartitions.enabled', 'true')) \
        .getOrCreate()

    spark.sparkContext.setLogLevel('WARN')
    return spark
