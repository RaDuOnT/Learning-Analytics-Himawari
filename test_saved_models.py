# test_saved_models.py
import os
import math
import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, to_timestamp, lag, sin, cos, lit
from pyspark.sql.window import Window

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator


# ---------------- CONFIG ----------------
PARQUET_PATH = "./data_parquet/himawari_rr_features.parquet"
RUN_DIR = Path("./models/20251214_123557")  # <- change if needed

TEST_START = "2021-06-01"
TEST_END = "2021-06-30"
N_LAGS = 3
# ----------------------------------------


def add_lags(df, target_col="mean_rr"):
    w = Window.partitionBy("year").orderBy(col("timestamp_ms"))
    out = df
    for k in range(1, N_LAGS + 1):
        out = out.withColumn(
            f"{target_col}_lag{k}",
            lag(col(target_col), k).over(w)
        )
    return out


def add_cyclical_time(df):
    two_pi = 2.0 * math.pi
    return (
        df.withColumn("hour", col("hour").cast("double"))
          .withColumn("hour_sin", sin(lit(two_pi) * col("hour") / 24.0))
          .withColumn("hour_cos", cos(lit(two_pi) * col("hour") / 24.0))
    )


def prepare_dataset(df):
    df = df.withColumn(
        "timestamp",
        to_timestamp(col("timestamp_str"), "yyyy-MM-dd HH:mm:ss")
    )
    df = df.withColumn("date", to_date(col("timestamp")))
    df = add_cyclical_time(df)
    df = add_lags(df)
    return df


def evaluate(pred_df, model_name):
    evaluator_rmse = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="rmse"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName="r2"
    )
    rmse = evaluator_rmse.evaluate(pred_df)
    r2 = evaluator_r2.evaluate(pred_df)

    print(f"{model_name:18s}  RMSE={rmse:.6f}  R2={r2:.6f}")


def main():
    spark = (
        SparkSession.builder
        .appName("Himawari8-Inference")
        .master("local[8]")
        .config("spark.driver.memory", "6g")
        .getOrCreate()
    )

    print("Loading dataset...")
    df = spark.read.parquet(PARQUET_PATH)
    df = prepare_dataset(df)

    test = df.filter(
        (col("date") >= TEST_START) & (col("date") <= TEST_END)
    )

    feature_cols = [
        "max_rr", "std_rr", "frac_rainy",
        "delta_minutes",
        "hour_sin", "hour_cos",
        "mean_rr_lag1", "mean_rr_lag2", "mean_rr_lag3",
    ]

    needed = ["mean_rr"] + feature_cols
    test = test.dropna(subset=needed)

    test_base = test.select(
        col("mean_rr").alias("label"),
        *feature_cols
    ).cache()

    print("Test samples:", test_base.count())
    print("\n=== Evaluation on June 2021 ===")

    for model_dir in ["linear_regression", "decision_tree",
                      "random_forest", "xgboost"]:

        model_path = RUN_DIR / model_dir
        print(f"\nLoading model: {model_dir}")

        model = PipelineModel.load(str(model_path))
        preds = model.transform(test_base)

        evaluate(preds, model_dir)

    spark.stop()


if __name__ == "__main__":
    main()
