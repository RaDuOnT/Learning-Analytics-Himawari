# train_spark_models.py
import os
import math
import json
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, to_timestamp, lag, sin, cos, lit
from pyspark.sql.window import Window

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

from xgboost.spark import SparkXGBRegressor

PARQUET_DIR = "./data_parquet"
PARQUET_FILE = "himawari_rr_features.parquet"

TRAIN_YEAR = 2020
TEST_START = "2021-06-20"
TEST_END = "2021-06-27"
N_LAGS = 3

# ---- MODEL OUTPUT ----
MODELS_DIR = Path("./models")
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = MODELS_DIR / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ---- GPU SWITCH (XGBoost only) ----
USE_GPU_FOR_XGB = False   # set False to force CPU
GPU_ID = 0               # your RTX 4070 Ti SUPER


def add_lags(df, target_col="mean_rr"):
    w = Window.partitionBy("year").orderBy(col("timestamp_ms"))
    out = df
    for k in range(1, N_LAGS + 1):
        out = out.withColumn(f"{target_col}_lag{k}",
                             lag(col(target_col), k).over(w))
    return out


def add_cyclical_time(df):
    two_pi = 2.0 * math.pi
    return (
        df.withColumn("hour", col("hour").cast("double"))
          .withColumn("hour_sin", sin(lit(two_pi) * col("hour") / lit(24.0)))
          .withColumn("hour_cos", cos(lit(two_pi) * col("hour") / lit(24.0)))
    )


def prepare_dataset(df):
    df = df.withColumn("timestamp", to_timestamp(
        col("timestamp_str"), "yyyy-MM-dd HH:mm:ss"))
    df = df.withColumn("date", to_date(col("timestamp")))
    df = add_cyclical_time(df)
    df = add_lags(df, target_col="mean_rr")
    return df


def split_train_test(df):
    train = df.filter(col("year") == TRAIN_YEAR)
    test = df.filter((col("date") >= TEST_START) & (col("date") <= TEST_END))
    return train, test


def eval_model(pred, name):
    evaluator_rmse = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="rmse"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="r2"
    )
    rmse = evaluator_rmse.evaluate(pred)
    r2 = evaluator_r2.evaluate(pred)
    print(f"{name:18s}  RMSE={rmse:.6f}  R2={r2:.6f}")
    return rmse, r2


def save_model(model, subdir_name: str):
    path = str(RUN_DIR / subdir_name)
    # PipelineModel + Spark ML models support this
    model.write().overwrite().save(path)
    print(f"Saved: {subdir_name} -> {path}")


def main():
    # With 15GiB total RAM, keep Spark conservative.
    # These values usually work well on your machine.
    spark = (
        SparkSession.builder
        .appName("Himawari8-ML-Local")
        .master("local[8]")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.default.parallelism", "64")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

    parquet_path = os.path.join(PARQUET_DIR, PARQUET_FILE)
    print("Reading parquet from:", parquet_path)
    df = spark.read.parquet(parquet_path)

    df = prepare_dataset(df)

    train, test = split_train_test(df)
    print("Train count (raw):", train.count())
    print("Test count (raw):", test.count())

    feature_cols = [
        "max_rr", "std_rr", "frac_rainy",
        "delta_minutes",
        "hour_sin", "hour_cos",
        "mean_rr_lag1", "mean_rr_lag2", "mean_rr_lag3",
    ]

    needed = ["mean_rr"] + feature_cols
    train = train.dropna(subset=needed)
    test = test.dropna(subset=needed)

    train_base = train.select(
        col("mean_rr").alias("label"), *feature_cols).cache()
    test_base = test.select(col("mean_rr").alias(
        "label"), *feature_cols).cache()
    train_base.count()
    test_base.count()

    print("Train count (after dropna):", train_base.count())
    print("Test count (after dropna):", test_base.count())

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )

    results = {}

    print("\n=== Models ===")

    # ---- Linear Regression (CPU, scaled) ----
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaledFeatures",
        withMean=True,
        withStd=True
    )
    lr = LinearRegression(
        featuresCol="scaledFeatures",
        labelCol="label",
        predictionCol="prediction"
    )
    lr_pipe = Pipeline(stages=[assembler, scaler, lr])
    lr_model = lr_pipe.fit(train_base)
    pred_lr = lr_model.transform(test_base)
    results["linear_regression"] = eval_model(pred_lr, "Linear Regression")
    save_model(lr_model, "linear_regression")

    # ---- Decision Tree (CPU, no scaling) ----
    dt = DecisionTreeRegressor(
        featuresCol="features",
        labelCol="label",
        predictionCol="prediction",
        maxDepth=8,
        minInstancesPerNode=10
    )
    dt_pipe = Pipeline(stages=[assembler, dt])
    dt_model = dt_pipe.fit(train_base)
    pred_dt = dt_model.transform(test_base)
    results["decision_tree"] = eval_model(pred_dt, "Decision Tree")
    save_model(dt_model, "decision_tree")

    # ---- Random Forest (CPU, lighter) ----
    rf = RandomForestRegressor(
        featuresCol="features", labelCol="label", predictionCol="prediction",
        numTrees=100,
        maxDepth=8,
        minInstancesPerNode=20,
        subsamplingRate=0.7,
        featureSubsetStrategy="sqrt"
    )
    rf_pipe = Pipeline(stages=[assembler, rf])
    rf_model = rf_pipe.fit(train_base)
    pred_rf = rf_model.transform(test_base)
    results["random_forest"] = eval_model(pred_rf, "Random Forest")
    save_model(rf_model, "random_forest")

    # ---- XGBoost (CPU or GPU) ----
    # NOTE: only this can use your RTX GPU.
    xgb_params = dict(
        features_col="features",
        label_col="label",
        prediction_col="prediction",
        num_workers=min(8, os.cpu_count() or 4),
        objective="reg:squarederror",
        eta=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_estimators=600,
        tree_method="hist",   # CPU
        device="cpu",         # CPU
    )

    xgb = SparkXGBRegressor(**xgb_params)

    xgb_pipe = Pipeline(stages=[assembler, xgb])
    xgb_model = xgb_pipe.fit(train_base)
    pred_xgb = xgb_model.transform(test_base)
    results["xgboost"] = eval_model(pred_xgb, "XGBoost")
    save_model(xgb_model, "xgboost")

    # ---- Save run metadata ----
    info = {
        "run_id": RUN_ID,
        "train_year": TRAIN_YEAR,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "feature_cols": feature_cols,
        "use_gpu_for_xgb": USE_GPU_FOR_XGB,
        "gpu_id": GPU_ID,
        "results": {k: {"rmse": v[0], "r2": v[1]} for k, v in results.items()},
    }
    (RUN_DIR / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"Saved run_info.json -> {RUN_DIR / 'run_info.json'}")

    spark.stop()


if __name__ == "__main__":
    main()
