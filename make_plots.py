import os
import json
import math
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, to_timestamp, lag, sin, cos, lit
from pyspark.sql.window import Window
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator


# -------------------------
# Config / dataset
# -------------------------
PARQUET_DIR = "./data_parquet"
PARQUET_FILE = "himawari_rr_features.parquet"
N_LAGS = 3

# June 2021 default interval (poți schimba din CLI)
DEFAULT_START = "2021-06-01"
DEFAULT_END = "2021-06-30"

FEATURE_COLS = [
    "max_rr", "std_rr", "frac_rainy",
    "delta_minutes",
    "hour_sin", "hour_cos",
    "mean_rr_lag1", "mean_rr_lag2", "mean_rr_lag3",
]


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


def eval_metrics(pred_df, label_col="label", pred_col="prediction"):
    evaluator_rmse = RegressionEvaluator(
        labelCol=label_col, predictionCol=pred_col, metricName="rmse")
    evaluator_r2 = RegressionEvaluator(
        labelCol=label_col, predictionCol=pred_col, metricName="r2")
    rmse = evaluator_rmse.evaluate(pred_df)
    r2 = evaluator_r2.evaluate(pred_df)
    return rmse, r2


def find_latest_run_dir(models_root: Path) -> Path:
    runs = [p for p in models_root.iterdir() if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"No runs found in {models_root}")
    return sorted(runs, key=lambda p: p.name)[-1]


def load_run_info(run_dir: Path):
    info_path = run_dir / "run_info.json"
    if info_path.exists():
        return json.loads(info_path.read_text())
    return None


def safe_collect_to_pandas(df, max_rows=20000):
    # ca să nu explodeze RAM-ul dacă ai multe rânduri
    cnt = df.count()
    if cnt > max_rows:
        df = df.limit(max_rows)
    return df.toPandas()


def plot_bar(values_dict, title, ylabel, out_path):
    names = list(values_dict.keys())
    vals = [values_dict[n] for n in names]

    plt.figure()
    plt.bar(names, vals)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_timeseries(df_pd, model_name, out_path):
    # df_pd must contain: timestamp, label, prediction
    df_pd = df_pd.sort_values("timestamp")

    plt.figure()
    plt.plot(df_pd["timestamp"], df_pd["label"], label="Real")
    plt.plot(df_pd["timestamp"], df_pd["prediction"], label="Predictie")
    plt.title(f"Predictii vs Real - {model_name}")
    plt.xlabel("Timp")
    plt.ylabel("mean_rr")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scatter(df_pd, model_name, out_path):
    plt.figure()
    plt.scatter(df_pd["label"], df_pd["prediction"], s=10)
    plt.title(f"Scatter: Predictie vs Real - {model_name}")
    plt.xlabel("Real (label)")
    plt.ylabel("Predictie")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_residual_hist(df_pd, model_name, out_path):
    resid = df_pd["prediction"] - df_pd["label"]
    plt.figure()
    plt.hist(resid, bins=40)
    plt.title(f"Distribuția erorilor (prediction-label) - {model_name}")
    plt.xlabel("Eroare")
    plt.ylabel("Frecvență")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_abs_error_timeseries(df_pd, model_name, out_path):
    df_pd = df_pd.sort_values("timestamp")
    abs_err = (df_pd["prediction"] - df_pd["label"]).abs()

    plt.figure()
    plt.plot(df_pd["timestamp"], abs_err)
    plt.title(f"Eroare absolută în timp - {model_name}")
    plt.xlabel("Timp")
    plt.ylabel("|prediction - label|")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_feature_importance(importances, feature_names, title, out_path, top_k=20):
    imp = np.array(importances, dtype=float)
    idx = np.argsort(imp)[::-1][:top_k]
    names = [feature_names[i] for i in idx]
    vals = imp[idx]

    plt.figure()
    plt.bar(names, vals)
    plt.title(title)
    plt.ylabel("Importance")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scatter_with_identity(df_pd, model_name, out_path):
    y = df_pd["label"].to_numpy()
    p = df_pd["prediction"].to_numpy()

    plt.figure()
    plt.scatter(y, p, s=10)

    mn = float(min(y.min(), p.min()))
    mx = float(max(y.max(), p.max()))
    plt.plot([mn, mx], [mn, mx], linewidth=1)  # linia y=x

    plt.title(f"Pred vs Real (+ y=x) - {model_name}")
    plt.xlabel("Real (label)")
    plt.ylabel("Predicție")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_lr_coefficients(coeffs, feature_names, title, out_path):
    coeffs = np.array(coeffs, dtype=float)
    idx = np.argsort(np.abs(coeffs))[::-1]
    names = [feature_names[i] for i in idx]
    vals = coeffs[idx]

    plt.figure()
    plt.bar(names, vals)
    plt.title(title)
    plt.ylabel("Coefficient (signed)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_correlation_matrix(df_pd_features, out_path, title="Corelații între features"):
    corr = df_pd_features.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(corr.to_numpy(), aspect="auto")
    plt.title(title)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_target_distribution(df_pd, out_path, title="Distribuția lui mean_rr"):
    plt.figure()
    plt.hist(df_pd["label"], bins=60)
    plt.title(title)
    plt.xlabel("mean_rr (label)")
    plt.ylabel("Frecvență")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_rmse_r2_combined(rmse_dict, r2_dict, out_path):
    models = list(rmse_dict.keys())
    rmse_vals = [rmse_dict[m] for m in models]
    r2_vals = [r2_dict[m] for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, rmse_vals, width, label="RMSE")
    plt.bar(x + width/2, r2_vals, width, label="R²")

    plt.xticks(x, models, rotation=20, ha="right")
    plt.ylabel("Valoare")
    plt.title("Compararea RMSE și R² între modele (June 2021)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_top_errors(df_pd, model_name, out_path, top_n=20):
    df_pd = df_pd.copy()
    df_pd["abs_error"] = (df_pd["prediction"] - df_pd["label"]).abs()
    top = df_pd.sort_values("abs_error", ascending=False).head(top_n)

    plt.figure()
    plt.bar(range(top_n), top["abs_error"])
    plt.xticks(range(top_n), top["timestamp"].dt.strftime(
        "%m-%d %H:%M"), rotation=45)
    plt.title(f"Top {top_n} cele mai mari erori - {model_name}")
    plt.xlabel("Timp")
    plt.ylabel("|prediction - label|")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default=None,
                    help="Ex: models/20251214_123557")
    ap.add_argument("--models_root", type=str, default="models")
    ap.add_argument("--start", type=str, default=DEFAULT_START)
    ap.add_argument("--end", type=str, default=DEFAULT_END)
    ap.add_argument("--out_dir", type=str, default="plots")
    args = ap.parse_args()

    models_root = Path(args.models_root)

    run_dir = Path(args.run_dir) if args.run_dir else find_latest_run_dir(
        models_root)
    run_info = load_run_info(run_dir)

    out_dir = Path(args.out_dir) / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using run_dir: {run_dir}")
    print(f"Saving plots to: {out_dir}")

    spark = (
        SparkSession.builder
        .appName("Himawari8-Plot-Saved-Models")
        .master("local[8]")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "6g")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.default.parallelism", "64")
        .getOrCreate()
    )

    parquet_path = os.path.join(PARQUET_DIR, PARQUET_FILE)
    print("Loading parquet:", parquet_path)
    df = spark.read.parquet(parquet_path)

    df = prepare_dataset(df)

    needed = ["mean_rr"] + FEATURE_COLS + ["timestamp", "date"]
    df = df.dropna(subset=needed)

    print("Dataset schema:")
    df.printSchema()
    print("Total samples:", df.count())

    test = df.filter((col("date") >= args.start) & (col("date") <= args.end))

    test_base = test.select(
        col("timestamp"),
        col("mean_rr").alias("label"),
        *FEATURE_COLS
    ).cache()

    # Sample pt corelații + distribuția targetului
    pd_sample = safe_collect_to_pandas(test_base.select(
        ["label"] + FEATURE_COLS), max_rows=50000)
    plot_target_distribution(pd_sample, out_dir /
                             "target_distribution_mean_rr.png")
    plot_correlation_matrix(
        pd_sample[FEATURE_COLS], out_dir / "feature_correlations.png")

    print("Test samples:", test_base.count())

    # assembler (pt cazul în care ai salvat doar modelul, nu pipeline-ul)
    assembler = VectorAssembler(
        inputCols=FEATURE_COLS, outputCol="features", handleInvalid="skip")

    model_names = ["linear_regression",
                   "decision_tree", "random_forest", "xgboost"]
    rmse_dict = {}
    r2_dict = {}

    all_pred_pd = {}  # model_name -> pandas df

    for name in model_names:
        model_path = run_dir / name
        if not model_path.exists():
            print(f"Skipping (not found): {name} -> {model_path}")
            continue

        print(f"\nLoading model: {name}")
        model = PipelineModel.load(str(model_path))

        # Pipeline => transform direct
        pred = model.transform(test_base)

        rmse, r2 = eval_metrics(pred)
        rmse_dict[name] = rmse
        r2_dict[name] = r2

        # colectăm pt grafice
        pred_small = pred.select("timestamp", "label", "prediction")
        pred_pd = safe_collect_to_pandas(pred_small, max_rows=50000)
        pred_pd["timestamp"] = pd.to_datetime(pred_pd["timestamp"])
        all_pred_pd[name] = pred_pd

        # -----------------------
        # Grafice cerute / extra
        # -----------------------
        plot_timeseries(pred_pd, name, out_dir /
                        f"timeseries_pred_vs_real_{name}.png")
        plot_scatter(pred_pd, name, out_dir /
                     f"scatter_pred_vs_real_{name}.png")
        plot_scatter_with_identity(
            pred_pd, name, out_dir / f"scatter_pred_vs_real_identity_{name}.png")
        plot_residual_hist(pred_pd, name, out_dir /
                           f"residual_hist_{name}.png")
        plot_abs_error_timeseries(
            pred_pd, name, out_dir / f"abs_error_timeseries_{name}.png")

        plot_top_errors(
            pred_pd,
            name,
            out_dir / f"top_errors_{name}.png",
            top_n=20
        )

        # -----------------------
        # Interpretabilitate:
        # Feature importance / coeficienți
        # -----------------------
        try:
            last_stage = model.stages[-1]

            # DecisionTree / RandomForest
            if hasattr(last_stage, "featureImportances"):
                fi = last_stage.featureImportances.toArray()
                plot_feature_importance(
                    fi,
                    FEATURE_COLS,
                    title=f"Feature importance - {name}",
                    out_path=out_dir / f"feature_importance_{name}.png",
                    top_k=len(FEATURE_COLS),
                )

            # Linear Regression coefficients
            if name == "linear_regression" and hasattr(last_stage, "coefficients"):
                plot_lr_coefficients(
                    last_stage.coefficients,
                    FEATURE_COLS,
                    title="Linear Regression coefficients (scaled features)",
                    out_path=out_dir / "lr_coefficients.png",
                )
        except Exception as e:
            print(
                f"[WARN] Could not generate importance/coeff plots for {name}: {e}")

        # -----------------------
        # Extra: salvează predicțiile pt eseu (opțional, foarte util)
        # -----------------------
        try:
            pred_pd.to_csv(out_dir / f"predictions_{name}.csv", index=False)
        except Exception as e:
            print(f"[WARN] Could not save predictions CSV for {name}: {e}")

        print(f"{name:16s} RMSE={rmse:.6f}  R2={r2:.6f}")

    # grafice de comparație
    if rmse_dict:
        plot_bar(
            rmse_dict,
            title="Compararea RMSE între modele (June 2021)",
            ylabel="RMSE",
            out_path=out_dir / "compare_rmse.png",
        )

    if r2_dict:
        plot_bar(
            r2_dict,
            title="Compararea R² între modele (June 2021)",
            ylabel="R²",
            out_path=out_dir / "compare_r2.png",
        )

    if rmse_dict and r2_dict:
        plot_rmse_r2_combined(
            rmse_dict,
            r2_dict,
            out_dir / "compare_rmse_r2_combined.png",
        )

    # un grafic “combo” (toate modelele peste real)
    if all_pred_pd:
        # folosim indexul temporal din primul model
        first_name = list(all_pred_pd.keys())[0]
        base = all_pred_pd[first_name].sort_values(
            "timestamp")[["timestamp", "label"]].copy()

        plt.figure()
        plt.plot(base["timestamp"], base["label"], label="Real")

        for name, pred_pd in all_pred_pd.items():
            pred_pd = pred_pd.sort_values("timestamp")
            plt.plot(pred_pd["timestamp"],
                     pred_pd["prediction"], label=f"Pred: {name}")

        plt.title("Predicții vs Real (toate modelele) - June 2021")
        plt.xlabel("Timp")
        plt.ylabel("mean_rr")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "timeseries_all_models.png", dpi=200)
        plt.close()

    # salvează și un CSV cu rezultate pentru eseu
    summary_path = out_dir / "metrics_summary.json"
    summary = {
        "run_dir": str(run_dir),
        "date_range": {"start": args.start, "end": args.end},
        "rmse": rmse_dict,
        "r2": r2_dict,
        "generated_at": datetime.now().isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print("\nSaved:", summary_path)

    spark.stop()
    print("Done.")


if __name__ == "__main__":
    main()
