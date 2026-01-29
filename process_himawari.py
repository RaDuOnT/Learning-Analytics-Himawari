# process_himawari.py

import os
import glob
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

RAW_DIR = "./data_raw_2020"         # keep your path
PARQUET_DIR = "./data_parquet"
OUT_FILE = "himawari_rr_features.parquet"

# Patch size (center crop)
WIN = 200  # 200..500

# Filename patterns:
# 1) New style: ..._sYYYYMMDDHHMMSS...
RE_S = re.compile(r"_s(\d{14})")
# 2) Old style: ..._YYYYDOY_HHMM_... e.g. _2020199_1350_
RE_DOY = re.compile(r"_(\d{4})(\d{3})_(\d{2})(\d{2})_")


def parse_timestamp_from_name(fname: str):
    m = RE_S.search(fname)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
        except Exception:
            return None

    m = RE_DOY.search(fname)
    if m:
        try:
            year = int(m.group(1))
            doy = int(m.group(2))
            hh = int(m.group(3))
            mm = int(m.group(4))
            base = datetime(year, 1, 1) + timedelta(days=doy - 1)
            return base.replace(hour=hh, minute=mm, second=0)
        except Exception:
            return None

    return None


def pick_rainfall_var(ds: xr.Dataset):
    # Prefer RRQPE
    if "RRQPE" in ds.data_vars:
        return "RRQPE"

    # Fallback: any 2D float var that isn't lat/lon
    candidates = []
    for k, v in ds.data_vars.items():
        if k.lower() in {"latitude", "longitude"}:
            continue
        if len(v.dims) == 2 and np.issubdtype(v.dtype, np.floating):
            candidates.append(k)

    return candidates[0] if candidates else None


def process_nc_file(path: str):
    fname = os.path.basename(path)
    timestamp = parse_timestamp_from_name(fname)
    if timestamp is None:
        return None

    try:
        ds = xr.open_dataset(path)
    except Exception as e:
        print("Error opening", fname, e)
        return None

    varname = pick_rainfall_var(ds)
    if varname is None:
        print("No suitable rainfall var in",
              fname, "vars=", list(ds.data_vars))
        return None

    rr = ds[varname]
    dqf = ds["DQF"] if "DQF" in ds.data_vars else None

    # Expect 2D
    if len(rr.dims) != 2:
        return None

    dy, dx = rr.dims[0], rr.dims[1]
    ny, nx = rr.sizes[dy], rr.sizes[dx]

    cy, cx = ny // 2, nx // 2
    half = WIN // 2

    y0, y1 = max(0, cy - half), min(ny, cy + half)
    x0, x1 = max(0, cx - half), min(nx, cx + half)

    rr_patch = rr.isel({dy: slice(y0, y1), dx: slice(x0, x1)}
                       ).values.astype("float32")

    # Apply DQF mask if aligned
    if dqf is not None and dqf.dims == rr.dims:
        dqf_patch = dqf.isel({dy: slice(y0, y1), dx: slice(x0, x1)}).values
        rr_patch = np.where(dqf_patch == 0, rr_patch, np.nan)

    # Remove negatives
    rr_patch = np.where(rr_patch < 0, np.nan, rr_patch)

    # If everything is NaN, skip the file (prevents your warnings + NULL stats)
    if not np.isfinite(rr_patch).any():
        return None

    mean_rr = float(np.nanmean(rr_patch))
    max_rr = float(np.nanmax(rr_patch))
    std_rr = float(np.nanstd(rr_patch))
    frac_rainy = float(np.nanmean(rr_patch > 0))

    return {
        "file": fname,
        "timestamp": timestamp,
        "mean_rr": mean_rr,
        "max_rr": max_rr,
        "std_rr": std_rr,
        "frac_rainy": frac_rainy,
    }


def main():
    os.makedirs(PARQUET_DIR, exist_ok=True)

    nc_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.nc")))
    print("RAW_DIR:", RAW_DIR)
    print("Found", len(nc_files), "netCDF files")

    rows = []
    for i, f in enumerate(nc_files):
        if i % 200 == 0:
            print(f"[{i}/{len(nc_files)}] {f}")
        row = process_nc_file(f)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid rows, exiting.")
        return

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Time features
    df["year"] = df["timestamp"].dt.year
    df["month"] = df["timestamp"].dt.month
    df["day"] = df["timestamp"].dt.day
    df["hour"] = df["timestamp"].dt.hour

    df["timestamp_ms"] = (df["timestamp"].astype(
        "int64") // 1_000_000).astype("int64")
    df["timestamp_str"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    df["delta_minutes"] = (
        df["timestamp"].diff().dt.total_seconds().div(
            60).fillna(0).astype("float32")
    )

    df = df.drop(columns=["timestamp"])

    out_path = os.path.join(PARQUET_DIR, OUT_FILE)
    df.to_parquet(out_path, index=False)
    print("Saved parquet to:", out_path)

    print("Sanity:")
    print("rows:", len(df))
    print("years:", sorted(df["year"].unique().tolist()))
    print("mean_rr min/max:", df["mean_rr"].min(), df["mean_rr"].max())


if __name__ == "__main__":
    main()
