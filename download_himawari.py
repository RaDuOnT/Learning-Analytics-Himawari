import re
import subprocess
import datetime as dt
from pathlib import Path

BUCKET = "gs://noaa-himawari8/AHI-L2-FLDK-RainfallRate"
OUT_DIR = Path("./data_raw_2020")  # ✅ keep this fixed

TIME_RE = re.compile(r"/(\d{2})(\d{2})/[^/]+\.nc$")


def run_ls(day_prefix: str) -> list[str]:
    p = subprocess.run(
        ["gsutil", "-m", "ls", f"{day_prefix}/*/*.nc"],
        capture_output=True,
        text=True
    )
    return [line.strip() for line in p.stdout.splitlines() if line.strip()]


def parse_hhmm(url: str):
    m = TIME_RE.search(url)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2))
    return hh, mm


def minutes_since_midnight(hh: int, mm: int) -> int:
    return hh * 60 + mm


def choose_closest(files: list[str]) -> list[str]:
    items = []
    for u in files:
        t = parse_hhmm(u)
        if t is None:
            continue
        hh, mm = t
        items.append((u, minutes_since_midnight(hh, mm)))

    if not items:
        return []

    chosen = []
    for hour in range(24):
        target = hour * 60
        best = min(items, key=lambda x: (
            abs(x[1] - target), x[1]))  # tie -> earlier
        chosen.append(best[0])

    return sorted(set(chosen))  # dedupe


def list_local_nc_names(out_dir: Path) -> set[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return {p.name for p in out_dir.glob("*.nc")}


def download_urls(urls: list[str], dest: Path, already: set[str]):
    if not urls:
        return 0

    # pre-filter: skip urls whose basename already exists locally
    to_get = []
    for u in urls:
        name = u.rsplit("/", 1)[-1]
        if name not in already:
            to_get.append(u)

    if not to_get:
        return 0

    inp = "\n".join(to_get) + "\n"
    # -n ensures no overwrite even if something changed during runtime
    subprocess.run(
        ["gsutil", "-m", "cp", "-n", "-I", str(dest)],
        input=inp,
        text=True,
        check=False
    )
    return len(to_get)


def download_range(start: dt.date, end: dt.date):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # cache local existing files once; update as we download
    existing = list_local_nc_names(OUT_DIR)
    print(f"Local .nc already present: {len(existing)}")

    cur = start
    while cur <= end:
        y, m, d = cur.year, cur.month, cur.day
        day_prefix = f"{BUCKET}/{y}/{m:02d}/{d:02d}"
        print(f"=== {y}-{m:02d}-{d:02d} ===")

        files = run_ls(day_prefix)
        if not files:
            print("  (no files)")
            cur += dt.timedelta(days=1)
            continue

        chosen = choose_closest(files)
        n_before = len(existing)
        downloaded = download_urls(chosen, OUT_DIR, existing)

        # refresh existing set only if we attempted downloads
        if downloaded > 0:
            existing = list_local_nc_names(OUT_DIR)

        print(
            f"  available={len(files)} chosen={len(chosen)} new_requested={downloaded} local_now={len(existing)} (+{len(existing)-n_before})")

        cur += dt.timedelta(days=1)


if __name__ == "__main__":
    # test first (recommended)
    # download_range(dt.date(2020, 7, 1), dt.date(2020, 12, 31))
    download_range(dt.date(2021, 6, 1), dt.date(2021, 6, 30))
