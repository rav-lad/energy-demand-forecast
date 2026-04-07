"""
Daily data update pipeline.

Downloads fresh data from all sources then rebuilds dataset.csv.
Designed to run as a daily cron/scheduler at 14:00 (after ENTSO-E J+1 publication).

Usage:
  python scripts/update_data.py

Cron example (14:00 every day):
  0 14 * * * cd /path/to/energy-demand-forecast && python scripts/update_data.py >> logs/update.log 2>&1
"""

import sys
import json
import logging
from datetime import date, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "update.log"),
    ],
)
log = logging.getLogger(__name__)

STATUS_FILE = PROJECT_ROOT / "data" / ".update_status.json"


def load_status() -> dict:
    if STATUS_FILE.exists():
        return json.loads(STATUS_FILE.read_text())
    return {}


def save_status(status: dict):
    STATUS_FILE.write_text(json.dumps(status, indent=2, default=str))


def _last_date_in_csv(path: Path, date_col: str, sep: str = ",") -> "pd.Timestamp | None":
    """Return the last datetime value in a CSV file, or None if the file doesn't exist."""
    import pandas as pd
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, usecols=[date_col], sep=sep)
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        last = df[date_col].dropna().max()
        return last if pd.notna(last) else None
    except Exception:
        return None


def run_download() -> bool:
    import os
    import pandas as pd
    from entsoe import EntsoePandasClient
    import requests

    api_key = os.getenv("ENTSOE_API_KEY", "3908b400-f5a2-4459-b393-c733e4946729")
    today = date.today().isoformat()

    RAW = PROJECT_ROOT / "data" / "raw"
    EXTERNAL = PROJECT_ROOT / "data" / "external"
    RAW.mkdir(parents=True, exist_ok=True)
    EXTERNAL.mkdir(parents=True, exist_ok=True)

    end_ts = pd.Timestamp(today, tz="Europe/Paris") + pd.Timedelta(days=2)

    # ----- 1. ENTSO-E spot prices FR (incremental) -----
    log.info("Downloading ENTSO-E FR spot prices (incremental)...")
    try:
        client = EntsoePandasClient(api_key=api_key)
        fr_path = RAW / "fr_spot_hourly.csv"
        last_fr = _last_date_in_csv(fr_path, "datetime")
        if last_fr is not None:
            # Overlap by 2 days to patch any late-published corrections
            fr_start = pd.Timestamp(last_fr.date() - pd.Timedelta(days=2), tz="Europe/Paris")
            log.info(f"  FR existing data up to {last_fr.date()}, fetching from {fr_start.date()}")
        else:
            fr_start = pd.Timestamp("2023-01-01", tz="Europe/Paris")
            log.info("  FR: no existing file, full download from 2023-01-01")

        prices = client.query_day_ahead_prices("FR", start=fr_start, end=end_ts)
        df_new = pd.DataFrame({"datetime": prices.index, "price": prices.values, "country": "FR"})
        df_new["datetime"] = pd.to_datetime(df_new["datetime"]).dt.tz_localize(None)

        if last_fr is not None and fr_path.exists():
            df_existing = pd.read_csv(fr_path, parse_dates=["datetime"])
            cutoff = df_new["datetime"].min()
            df_existing = df_existing[df_existing["datetime"] < cutoff]
            df_fr = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_fr = df_new

        df_fr = df_fr.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
        df_fr.to_csv(fr_path, index=False)

        daily = df_fr.copy()
        daily["date"] = daily["datetime"].dt.date
        daily = daily.groupby("date")["price"].mean().reset_index()
        daily["date"] = pd.to_datetime(daily["date"])
        daily.to_csv(RAW / "fr_spot_daily.csv", index=False)
        log.info(f"  FR: {len(df_fr)} hours total, up to {df_fr['datetime'].max().date()}")
    except Exception as e:
        log.error(f"  ENTSO-E FR failed: {e}")
        return False

    # ----- 2. ENTSO-E spot prices DE (incremental) -----
    log.info("Downloading ENTSO-E DE spot prices (incremental)...")
    try:
        de_path = RAW / "de_spot_hourly.csv"
        last_de = _last_date_in_csv(de_path, "datetime")
        if last_de is not None:
            de_start = pd.Timestamp(last_de.date() - pd.Timedelta(days=2), tz="Europe/Paris")
            log.info(f"  DE existing data up to {last_de.date()}, fetching from {de_start.date()}")
        else:
            de_start = pd.Timestamp("2023-01-01", tz="Europe/Paris")

        prices_de = client.query_day_ahead_prices("DE_LU", start=de_start, end=end_ts)
        df_new_de = pd.DataFrame({"datetime": prices_de.index, "price": prices_de.values, "country": "DE_LU"})
        df_new_de["datetime"] = pd.to_datetime(df_new_de["datetime"]).dt.tz_localize(None)

        if last_de is not None and de_path.exists():
            df_existing_de = pd.read_csv(de_path, parse_dates=["datetime"])
            cutoff_de = df_new_de["datetime"].min()
            df_existing_de = df_existing_de[df_existing_de["datetime"] < cutoff_de]
            df_de = pd.concat([df_existing_de, df_new_de], ignore_index=True)
        else:
            df_de = df_new_de

        df_de = df_de.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
        df_de.to_csv(de_path, index=False)

        daily_de = df_de.copy()
        daily_de["date"] = daily_de["datetime"].dt.date
        daily_de = daily_de.groupby("date")["price"].mean().reset_index()
        daily_de["date"] = pd.to_datetime(daily_de["date"])
        daily_de.to_csv(RAW / "de_spot_daily.csv", index=False)
        log.info(f"  DE: {len(df_de)} hours total, up to {df_de['datetime'].max().date()}")
    except Exception as e:
        log.warning(f"  ENTSO-E DE failed (non-blocking): {e}")

    # ----- 3. Open-Meteo weather (incremental) -----
    log.info("Downloading weather from Open-Meteo (incremental)...")
    try:
        # Find existing weather file
        existing_weather = sorted(EXTERNAL.glob("weather_hourly_*.csv"))
        weather_start = "2023-01-01"
        df_existing_w = None

        if existing_weather:
            last_w = _last_date_in_csv(existing_weather[-1], "datetime")
            if last_w is not None and last_w.date() >= date.today() - __import__("datetime").timedelta(days=2):
                log.info(f"  Weather already up to {last_w.date()}, skipping download")
                # Still rename to today's file if needed
                df_existing_w = pd.read_csv(existing_weather[-1], parse_dates=["datetime"])
            elif last_w is not None:
                # Download only missing portion (overlap 2 days)
                weather_start = (last_w.date() - __import__("datetime").timedelta(days=2)).isoformat()
                df_existing_w = pd.read_csv(existing_weather[-1], parse_dates=["datetime"])
                df_existing_w = df_existing_w[df_existing_w["datetime"] < pd.Timestamp(weather_start)]
                log.info(f"  Weather existing up to {last_w.date()}, fetching from {weather_start}")

        if df_existing_w is None or last_w.date() < date.today() - __import__("datetime").timedelta(days=2):
            resp = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude": 48.8566,
                    "longitude": 2.3522,
                    "start_date": weather_start,
                    "end_date": today,
                    "hourly": "temperature_2m,wind_speed_10m,precipitation,shortwave_radiation",
                    "timezone": "Europe/Paris",
                },
                timeout=60,
            )
            resp.raise_for_status()
            h = resp.json()["hourly"]
            df_new_w = pd.DataFrame({
                "datetime": pd.to_datetime(h["time"]),
                "temperature_2m": h["temperature_2m"],
                "wind_speed_10m": h["wind_speed_10m"],
                "precipitation": h["precipitation"],
                "shortwave_radiation": h["shortwave_radiation"],
            })

            if df_existing_w is not None and len(df_existing_w) > 0:
                df_w = pd.concat([df_existing_w, df_new_w], ignore_index=True)
            else:
                df_w = df_new_w

            df_w = df_w.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
        else:
            df_w = df_existing_w

        # Remove old weather files and write the merged one
        for old in EXTERNAL.glob("weather_hourly_*.csv"):
            old.unlink()
        out_weather = EXTERNAL / f"weather_hourly_2023-01-01_{today}.csv"
        df_w.to_csv(out_weather, index=False)
        log.info(f"  Weather: {len(df_w)} hours total, up to {df_w['datetime'].max().date()}")
    except Exception as e:
        log.error(f"  Open-Meteo failed: {e}")
        return False

    # ----- 4. ODRE consumption (full re-download — API does not support incremental) -----
    log.info("Downloading ODRE consumption (always full — API limitation)...")
    try:
        resp = requests.get(
            "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/consommation-quotidienne-brute-regionale/exports/csv",
            params={
                "where": "date_heure >= '2023-01-01'",
                "limit": -1,
                "timezone": "Europe/Paris",
                "delimiter": ";",
            },
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()
        out_cons = RAW / "energy_consumption_2023-2026.csv"
        with open(out_cons, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        df_cons = pd.read_csv(out_cons, sep=";", usecols=["date_heure"])
        log.info(f"  Consumption: {len(df_cons)} rows")
    except Exception as e:
        log.warning(f"  ODRE failed (non-blocking, using existing): {e}")

    return True


def run_build() -> bool:
    log.info("Building dataset.csv...")
    try:
        from scripts.build_dataset import build
        df = build()
        log.info(f"  dataset.csv: {len(df)} rows, up to {df['datetime'].max().date()}")
        return True
    except Exception as e:
        log.error(f"  Build failed: {e}")
        return False


def main():
    log.info("=" * 60)
    log.info(f"Daily update started — {datetime.now().isoformat()}")
    log.info("=" * 60)

    status = load_status()
    status["last_run"] = datetime.now().isoformat()

    ok_download = run_download()
    status["download_ok"] = ok_download
    status["download_time"] = datetime.now().isoformat()

    if not ok_download:
        log.error("Download failed — aborting build.")
        status["build_ok"] = False
        save_status(status)
        return 1

    ok_build = run_build()
    status["build_ok"] = ok_build
    status["build_time"] = datetime.now().isoformat()

    # Record dataset date range for dashboard
    try:
        import pandas as pd
        df = pd.read_csv(PROJECT_ROOT / "data" / "dataset.csv", usecols=["datetime"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        status["dataset_start"] = str(df["datetime"].min().date())
        status["dataset_end"] = str(df["datetime"].max().date())
        status["dataset_rows"] = len(df)
    except Exception:
        pass

    save_status(status)

    if ok_build:
        log.info("Update complete.")
        return 0
    else:
        log.error("Build failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
