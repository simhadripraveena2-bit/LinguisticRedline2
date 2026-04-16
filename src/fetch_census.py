"""Fetch ACS tract-level demographic data with flexible city sampling for scale-up studies."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import pandas as pd
import requests

from config_loader import load_config

OUTPUT_PATH = Path("data/census_tracts.csv")
ACS_VARIABLES = [
    "B02001_002E", "B02001_003E", "B02001_005E", "B03001_003E", "B01003_001E", "B19013_001E", "B25002_003E", "B25001_001E",
]
CITY_COUNTIES: Dict[str, List[Dict[str, str]]] = {
    "New York": [{"state": "36", "county": "005"}, {"state": "36", "county": "047"}, {"state": "36", "county": "061"}, {"state": "36", "county": "081"}, {"state": "36", "county": "085"}],
    "Los Angeles": [{"state": "06", "county": "037"}], "Chicago": [{"state": "17", "county": "031"}], "Houston": [{"state": "48", "county": "201"}],
    "Phoenix": [{"state": "04", "county": "013"}], "Philadelphia": [{"state": "42", "county": "101"}], "San Antonio": [{"state": "48", "county": "029"}],
    "San Diego": [{"state": "06", "county": "073"}], "Dallas": [{"state": "48", "county": "113"}], "San Jose": [{"state": "06", "county": "085"}],
    "Atlanta": [{"state": "13", "county": "121"}], "Detroit": [{"state": "26", "county": "163"}], "Baltimore": [{"state": "24", "county": "510"}],
    "Cleveland": [{"state": "39", "county": "035"}], "Memphis": [{"state": "47", "county": "157"}], "New Orleans": [{"state": "22", "county": "071"}],
    "St. Louis": [{"state": "29", "county": "510"}], "Milwaukee": [{"state": "55", "county": "079"}], "Oakland": [{"state": "06", "county": "001"}],
    "Minneapolis": [{"state": "27", "county": "053"}],
}
CITY_REGION = {
    "New York": "Northeast", "Philadelphia": "Northeast", "Baltimore": "South", "Atlanta": "South", "Houston": "South", "Memphis": "South", "New Orleans": "South", "San Antonio": "South",
    "Chicago": "Midwest", "Detroit": "Midwest", "Cleveland": "Midwest", "St. Louis": "Midwest", "Milwaukee": "Midwest", "Minneapolis": "Midwest",
    "Los Angeles": "West", "Phoenix": "West", "San Diego": "West", "San Jose": "West", "Oakland": "West", "Dallas": "South",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI options for cities and target sample size per city."""
    parser = argparse.ArgumentParser(description="Fetch and sample ACS tracts for LinguisticRedline.")
    parser.add_argument("--cities", nargs="*", default=None, help="Override city list.")
    parser.add_argument("--sample-per-city", type=int, default=None, help="Sample target per city (default from config or 200).")
    return parser.parse_args()


def request_with_retry(url: str, params: Dict[str, str], attempts: int = 3) -> List[List[str]]:
    """Request Census endpoint JSON with retries."""
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception:
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable")


def fetch_city_tracts(city: str, year: int, api_key: str) -> pd.DataFrame:
    """Fetch all tracts for configured city county definitions."""
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    all_frames: list[pd.DataFrame] = []
    for loc in CITY_COUNTIES[city]:
        raw = request_with_retry(base_url, {"get": ",".join(ACS_VARIABLES), "for": "tract:*", "in": f"state:{loc['state']} county:{loc['county']}", "key": api_key})
        city_df = pd.DataFrame(raw[1:], columns=raw[0])
        city_df["city"] = city
        all_frames.append(city_df)
    return pd.concat(all_frames, ignore_index=True)


def transform(df: pd.DataFrame, min_population: int) -> pd.DataFrame:
    """Convert ACS columns and derive race, income, vacancy, and region metadata."""
    df = df.rename(columns={
        "B02001_002E": "white_pop", "B02001_003E": "black_pop", "B02001_005E": "asian_pop", "B03001_003E": "hispanic_pop",
        "B01003_001E": "total_population", "B19013_001E": "income", "B25002_003E": "vacant_units", "B25001_001E": "total_units",
    }).copy()
    for c in ["white_pop", "black_pop", "asian_pop", "hispanic_pop", "total_population", "income", "vacant_units", "total_units"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["tract_fips"] = df["state"] + df["county"] + df["tract"]
    denom = df["total_population"].replace(0, pd.NA)
    df["pct_white"] = df["white_pop"] / denom * 100
    df["pct_black"] = df["black_pop"] / denom * 100
    df["pct_hispanic"] = df["hispanic_pop"] / denom * 100
    df["pct_asian"] = df["asian_pop"] / denom * 100
    df["vacancy_rate"] = df["vacant_units"] / df["total_units"].replace(0, pd.NA)
    df["income_bucket"] = pd.qcut(df["income"].fillna(df["income"].median()), q=5, labels=["low", "lower_middle", "middle", "upper_middle", "high"])
    df["dominant_race"] = df[["pct_white", "pct_black", "pct_hispanic", "pct_asian"]].idxmax(axis=1).str.replace("pct_", "", regex=False)
    df["region"] = df["city"].map(CITY_REGION).fillna("Unknown")
    return df[df["total_population"] >= min_population].reset_index(drop=True)


def attach_geometry(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Attach tract geometries and centroid coordinates."""
    frames = []
    for state in sorted(df["state"].unique()):
        tracts = gpd.read_file(f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state}_tract.zip")
        frames.append(tracts[["GEOID", "geometry"]])
    geoms = pd.concat(frames, ignore_index=True)
    gdf = gpd.GeoDataFrame(df.merge(geoms, left_on="tract_fips", right_on="GEOID", how="left"), geometry="geometry", crs="EPSG:4269").to_crs(epsg=4326)
    gdf["centroid_lon"] = gdf.geometry.centroid.x
    gdf["centroid_lat"] = gdf.geometry.centroid.y
    return pd.DataFrame(gdf.drop(columns=["GEOID", "geometry"]))


def stratified_sample(df: pd.DataFrame, sample_per_city: int) -> pd.DataFrame:
    """Stratified sample by race x income x region within city."""
    out = []
    for city, city_df in df.groupby("city"):
        target = min(sample_per_city, len(city_df))
        shares = city_df.groupby(["dominant_race", "income_bucket", "region"]).size()
        alloc = (shares / shares.sum() * target).round().astype(int)
        parts = []
        for stratum, n in alloc.items():
            if n <= 0:
                continue
            a, b, c = stratum
            s = city_df[(city_df["dominant_race"] == a) & (city_df["income_bucket"] == b) & (city_df["region"] == c)]
            parts.append(s.sample(n=min(n, len(s)), random_state=42))
        sampled = pd.concat(parts, ignore_index=False) if parts else city_df.sample(n=target, random_state=42)
        if len(sampled) < target:
            rem = city_df.loc[~city_df.index.isin(sampled.index)]
            sampled = pd.concat([sampled, rem.sample(n=min(target - len(sampled), len(rem)), random_state=42)], ignore_index=False)
        out.append(sampled)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    """Fetch, transform, stratify, and save tract-level ACS dataset."""
    args = parse_args()
    config = load_config()
    api_key = config.get("census_api_key", "")
    if not api_key or api_key.startswith("YOUR_"):
        raise ValueError("config.yaml must contain valid census_api_key")

    default_cities = config.get("cities", list(CITY_COUNTIES)) + config.get("extended_cities", [])
    cities = args.cities if args.cities else list(dict.fromkeys(default_cities))
    sample_per_city = args.sample_per_city or int(config.get("sample_per_city", 200))
    year = int(config.get("census_year", 2022))
    min_population = int(config.get("min_population", 500))

    raw = pd.concat([fetch_city_tracts(city, year, api_key) for city in cities], ignore_index=True)
    transformed = transform(raw, min_population=min_population)
    final = attach_geometry(transformed, year=year)
    sampled = stratified_sample(final, sample_per_city=sample_per_city)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(sampled)} sampled tracts to {OUTPUT_PATH}")
    if len(sampled) < 5000:
        print("[warning] Sample below 5,000 tracts; increase --sample-per-city or city list for EMNLP scale.")


if __name__ == "__main__":
    main()
