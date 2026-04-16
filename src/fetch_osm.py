"""Fetch tract-level amenity counts from OpenStreetMap using Overpass API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from tqdm import tqdm

from config_loader import load_config

INPUT_PATH = Path("data/census_tracts.csv")
OUTPUT_PATH = Path("data/tracts_with_amenities.csv")
SKIPPED_PATH = Path("data/osm_skipped_tracts.csv")
OVERPASS_URL = "http://overpass-api.de/api/interpreter"
REQUEST_SLEEP_SECONDS = 0.5
REQUEST_TIMEOUT_SECONDS = 10
DEFAULT_MAX_WORKERS = 5

TAGS = {
    "amenity": ["restaurant", "cafe", "bar", "nightclub", "pharmacy", "school"],
    "shop": ["alcohol", "money_lender", "supermarket", "convenience"],
    "leisure": ["park"],
}


def build_bbox(centroid_lat: float, centroid_lon: float, buffer_deg: float = 0.01) -> Tuple[float, float, float, float]:
    """Create a bbox tuple (min_lat, min_lon, max_lat, max_lon) around a centroid."""
    return (
        centroid_lat - buffer_deg,
        centroid_lon - buffer_deg,
        centroid_lat + buffer_deg,
        centroid_lon + buffer_deg,
    )


def build_overpass_query(bbox: Tuple[float, float, float, float], tags: Dict[str, List[str]]) -> str:
    """Construct Overpass QL query for node/way/relation matches in a bbox."""
    min_lat, min_lon, max_lat, max_lon = bbox
    bbox_text = f"({min_lat},{min_lon},{max_lat},{max_lon})"

    clauses: List[str] = []
    for key, values in tags.items():
        value_filter = "|".join(values)
        clauses.extend(
            [
                f'node["{key}"~"^{value_filter}$"]{bbox_text};',
                f'way["{key}"~"^{value_filter}$"]{bbox_text};',
                f'relation["{key}"~"^{value_filter}$"]{bbox_text};',
            ]
        )

    return "[out:json][timeout:10];(" + "".join(clauses) + ");out tags;"


def empty_counts() -> Dict[str, int]:
    return {
        "restaurants_cafes": 0,
        "bars_nightclubs": 0,
        "liquor_stores": 0,
        "check_cashing_payday": 0,
        "parks_green_spaces": 0,
        "grocery_stores": 0,
        "pharmacies": 0,
        "schools": 0,
    }


def count_amenities_from_elements(elements: Iterable[Dict[str, object]]) -> Dict[str, int]:
    """Count target amenity categories from Overpass elements payload."""
    counts = empty_counts()

    for element in elements:
        tags = element.get("tags", {})
        if not isinstance(tags, dict):
            continue

        amenity = tags.get("amenity", "")
        shop = tags.get("shop", "")
        leisure = tags.get("leisure", "")

        if amenity in {"restaurant", "cafe"}:
            counts["restaurants_cafes"] += 1
        if amenity in {"bar", "nightclub"}:
            counts["bars_nightclubs"] += 1
        if shop == "alcohol":
            counts["liquor_stores"] += 1
        if shop == "money_lender":
            counts["check_cashing_payday"] += 1
        if leisure == "park":
            counts["parks_green_spaces"] += 1
        if shop in {"supermarket", "convenience"}:
            counts["grocery_stores"] += 1
        if amenity == "pharmacy":
            counts["pharmacies"] += 1
        if amenity == "school":
            counts["schools"] += 1

    return counts


def query_overpass(
    bbox: Tuple[float, float, float, float],
    tags: Dict[str, List[str]],
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
) -> Dict[str, int]:
    """Query Overpass for a bbox and return amenity counts."""
    query = build_overpass_query(bbox, tags)
    response = requests.post(
        OVERPASS_URL,
        data={"data": query},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    elements = payload.get("elements", [])
    return count_amenities_from_elements(elements)


def amenity_bucket(row: pd.Series, community_thr: int, underserved_thr: int) -> str:
    """Assign amenity bucket based on a simple positive-minus-negative scoring rule."""
    score = (
        row["parks_green_spaces"]
        + row["restaurants_cafes"]
        + row["schools"]
        + row["grocery_stores"]
        - row["liquor_stores"]
        - row["check_cashing_payday"]
    )
    if score >= community_thr:
        return "community_rich"
    if score <= underserved_thr:
        return "financially_underserved"
    return "commercial_mixed"


def load_or_fetch_counts(
    tract_fips: str,
    centroid_lat: float,
    centroid_lon: float,
    cache_dir: Path,
    resume: bool,
    timeout_seconds: int,
    request_delay: float,
) -> Tuple[str, Optional[Dict[str, int]], Optional[str]]:
    """Load cached counts if available, otherwise fetch from Overpass and cache."""
    cache_path = cache_dir / f"{tract_fips}.json"
    if resume and cache_path.exists():
        return tract_fips, json.loads(cache_path.read_text(encoding="utf-8")), None

    try:
        bbox = build_bbox(centroid_lat, centroid_lon)
        counts = query_overpass(bbox, TAGS, timeout_seconds=timeout_seconds)
        cache_path.write_text(json.dumps(counts), encoding="utf-8")
        time.sleep(request_delay)
        return tract_fips, counts, None
    except Exception as exc:  # includes timeout and any request failure
        return tract_fips, None, str(exc)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch OSM amenities for census tracts")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N tracts")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tracts that already have cached JSON counts",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Fetch OSM amenities per tract, merge with census data, and persist merged output."""
    args = parse_args(argv)
    config = load_config()
    cache_dir = Path(config.get("osm_cache_dir", "data/osm_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    max_workers = int(config.get("osm_max_workers", config.get("yamlosm_max_workers", DEFAULT_MAX_WORKERS)))
    timeout_seconds = int(config.get("osm_timeout_per_tract", REQUEST_TIMEOUT_SECONDS))
    request_delay = float(config.get("osm_request_delay", REQUEST_SLEEP_SECONDS))

    thresholds = config.get("amenity_score_threshold", {})
    community_thr = int(thresholds.get("community_rich", 3))
    underserved_thr = int(thresholds.get("financially_underserved", -1))

    census_df = pd.read_csv(INPUT_PATH)
    if args.limit is not None:
        census_df = census_df.head(args.limit).copy()

    amenity_results: Dict[str, Dict[str, int]] = {}
    skipped_rows: List[Dict[str, str]] = []

    total = len(census_df)
    start_time = time.time()
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                load_or_fetch_counts,
                row.tract_fips,
                row.centroid_lat,
                row.centroid_lon,
                cache_dir,
                args.resume,
                timeout_seconds,
                request_delay,
            ): row.tract_fips
            for row in census_df.itertuples(index=False)
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=total,
            desc="Fetching OSM amenities",
        ):
            tract_fips = futures[future]
            try:
                returned_tract, counts, error = future.result()
                if counts is not None:
                    amenity_results[returned_tract] = counts
                else:
                    skipped_rows.append({"tract_fips": returned_tract, "error": error or "unknown_error"})
            except Exception as exc:
                skipped_rows.append({"tract_fips": tract_fips, "error": str(exc)})

            completed += 1
            if completed % 50 == 0 or completed == total:
                elapsed = time.time() - start_time
                per_tract = elapsed / completed if completed else 0
                remaining = max(total - completed, 0)
                eta_minutes = (per_tract * remaining) / 60
                print(
                    f"Completed {completed}/{total} tracts — estimated time remaining: {eta_minutes:.1f} minutes"
                )

    amenity_rows = []
    for row in census_df.itertuples(index=False):
        counts = amenity_results.get(row.tract_fips, empty_counts())
        counts["tract_fips"] = row.tract_fips
        amenity_rows.append(counts)

    amenity_df = pd.DataFrame(amenity_rows)
    merged = census_df.merge(amenity_df, on="tract_fips", how="left")
    merged["amenity_bucket"] = merged.apply(
        amenity_bucket,
        axis=1,
        community_thr=community_thr,
        underserved_thr=underserved_thr,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    if skipped_rows:
        skipped_df = pd.DataFrame(skipped_rows)
        SKIPPED_PATH.parent.mkdir(parents=True, exist_ok=True)
        skipped_df.to_csv(SKIPPED_PATH, index=False)
        print(f"Skipped {len(skipped_df)} tracts; details written to {SKIPPED_PATH}")

    print(f"Saved tract + amenity dataset to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()