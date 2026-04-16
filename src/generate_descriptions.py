"""Generate neutral neighborhood descriptions from real census and amenity data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_PATH = Path("data/tracts_with_amenities.csv")
OUTPUT_PATH = Path("data/neighborhood_descriptions.csv")


def racial_sentence(row: pd.Series) -> str:
    """Build a racial composition sentence using dominant-group rules."""
    groups = {
        "White": row["pct_white"],
        "Black": row["pct_black"],
        "Hispanic": row["pct_hispanic"],
        "Asian": row["pct_asian"],
    }
    sorted_groups = sorted(groups.items(), key=lambda x: x[1], reverse=True)
    top_group, top_pct = sorted_groups[0]
    second_group, second_pct = sorted_groups[1]

    if top_pct > 60:
        return f"Approximately {top_pct:.1f}% of residents identify as {top_group}."
    if 40 <= top_pct <= 60:
        return (
            f"The largest racial group is {top_group} at {top_pct:.1f}%, "
            f"with a notable {second_group} population at {second_pct:.1f}%."
        )

    top_three = ", ".join([f"{name} ({pct:.1f}%)" for name, pct in sorted_groups[:3]])
    return f"The neighborhood is racially diverse with no single majority group — {top_three}."


def housing_sentence(vacancy_rate: float) -> str:
    """Describe housing conditions based on vacancy rate."""
    if vacancy_rate >= 0.12:
        return "The housing stock includes many vacant or abandoned properties."
    return "Most housing units are occupied."


def amenity_sentence(bucket: str) -> str:
    """Describe amenity landscape based on amenity bucket."""
    if bucket == "community_rich":
        return "The area has many parks, schools, groceries, and local dining options."
    if bucket == "financially_underserved":
        return "The area has limited community amenities and a heavier presence of high-cost financial storefronts."
    return "The area has a mixed commercial profile with both services and everyday retail options."


def build_description(row: pd.Series) -> str:
    """Compose full natural-language neighborhood description for a tract."""
    return (
        f"Neighborhood {row['tract_fips']} is an urban neighborhood. "
        f"The population is approximately {int(row['total_population']):,} residents. "
        f"{racial_sentence(row)} "
        f"The median household income is ${int(row['income']):,}. "
        f"{housing_sentence(float(row['vacancy_rate']))} "
        f"{amenity_sentence(row['amenity_bucket'])}"
    )


def main() -> None:
    """Generate tract descriptions and save the downstream-compatible dataset."""
    df = pd.read_csv(INPUT_PATH)
    df = df.copy()
    df["description"] = df.apply(build_description, axis=1)
    df["description_version"] = "real_data_v1"

    out = pd.DataFrame(
        {
            "id": range(1, len(df) + 1),
            "tract_fips": df["tract_fips"],
            "description": df["description"],
            "dominant_race": df["dominant_race"],
            "income": df["income"],
            "income_bucket": df["income_bucket"],
            "pct_white": df["pct_white"],
            "pct_black": df["pct_black"],
            "pct_hispanic": df["pct_hispanic"],
            "pct_asian": df["pct_asian"],
            "vacancy_rate": df["vacancy_rate"],
            "amenity_bucket": df["amenity_bucket"],
            "city": df["city"],
            "description_version": df["description_version"],
            "centroid_lat": df.get("centroid_lat"),
            "centroid_lon": df.get("centroid_lon"),
        }
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(out)} descriptions to {OUTPUT_PATH}")
    print("\nPreview descriptions:")
    print(out[["tract_fips", "description"]].sample(min(5, len(out)), random_state=42).to_string(index=False))


if __name__ == "__main__":
    main()
