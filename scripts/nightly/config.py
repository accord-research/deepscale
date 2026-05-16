"""Config loader. Parses scripts/nightly/countries.yml into typed objects."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Season:
    init_months: list[int]
    target_months: list[int]
    season_start_month: int


@dataclass(frozen=True)
class Country:
    bbox: dict[str, float]
    seasons: dict[str, Season]


@dataclass(frozen=True)
class Shared:
    models: list[str]
    observations: str
    predictand_var: str
    hindcast_period: tuple[int, int]
    cv: str
    method: str
    cpt_args: dict


@dataclass(frozen=True)
class Config:
    shared: Shared
    countries: dict[str, Country]


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    if "shared" not in raw or "countries" not in raw:
        raise ValueError("countries.yml must define 'shared' and 'countries'")

    shared_raw = raw["shared"]
    required_shared = (
        "models", "observations", "predictand_var", "hindcast_period",
        "cv", "method", "cpt_args",
    )
    missing = [key for key in required_shared if key not in shared_raw]
    if missing:
        raise ValueError(
            "shared section missing required field(s): " + ", ".join(missing)
        )

    shared = Shared(
        models=list(shared_raw["models"]),
        observations=shared_raw["observations"],
        predictand_var=shared_raw["predictand_var"],
        hindcast_period=tuple(shared_raw["hindcast_period"]),
        cv=shared_raw["cv"],
        method=shared_raw["method"],
        cpt_args=dict(shared_raw["cpt_args"]),
    )

    countries: dict[str, Country] = {}
    for name, c_raw in raw["countries"].items():
        seasons = {
            s_name: Season(
                init_months=list(s_raw["init_months"]),
                target_months=list(s_raw["target_months"]),
                season_start_month=int(s_raw["season_start_month"]),
            )
            for s_name, s_raw in c_raw["seasons"].items()
        }
        countries[name] = Country(bbox=dict(c_raw["bbox"]), seasons=seasons)

    return Config(shared=shared, countries=countries)
