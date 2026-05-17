"""Config loader. Parses scripts/nightly/nightly.yml into typed objects."""
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
    models: list[str]
    observations: str
    predictand_var: str
    hindcast_period: tuple[int, int]
    cv: str
    method: str
    cpt_args: dict
    seasons: dict[str, Season]


@dataclass(frozen=True)
class Config:
    countries: dict[str, Country]


_REQUIRED_COUNTRY_FIELDS = (
    "bbox", "models", "observations", "predictand_var", "hindcast_period",
    "cv", "method", "cpt_args", "seasons",
)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    if "countries" not in raw:
        raise ValueError("nightly.yml must define 'countries'")

    countries: dict[str, Country] = {}
    for name, c_raw in raw["countries"].items():
        missing = [k for k in _REQUIRED_COUNTRY_FIELDS if k not in c_raw]
        if missing:
            raise ValueError(
                f"country {name!r} missing required field(s): "
                + ", ".join(missing)
            )
        seasons = {
            s_name: Season(
                init_months=list(s_raw["init_months"]),
                target_months=list(s_raw["target_months"]),
                season_start_month=int(s_raw["season_start_month"]),
            )
            for s_name, s_raw in c_raw["seasons"].items()
        }
        countries[name] = Country(
            bbox=dict(c_raw["bbox"]),
            models=list(c_raw["models"]),
            observations=c_raw["observations"],
            predictand_var=c_raw["predictand_var"],
            hindcast_period=tuple(c_raw["hindcast_period"]),
            cv=c_raw["cv"],
            method=c_raw["method"],
            cpt_args=dict(c_raw["cpt_args"]),
            seasons=seasons,
        )

    return Config(countries=countries)
