"""Pure function: (Config, today) -> list[Target].

For each country.season, expand init_months into (year, month) calendar
dates considering year wrap (init_month > today.month implies previous
calendar year). Drop any inits that are at or after the season's start
month (no point issuing a forecast for a season already underway). Pick
the most recent surviving init. Skip the season if none survive.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .config import Config


@dataclass(frozen=True)
class Target:
    country: str
    season: str
    init_year: int
    init_month: int


def select_targets(cfg: Config, today: date) -> list[Target]:
    out: list[Target] = []
    today_month_start = date(today.year, today.month, 1)
    for c_name, country in cfg.countries.items():
        for s_name, season in country.seasons.items():
            # The target season we'd be forecasting tonight is the next
            # occurrence of season_start_month that has NOT yet begun. If
            # season_start_month is still ahead of us in the current calendar
            # year, target_year = today.year. Otherwise the season has
            # already started this year, so the next instance is next year.
            if season.season_start_month > today.month:
                target_year = today.year
            else:
                target_year = today.year + 1
            season_start = date(target_year, season.season_start_month, 1)

            candidates: list[tuple[int, int]] = []
            for m in season.init_months:
                # Pick the most recent occurrence of init-month m that is
                # strictly before season_start AND not in the future.
                init_year = target_year if m < season.season_start_month else target_year - 1
                init_d = date(init_year, m, 1)
                if init_d >= season_start:
                    continue
                if init_d > today_month_start:
                    continue
                candidates.append((init_year, m))
            if not candidates:
                continue
            init_year, init_month = max(candidates)  # most recent date
            out.append(Target(c_name, s_name, init_year, init_month))
    return out
