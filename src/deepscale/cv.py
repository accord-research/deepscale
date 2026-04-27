def loyo(years, window=1):
    """Leave-years-out CV. Yields (train_years, test_year) for each year.

    window: number of years to leave out (1=strict LOO, 3=target±1, etc.)
    """
    years = list(years)
    hcw = (window - 1) // 2
    for i, test_year in enumerate(years):
        train_years = [y for j, y in enumerate(years) if abs(j - i) > hcw]
        yield train_years, test_year


def get_cv(name):
    if name == "loyo":
        return loyo
    raise KeyError(f"Unknown CV scheme: {name}")
