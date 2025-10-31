# ps5.py
# Completed by:
import csv
from typing import Callable
from math import ceil
from pathlib import Path

def readCSV(file_path: Path) -> tuple[list[float], list[float]]:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        xs = []
        ys = []
        for row in reader:
            xs.append(float(row[0]))
            ys.append(float(row[1]))
        return (xs, ys)

def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)

def variance(xs: list[float]) -> float:
    x_mean = mean(xs)
    return sum((x - x_mean) ** 2 for x in xs) / len(xs)

def stddev(xs: list[float]) -> float:
    return variance(xs) ** 0.5

def covariance(xs: list[float], ys: list[float]) -> float:
    x_mean = mean(xs)
    y_mean = mean(ys)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / len(xs)

def slope(xs: list[float], ys: list[float]) -> float:
    return covariance(xs, ys) / variance(xs)

def intercept(xs: list[float], ys: list[float]) -> float:
    return mean(ys) - slope(xs, ys) * mean(xs)

def predict(xs: list[float], ys: list[float], x: float) -> float:
    return slope(xs, ys) * x + intercept(xs, ys)

def r_squared(xs: list[float], ys: list[float]) -> float:
    y_mean = mean(ys)
    return 1 - sum((y - predict(xs, ys, x)) ** 2 for x, y in zip(xs, ys)) / sum((y - y_mean) ** 2 for y in ys)

def plot(xs: list[float], ys: list[float]) -> None:
    import matplotlib.pyplot as plt
    plt.scatter(xs, ys)
    plt.plot(xs, [predict(xs, ys, x) for x in xs])
    plt.show()

def clean_outliers(xs: list[float], ys: list[float], band_width: int = 500) -> tuple[list[float], list[float]]:
    cleaned_xs = []
    cleaned_ys = []
    min_sf = min(xs)
    max_sf = max(xs)
    # Loop over bands
    band_start = 0
    while band_start < max_sf:
        band_end = band_start + band_width
        # Indices in this band
        indices_in_band = [i for i, x in enumerate(xs) if band_start < x <= band_end]
        if len(indices_in_band) == 0:
            band_start = band_end
            continue
        band_prices = [ys[i] for i in indices_in_band]
        band_mean = mean(band_prices)
        band_std = stddev(band_prices)
        lower = band_mean - 2 * band_std
        upper = band_mean + 2 * band_std
        # Only include non-outliers in this band
        for i in indices_in_band:
            if lower <= ys[i] <= upper:
                cleaned_xs.append(xs[i])
                cleaned_ys.append(ys[i])
        band_start = band_end
    return cleaned_xs, cleaned_ys

if __name__ == '__main__':
    p = Path(__file__).with_name('SingleFamilyLessThan1M.csv')
    xs, ys = readCSV(p.absolute())
    # Remove outliers in bands of 500 finished square feet
    cleaned_xs, cleaned_ys = clean_outliers(xs, ys, band_width=500)
    # Compute improved r-squared
    if len(cleaned_xs) > 1:
        r2_cleaned = r_squared(cleaned_xs, cleaned_ys)
        print(f'Improved r-squared after outlier removal: {round(r2_cleaned, 3)}')
    else:
        print("Not enough data points after cleaning to compute r-squared.")
