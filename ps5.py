# ps5.py
# Completed by:
import csv
from typing import Callable
from math import ceil
from pathlib import Path

# Read a CSV file
# We assume it is a CSV file with a header row and two columns
# Created with assistance from GitHub Copilot
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

# Find the mean of a list of numbers
def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)

# Find the covariance of two lists of numbers
def covariance(xs: list[float], ys: list[float]) -> float:
    x_mean = mean(xs)
    y_mean = mean(ys)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / len(xs)

# Find the variance of a list of numbers
def variance(xs: list[float]) -> float:
    x_mean = mean(xs)
    return sum((x - x_mean) ** 2 for x in xs) / len(xs)

# Find the slope of a line of best fit
def slope(xs: list[float], ys: list[float]) -> float:
    return covariance(xs, ys) / variance(xs)

# Find the y-intercept of a line of best fit
def intercept(xs: list[float], ys: list[float]) -> float:
    return mean(ys) - slope(xs, ys) * mean(xs)

# Find the y-value of a line of best fit at a given x-value
def predict(xs: list[float], ys: list[float], x: float) -> float:
    return slope(xs, ys) * x + intercept(xs, ys)

# Find the r-squared value of a line of best fit
def r_squared(xs: list[float], ys: list[float]) -> float:
    y_mean = mean(ys)
    return 1 - sum((y - predict(xs, ys, x)) ** 2 for x, y in zip(xs, ys)) / sum((y - y_mean) ** 2 for y in ys)

# Plot a line of best fit
# Requires matplotlib (pip or pip3 install matplotlib)
def plot(xs: list[float], ys: list[float]) -> None:
    import matplotlib.pyplot as plt
    plt.scatter(xs, ys)
    plt.plot(xs, [predict(xs, ys, x) for x in xs])
    plt.show()

if __name__ == '__main__':
    p = Path(__file__).with_name('SingleFamilyLessThan1M.csv')
    xs, ys = readCSV(p.absolute())
    print(f'slope: {slope(xs, ys)}')
    print(f'intercept: {intercept(xs, ys)}')
    print(f'y = {slope(xs, ys)}x + {intercept(xs, ys)}')
    print(f'r-squared: {r_squared(xs, ys)}')
    plot(xs, ys)
