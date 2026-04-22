#!/usr/bin/env python3

"""
This script performs linear regression between petal length and sepal length
for each Iris species and generates one plot per species.
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def load_data(file_name):
    """Load the Iris dataset from a CSV file."""
    return pd.read_csv(file_name)


def compute_regression(x, y):
    """Return slope and intercept from linear regression."""
    result = stats.linregress(x, y)
    return result.slope, result.intercept


def plot_regression(data, species_name):
    """
    Create a scatter plot and regression line for a given species.
    """
    subset = data[data["species"] == species_name]

    x = subset["petal_length_cm"]
    y = subset["sepal_length_cm"]

    slope, intercept = compute_regression(x, y)

    # sort x for a clean regression line
    x_sorted = x.sort_values()
    y_line = slope * x_sorted + intercept

    plt.figure()
    plt.scatter(x, y, label="Data")
    plt.plot(x_sorted, y_line, label="Regression line")

    plt.xlabel("Petal length (cm)")
    plt.ylabel("Sepal length (cm)")
    plt.title(f"{species_name} regression")
    plt.legend()

    filename = f"{species_name}_plot.png"
    plt.savefig(filename)
    plt.close()


def main():
    """Main function to run the analysis."""
    df = load_data("iris.csv")

    species_list = df["species"].unique()

    for species in species_list:
        plot_regression(df, species)


if __name__ == "__main__":
    main()
