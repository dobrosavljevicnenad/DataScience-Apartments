#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------
# BOX PLOTS
# ---------------------

def boxplot_numeric(df, numeric_cols):
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x=df[col], color='skyblue')
        plt.title(col)
    plt.tight_layout()
    plt.show()


# ---------------------
# OUTLIER STATS
# ---------------------

def print_outlier_stats(df, numeric_cols):
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = (df[col] < lower).sum() + (df[col] > upper).sum()

        print(f"Column: {col}")
        print(f"  Q1 = {Q1:.2f}")
        print(f"  Q3 = {Q3:.2f}")
        print(f"  IQR = {IQR:.2f}")
        print(f"  Lower bound = {lower:.2f}")
        print(f"  Upper bound = {upper:.2f}")
        print(f"  Outliers = {outliers}\n")


# ---------------------
# AREA OUTLIERS
# ---------------------

def plot_large_areas(df):
    df_filtered = df[df["area"] > 250]

    counts = df_filtered["neighborhood"].value_counts()
    max_area = df_filtered.groupby("neighborhood")["area"].max()

    combined = pd.DataFrame({
        "count": counts,
        "max_area": max_area
    }).sort_values("max_area", ascending=False)

    labels = [f"{n} ({int(a)} m2)" for n, a in zip(combined.index, combined["max_area"])]

    plt.figure(figsize=(10, 8))
    plt.bar(labels, combined["count"])
    plt.xticks(rotation=90)
    plt.xlabel("Neighborhood (Max Area)")
    plt.ylabel("Properties > 250m2")
    plt.suptitle("Large Properties by Neighborhood", fontsize=30)
    plt.figtext(0.5, -0.05, 
                "Because these are large neighborhoods, higher area values are normal.",
                ha="center")
    plt.tight_layout()
    plt.show()


# ---------------------
# PRICE OUTLIERS
# ---------------------

def plot_large_prices(df):
    df_filtered = df[df["price"] > 700000]

    counts = df_filtered["neighborhood"].value_counts()
    max_price = df_filtered.groupby("neighborhood")["price"].max()

    combined = pd.DataFrame({
        "count": counts,
        "max_price": max_price
    }).sort_values("max_price", ascending=False)

    labels = [f"{n} ({int(p):,}€)" for n, p in zip(combined.index, combined["max_price"])]

    plt.figure(figsize=(12, 8))
    plt.bar(labels, combined["count"], color="#4c72b0")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Frequency > 700,000€")
    plt.title("Price Outliers")

    plt.figtext(0.5, -0.07,
                "Novo Naselje appears expensive but has 230 m², so the price is reasonable.",
                ha="center")
    plt.tight_layout()
    plt.show()


# ---------------------
# FLOORS OVER 15
# ---------------------

def plot_high_floors(df):
    counts = df[df["floor"] > 15]["neighborhood"].value_counts()

    plt.figure(figsize=(10, 5))
    plt.bar(counts.index, counts.values)
    plt.xticks(rotation=90)
    plt.title("Properties with Floor > 15")
    plt.show()


# ---------------------
# COLUMN HISTOGRAM
# ---------------------

def hist_col(df, col):
    plt.figure(figsize=(10, 8))

    _, bins, _ = plt.hist(df[col], bins=50, edgecolor="black")

    centers = 0.5 * (bins[1:] + bins[:-1])
    plt.xticks(centers, [f"{int(x):,}" for x in centers], rotation=90)

    plt.title(col, fontsize=20)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------
# BARPLOTS
# ---------------------

def plot_neighborhood_counts(df):
    counts = df["neighborhood"].value_counts()

    plt.figure(figsize=(30, 12))
    sns.barplot(x=counts.index, y=counts.values, color="skyblue", edgecolor="black")
    plt.xticks(rotation=90)
    plt.title("Number of Properties per Neighborhood")
    plt.show()


def plot_avg_price_city(df):
    avg_price = df.groupby("city")["price"].mean().sort_values(ascending=False)

    plt.figure(figsize=(20, 10))
    sns.barplot(x=avg_price.index, y=avg_price.values)
    plt.xticks(rotation=90)
    plt.title("Average Price per City")
    plt.show()


def plot_avg_area_city(df):
    avg_area = df.groupby("city")["area"].mean().sort_values(ascending=False)

    plt.figure(figsize=(20, 10))
    sns.barplot(x=avg_area.index, y=avg_area.values)
    plt.xticks(rotation=90)
    plt.title("Average Area per City")
    plt.show()


def corr_heatmap(df):
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def run_basic_eda(df):
    """Run basic exploratory data analysis on the dataset."""

    print("→ Generating boxplots...")
    numeric_cols = ['area', 'floor', 'price', 'rooms', 'square_price']
    boxplot_numeric(df, numeric_cols)

    print("→ Printing outlier statistics...")
    print_outlier_stats(df, numeric_cols)

    print("→ Plotting large-area properties...")
    plot_large_areas(df)

    print("→ Plotting high-price properties...")
    plot_large_prices(df)

    print("→ Plotting properties with floor > 15...")
    plot_high_floors(df)

    print("→ Plotting histograms for price and area...")
    hist_col(df, "price")
    hist_col(df, "area")

    print("→ Plotting neighborhood counts...")
    plot_neighborhood_counts(df)

    print("→ Plotting average price per city...")
    plot_avg_price_city(df)

    print("→ Plotting average area per city...")
    plot_avg_area_city(df)

    print("→ Plotting correlation heatmap...")
    corr_heatmap(df)

    print("=== EDA COMPLETED ===")


# In[ ]:




