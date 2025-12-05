#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# ---------------------
# LOCATION SPLITTING
# ---------------------

def split_location(df):
    """Split 'location' into municipality, neighborhood and street."""
    location_split = df["location"].str.split(",", expand=True)

    df["opstina"] = location_split[0].str.strip()
    df["naselje"] = location_split[1].str.strip()
    df["ulica"] = location_split[2].str.strip()

    return df


def rename_location_columns(df):
    """Rename Serbian columns to English column names."""
    df.rename(columns={
        "opstina": "municipality",
        "naselje": "neighborhood",
        "ulica": "street"
    }, inplace=True)
    return df


# ---------------------
# COLUMN CLEANUP
# ---------------------

def drop_unneeded_columns(df):
    """Drop unnamed + title + source + location."""
    df.drop(["Unnamed: 9", "Unnamed: 10"], axis=1, inplace=True)
    df.drop(["title", "source"], axis=1, inplace=True)
    df.drop("location", axis=1, inplace=True)
    return df


# ---------------------
# MISSING VALUES
# ---------------------

def fill_missing_neighborhood(df):
    """Fill NaN neighborhood values using municipality."""
    df["neighborhood"] = df["neighborhood"].fillna(df["municipality"])
    return df


# ---------------------
# FLOOR PROCESSING
# ---------------------

def normalize_floor(df):
    """Convert 'p' → 0 and ensure floor is float absolute value."""
    df["floor"] = df["floor"].replace("p", 0)
    df["floor"] = df["floor"].astype(float).abs()
    return df


def impute_floor(df):
    """
    Fill missing 'floor' values:
      - Use neighborhood median when available
      - Otherwise fallback to city median
    """
    city_median_floor = df.groupby("city")["floor"].transform("median")

    def fill_floor(group):
        if group.notna().any():
            return group.fillna(group.median())
        else:
            return group.fillna(city_median_floor[group.index])

    df["floor"] = df.groupby("neighborhood")["floor"].transform(fill_floor)
    return df


# ---------------------
# REMOVE COLUMNS
# ---------------------

def drop_street(df):
    df.drop("street", axis=1, inplace=True)
    return df


# In[ ]:




