#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

def load_dataset(path="apartmentsdata.csv"):
    """Load dataset from CSV."""
    df = pd.read_csv(path)
    return df


def preview_locations(df, count=5):
    """Print first few location values."""
    for i, loc in enumerate(df["location"]):
        print(loc)
        if i == count:
            break

