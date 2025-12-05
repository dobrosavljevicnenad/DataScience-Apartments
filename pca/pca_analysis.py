#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def prepare_pca(df):
    """Prepare dataset for PCA: add price_category and scale numeric features."""
    df_pca = df.copy()
    df_pca["price_category"] = pd.qcut(df_pca["price"], q=3, labels=["cheap", "medium", "expensive"])

    X = df_pca[["area", "floor", "rooms", "square_price"]]
    y = df_pca["price_category"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def run_pca(X_scaled, n_components=2):
    """Run PCA and return transformed matrix."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    return X_pca, pca


def plot_pca(X_pca, y):
    """Plot PCA 2D scatter with categories."""
    plt.figure(figsize=(10, 6))
    colors = {"cheap": "green", "medium": "orange", "expensive": "red"}

    for cat in y.unique():
        subset = X_pca[y == cat]
        plt.scatter(subset[:, 0], subset[:, 1], c=colors[cat], label=cat, alpha=0.6)

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Properties with 2 dimensions")
    plt.legend()

    plt.figtext(
        0.5, -0.05,
        "PCA shows price trends, but class boundaries overlap.",
        ha="center", fontsize=10
    )
    plt.tight_layout()
    plt.show()

def run_pca_analysis(df):
    """Full PCA analysis pipeline: prepare → run PCA → plot."""
    
    print("→ Preparing data for PCA...")
    X_scaled, y = prepare_pca(df)

    print("→ Running PCA...")
    X_pca, pca_model = run_pca(X_scaled, n_components=2)

    print("→ Plotting PCA results...")
    plot_pca(X_pca, y)

    print("=== PCA ANALYSIS COMPLETED ===")
    
    # If other modules need PCA results:
    return X_scaled, y, X_pca, pca_model


# In[ ]:




