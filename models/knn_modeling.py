#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


def prepare_price_categories(df):
    df["price_category"] = pd.qcut(df["price"], q=3, labels=["cheap", "medium", "expensive"])
    return df


def train_knn_classifier(df):
    df = prepare_price_categories(df)

    X = df[["area", "city", "floor", "rooms", "municipality", "neighborhood"]]
    y = df["price_category"]

    categorical_cols = ["city", "municipality", "neighborhood"]
    numerical_cols = ["area", "floor", "rooms"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols),
        ],
        remainder="passthrough"
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nKNN Classification Report:")
    print(classification_report(y_test, y_pred))

    return model, df


def plot_city_avg(df):
    city_avg = df.groupby("city")[["area", "price"]].mean().reset_index()
    city_avg["price_category"] = pd.qcut(city_avg["price"], q=3, labels=["cheap", "medium", "expensive"])

    colors = {"cheap": "green", "medium": "orange", "expensive": "red"}

    plt.figure(figsize=(10, 6))
    for cat in colors:
        subset = city_avg[city_avg["price_category"] == cat]
        plt.scatter(subset["area"], subset["price"], color=colors[cat], label=cat, s=100, alpha=0.7)

    plt.title("Average property prices by city")
    plt.xlabel("Average area (m²)")
    plt.ylabel("Average price (€)")
    plt.legend(title="Category")
    plt.show()


def plot_city_price_distribution(df):
    city_avg = df.groupby("city")[["area", "price"]].mean().reset_index()
    city_avg["price_category"] = pd.qcut(city_avg["price"], q=3, labels=["cheap", "medium", "expensive"])

    colors = {"cheap": "green", "medium": "orange", "expensive": "red"}

    plt.figure(figsize=(20, 10))
    for cat in colors:
        subset = city_avg[city_avg["price_category"] == cat]
        plt.scatter(subset.index, subset["price"], color=colors[cat], label=cat, s=120, alpha=0.8)

    plt.xticks(city_avg.index, city_avg["city"], rotation=90, ha="right")
    plt.title("Average city prices")
    plt.ylabel("Average price (€)")
    plt.legend(title="Price category")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# In[ ]:




