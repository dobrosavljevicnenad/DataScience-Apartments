#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report


def train_svm_simple(X_scaled, y):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    svm = SVC(kernel="rbf")
    svm.fit(X_train, y_train)

    y_pred = svm.predict(X_test)
    print(classification_report(y_test, y_pred))

    return svm, X_test, y_test, y_pred


def train_svm_pipeline(df):
    data = df.copy()
    X = data.drop(columns=["price_category"])
    y = data["price_category"]

    num = ["area", "floor", "price", "rooms", "square_price"]
    cat = ["city", "municipality", "neighborhood"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
        ]
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", SVC(kernel="rbf"))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\nSVM Pipeline Results:")
    print(classification_report(y_test, y_pred))

    return pipeline, X_test, y_test, y_pred


def plot_svm_heatmap(y_test, y_pred):
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).iloc[:-1, :3]

    plt.figure(figsize=(8, 4))
    sns.heatmap(df_report, annot=True, cmap="Blues", fmt=".2f")
    plt.title("SVM Classification Heatmap")
    plt.show()


# In[ ]:




