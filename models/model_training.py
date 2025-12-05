#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score


def build_random_forest_pipeline(categorical_cols, numerical_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",
        force_int_remainder_cols=False
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    return model


def build_mlp_pipeline(categorical_cols, numerical_cols):
    preprocessor_nn = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols)
        ],
        remainder="drop"
    )

    nn_model = Pipeline(steps=[
        ("preprocessor", preprocessor_nn),
        ("regressor", MLPRegressor(
            hidden_layer_sizes=(200, 100),
            activation="relu",
            solver="adam",
            max_iter=5000,
            early_stopping=True,
            random_state=42
        ))
    ])

    return nn_model


def train_models(df, categorical_cols, numerical_cols):
    X = df[categorical_cols + numerical_cols]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = build_random_forest_pipeline(categorical_cols, numerical_cols)
    nn = build_mlp_pipeline(categorical_cols, numerical_cols)

    rf.fit(X_train, y_train)
    nn.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    y_pred_nn = nn.predict(X_test)

    print("Comparison RandomForest-a i MLP")
    print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
    print("Random Forest R2:", r2_score(y_test, y_pred_rf))
    print("Neural Network RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_nn)))
    print("Neural Network R2:", r2_score(y_test, y_pred_nn))

    return rf, nn, X_test, y_test, y_pred_rf, y_pred_nn


def plot_model_comparison(y_test, y_pred_rf, y_pred_nn):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_rf, alpha=0.5, label="Random Forest")
    plt.scatter(y_test, y_pred_nn, alpha=0.5, label="MLP Neural Net")

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "k--", lw=2
    )

    plt.xlabel("Real values")
    plt.ylabel("Predicted values")
    plt.title("Comparison of Random Forest and MLP predictions")
    plt.legend()

    plt.figtext(
        0.1, -0.1,
        "Conclusion: Both models underestimate higher values. RF is more stable, MLP less stable.",
        ha="left", wrap=True, fontsize=10
    )

    plt.tight_layout()
    plt.show()


def plot_residuals(y_test, y_pred_rf, y_pred_nn):
    residual_rf = y_test - y_pred_rf
    residual_nn = y_test - y_pred_nn

    plt.hist(residual_rf, bins=30, alpha=0.5, label="Random Forest")
    plt.hist(residual_nn, bins=30, alpha=0.5, label="MLP NN")

    plt.xlabel("Error (y_true - y_pred)")
    plt.ylabel("Count")
    plt.title("Distribution of Errors")
    plt.legend()
    plt.show()


# In[ ]:




