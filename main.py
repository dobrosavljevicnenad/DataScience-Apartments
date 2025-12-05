import pandas as pd

from preprocessing.preprocessing import load_and_clean_dataset
from eda.eda import run_basic_eda
from pca.pca_analysis import run_pca_analysis
from models.model_training import train_models, plot_model_comparison, plot_residuals
from models.knn_modeling import train_knn_classifier, plot_city_avg, plot_city_price_distribution
from models.association_rules import run_association_rules, plot_association_rules
from models.svm_modeling import train_svm_simple, train_svm_pipeline, plot_svm_heatmap


def main():

    print("\nLOADING & CLEANING DATASET")
    df = load_and_clean_dataset("data/apartmentsdata.csv")

    print("\nRUNNING EDA")
    run_basic_eda(df)

    print("\nPCA ANALYSIS")
    run_pca_analysis(df)

    print("\nTRAINING RF & MLP MODELS")
    rf, nn, X_test, y_test, y_pred_rf, y_pred_nn = train_models(
    df,
    categorical_cols=["city", "municipality", "neighborhood"],
    numerical_cols=["area", "floor", "rooms"]
    )

    plot_model_comparison(y_test, y_pred_rf, y_pred_nn)
    plot_residuals(y_test, y_pred_rf, y_pred_nn)


    print("\nTRAINING KNN CLASSIFIER")
    knn_model, df_knn = train_knn_classifier(df)
    plot_city_avg(df_knn)
    plot_city_price_distribution(df_knn)

    print("\nASSOCIATION RULE MINING")
    rules = run_association_rules(df)
    plot_association_rules(rules)

    # 7. SVM (two approaches)
    print("\nTRAINING SIMPLE SVM")
    
    print("\nTRAINING SVM PIPELINE")
    svm_model, X_test, y_test, y_pred = train_svm_pipeline(df)

    print("\nSVM PERFORMANCE HEATMAP")
    plot_svm_heatmap(y_test, y_pred)

    print("\nALL TASKS COMPLETED")


if __name__ == "__main__":
    main()





