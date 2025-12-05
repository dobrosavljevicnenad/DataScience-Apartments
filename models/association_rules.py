#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules


def run_association_rules(df):
    df_assoc = df.copy()
    df_assoc["price_category"] = pd.qcut(df_assoc["price"], q=3, labels=["cheap", "medium", "expensive"])
    df_assoc["area_cat"] = pd.qcut(df_assoc["area"], q=3, labels=["small", "medium", "large"])

    assoc_data = df_assoc[["city", "area_cat", "price_category"]]
    assoc_encoded = pd.get_dummies(assoc_data)

    frequent_itemsets = apriori(assoc_encoded, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    filtered = rules[
        rules["antecedents"].apply(lambda x: any("city_" in i for i in x) and any("area_cat_" in i for i in x)) &
        rules["consequents"].apply(lambda x: any("price_category_" in i for i in x))
    ].sort_values(by=["lift", "confidence"], ascending=False)

    print(filtered[["antecedents", "consequents", "support", "confidence", "lift"]].head(10))

    return filtered


def plot_association_rules(filtered_rules):
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_rules["support"], filtered_rules["confidence"],
                s=filtered_rules["lift"] * 30, alpha=0.6)

    for i, rule in enumerate(filtered_rules.index):
        plt.text(filtered_rules["support"].iloc[i],
                 filtered_rules["confidence"].iloc[i],
                 f"{list(filtered_rules['antecedents'].iloc[i])} → {list(filtered_rules['consequents'].iloc[i])}",
                 fontsize=8)

    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Association Rules")
    plt.grid(True)
    plt.show()


# In[ ]:




