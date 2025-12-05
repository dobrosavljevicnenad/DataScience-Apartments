#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output


def svm_interactive_ui(df, svm_model):
    df = df.copy()
    df["price_category"] = pd.qcut(df["price"], q=3, labels=["cheap", "medium", "expensive"])

    city_dropdown = widgets.Dropdown(
        options=sorted(df["city"].unique()),
        description="City:"
    )

    municipality_dropdown = widgets.Dropdown(description="Municipality:")
    neighborhood_dropdown = widgets.Dropdown(description="Neighborhood:")

    def update_municipalities(*args):
        city = city_dropdown.value
        municipalities = sorted(df[df["city"] == city]["municipality"].unique())
        municipality_dropdown.options = municipalities
        if municipalities:
            municipality_dropdown.value = municipalities[0]
        update_neighborhoods()

    def update_neighborhoods(*args):
        municipality = municipality_dropdown.value
        neighborhoods = sorted(df[df["municipality"] == municipality]["neighborhood"].unique())
        neighborhood_dropdown.options = neighborhoods
        if neighborhoods:
            neighborhood_dropdown.value = neighborhoods[0]

    city_dropdown.observe(update_municipalities, "value")
    municipality_dropdown.observe(update_neighborhoods, "value")

    update_municipalities()

    area_input = widgets.FloatText(description="Area (m²):")
    floor_input = widgets.FloatText(description="Floor:")
    rooms_input = widgets.FloatText(description="Rooms:")

    predict_button = widgets.Button(
        description="Predict Price Category",
        button_style="info"
    )

    output = widgets.Output()

    def predict(b):
        with output:
            clear_output()

            user = pd.DataFrame([{
                "city": city_dropdown.value,
                "municipality": municipality_dropdown.value,
                "neighborhood": neighborhood_dropdown.value,
                "area": area_input.value,
                "floor": floor_input.value,
                "rooms": rooms_input.value
            }])

            pred = svm_model.predict(user)[0]
            conf = svm_model.predict_proba(user).max() * 100

            print(f"Predicted category: {pred.upper()}")
            print(f"Confidence: {conf:.2f}%")

    predict_button.on_click(predict)

    display(
        widgets.VBox([
            city_dropdown,
            municipality_dropdown,
            neighborhood_dropdown,
            area_input,
            floor_input,
            rooms_input,
            predict_button,
            output
        ])
    )


# In[ ]:




