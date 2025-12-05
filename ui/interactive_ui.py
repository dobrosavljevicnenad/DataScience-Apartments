#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output


def build_interactive_ui(df, model):
    """Build Jupyter interactive price prediction UI."""

    cities = sorted(df["city"].unique())

    city_dropdown = widgets.Dropdown(
        options=cities,
        description="City:",
        layout=widgets.Layout(width="300px")
    )

    municipality_dropdown = widgets.Dropdown(
        options=[],
        description="Municipality:",
        layout=widgets.Layout(width="300px")
    )

    neighborhood_dropdown = widgets.Dropdown(
        options=[],
        description="Neighborhood:",
        layout=widgets.Layout(width="300px")
    )

    def update_municipalities(change):
        city = change["new"]
        municipalities = sorted(df[df["city"] == city]["municipality"].unique())
        municipality_dropdown.options = municipalities
        if municipalities:
            municipality_dropdown.value = municipalities[0]

    def update_neighborhoods(change):
        municipality = change["new"]
        neighborhoods = sorted(df[df["municipality"] == municipality]["neighborhood"].unique())
        neighborhood_dropdown.options = neighborhoods
        if neighborhoods:
            neighborhood_dropdown.value = neighborhoods[0]

    city_dropdown.observe(update_municipalities, names="value")
    municipality_dropdown.observe(update_neighborhoods, names="value")

    # Initial population
    update_municipalities({"new": cities[0]})

    area_input = widgets.FloatText(description="Area (m²):", value=50, layout=widgets.Layout(width="300px"))
    floor_input = widgets.FloatText(description="Floor:", value=2, layout=widgets.Layout(width="300px"))
    rooms_input = widgets.FloatText(description="Rooms:", value=2, layout=widgets.Layout(width="300px"))

    predict_button = widgets.Button(
        description="Predict Price",
        button_style="success",
        layout=widgets.Layout(width="300px")
    )

    output = widgets.Output()

    def predict_price(b):
        with output:
            clear_output()
            user_input = pd.DataFrame([{
                "city": city_dropdown.value,
                "municipality": municipality_dropdown.value,
                "neighborhood": neighborhood_dropdown.value,
                "area": area_input.value,
                "floor": floor_input.value,
                "rooms": rooms_input.value
            }])
            price = model.predict(user_input)[0]
            print(f"Estimated property price: {round(price):,.0f} €".replace(",", "."))

    predict_button.on_click(predict_price)

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




