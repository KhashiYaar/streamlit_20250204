##task 8 in tab 2: create a filter for the dataset for countries
# show a scatterplot (as line chart): for life expectancy over the year
# additional: create a second line in the chart showing the GDP (extra ax ticks)

import streamlit as st
import pandas as pd
import plotly.express as px
from plots import scatter_gdp_vs_life_expectancy
import numpy as np
import joblib  # For saving/loading the model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from model import train_model, load_model, predict_life_expectancy
# from plotly.graph_objects as go

# Set page layout
st.set_page_config(layout="wide")

# Title and subtitle
st.title(":earth_africa: Worldwide Analysis of Quality of Life and Economic Factors")
st.subheader(
    "This app enables you to explore the relationships between poverty, "
    "life expectancy, and GDP across various countries and years. "
    "Use the panels to select options and interact with the data."
)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["Global Overview", "Country Deep Dive", "Data Explorer", "Predictive Model"]
)


# Load dataset
@st.cache_data
def load_data():
    global_developement_data = "./data/global_development_data.csv"
    return pd.read_csv(global_developement_data)


# should learn hot ow merge these: but for now, use the ready data from repo, already downloaded in data folder

# # Use cached function to load the dataset
df = load_data()


###
# Cache the model and data to prevent reloading/training on every UI change
@st.cache_resource
def get_model():
    """Loads or trains the model once, then caches it."""
    try:
        model = load_model()
        feature_importances = model.feature_importances_
    except:
        model, feature_importances = train_model(df)
    return model, feature_importances


@st.cache_data
def get_feature_importance(feature_importances):
    """Returns cached feature importance values as a DataFrame."""
    feature_names = ["GDP per capita", "Poverty Ratio", "Year"]
    return pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})


# Load cached model
model, feature_importances = get_model()

# Load cached feature importance
feature_importance_df = get_feature_importance(feature_importances)
###


# Tab 1 - Global Overview
with tab1:
    st.write("### 🌍 Global Overview")
    st.write(
        "This section provides a global perspective on quality of life indicators."
    )
    # 📌 Create a slider to filter dataset by year
    year_selected = st.slider(
        "Select Year",
        int(df["year"].min()),
        int(df["year"].max()),
        int(df["year"].median()),  # Default: Median year
    )

    # 📌 Filter data based on the selected year
    filtered_df = df[df["year"] == year_selected]

    scatter_gdp_vs_life_expectancy(filtered_df)

# Tab 2 - Country Deep Dive
with tab2:
    st.write("### :bar_chart: Country Deep Dive")
    st.write("Analyze specific countries in detail.")

    # create a filter for the dataset for countries
    # **MULTI-SELECT BOX**: Choose Countries
    country_options = df["country"].unique().tolist()
    selected_countries2 = st.multiselect(
        "Choose Countries", country_options, default=country_options[:3]
    )

    # **FILTER DATA** based on selected countries
    filtered_df = df[df["country"].isin(selected_countries2)]

    # **Display Filtered Data**
    st.write("#### Filtered Dataset Preview:")
    st.dataframe(filtered_df)

    # **Scatter Plot**: Life Expectancy over the Year for selected countries
    fig = px.line(
        filtered_df,
        x="year",
        y="Life Expectancy (IHME)",
        color="country",
        title="Life Expectancy over the Years",
        labels={"Life Expectancy (IHME)": "Life Expectancy"},  # , "year": "Year"},
    )

    # **Add GDP Line**: GDP over the Year for selected countries
    fig.add_scatter(
        x=filtered_df["year"],
        y=filtered_df["GDP"],
        mode="lines",
        yaxis="y2",
    )

    # **Update Layout**: Add secondary y-axis for GDP
    fig.update_layout(
        yaxis2=dict(
            title="GDP",
            overlaying="y",
            side="right",
        ),
        legend=dict(x=0, y=1.1, orientation="h"),
    )

    st.plotly_chart(fig)


# Tab 3 - Data Explorer
with tab3:
    st.write("### :open_file_folder: Data Explorer")
    st.write("Explore raw data and trends over time.")

    # **MULTI-SELECT BOX**: Select Countries
    country_options = df["country"].unique().tolist()
    selected_countries3 = st.multiselect(
        "Select Countries", country_options, default=country_options[:3]
    )

    # **SLIDER**: Select Year Range
    min_year, max_year = int(df["year"].min()), int(df["year"].max())
    year_range = st.slider(
        "Select Year Range", min_year, max_year, (min_year, max_year)
    )
    # **FILTER DATA** based on selections
    filtered_df = df[
        (df["country"].isin(selected_countries3)) & (df["year"].between(*year_range))
    ]

    # **Display Filtered Data**
    st.write("#### Filtered Dataset Preview:")
    st.dataframe(filtered_df)

    # **Make the Filtered Dataset Downloadable**
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode("utf-8")

    csv = convert_df(filtered_df)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"filtered_data_{year_range}.csv",
        mime="text/csv",
    )


with tab4:
    st.header("Predict Life Expectancy")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        # User Input Fields
        gdp_per_capita = st.slider(
            "Enter GDP per capita",
            min_value=float(df["GDP per capita"].min()),
            max_value=float(df["GDP per capita"].max()),
            value=float(df["GDP per capita"].median()),
        )
        poverty_ratio = st.slider(
            "Enter Poverty Ratio",
            min_value=float(df["headcount_ratio_upper_mid_income_povline"].min()),
            max_value=float(df["headcount_ratio_upper_mid_income_povline"].max()),
            value=float(df["headcount_ratio_upper_mid_income_povline"].median()),
        )
        year = st.slider(
            "Define Year",
            int(df["year"].min()),
            int(df["year"].max()),
            int(df["year"].median()),
        )

        # Predict Life Expectancy
        if st.button("Predict Life Expectancy"):
            prediction = predict_life_expectancy(
                model, gdp_per_capita, poverty_ratio, year
            )
            st.success(f"Predicted Life Expectancy: {prediction:.2f} years")

    with col_right:
        # Feature Importance Plot
        st.subheader("Feature Importance")
        feature_names = ["GDP per capita", "Poverty Ratio", "year"]
        feature_importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": model.feature_importances_}
        )
        fig = px.bar(
            feature_importance_df,
            x="Feature",
            y="Importance",
            title="Feature Importance",
            labels={"Importance": "Relative Importance"},
        )
        st.plotly_chart(fig)


###

###
