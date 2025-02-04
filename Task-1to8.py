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
import plotly.graph_objects as go


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
# should learn hot ow merge these: but for now, use the ready data from repo, already downloaded in data folder
@st.cache_data
def load_data():
    global_developement_data = "./data/global_development_data.csv"
    return pd.read_csv(global_developement_data)


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

# Create a slider to filter dataset by year
st.sidebar.header("Global Filters")
year_selected = st.sidebar.slider(
    "Select Year",
    int(df["year"].min()),
    int(df["year"].max()),
    int(df["year"].median()),  # Default: Median year
)

### Tab 1 - Global Overview ###
with tab1:
    st.write("### 🌍 Global Overview")
    st.write(
        "This section provides a global perspective on quality of life indicators."
    )

    # 📌 Filter data based on the selected year
    filtered_df = df[df["year"] == year_selected]

    scatter_gdp_vs_life_expectancy(filtered_df)


### Tab 2 - Country Deep Dive ###
# create a filter for the dataset for countries
with tab2:
    st.write("### :bar_chart: Country Deep Dive")
    st.write("Analyze specific countries in detail.")

    selected_countries = st.multiselect(
        "Select Countries",
        options=sorted(df["country"].unique()),
        default=[sorted(df["country"].unique())[0]],
        key="country_deep_dive",
    )

    country_data = (
        df[df["country"].isin(selected_countries)]
        .groupby(["year", "country"], as_index=False)
        .agg({"Life Expectancy (IHME)": "mean", "GDP per capita": "mean"})
    )

    if not country_data.empty:
        fig = go.Figure()
        for country in selected_countries:
            country_specific_data = country_data[country_data["country"] == country]
            fig.add_trace(
                go.Scatter(
                    x=country_specific_data["year"],
                    y=country_specific_data["Life Expectancy (IHME)"],
                    mode="lines+markers",
                    name=f"Life Expectancy - {country}",
                    line=dict(width=2),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=country_specific_data["year"],
                    y=country_specific_data["GDP per capita"],
                    mode="lines+markers",
                    name=f"GDP per Capita - {country}",
                    line=dict(width=2),
                    yaxis="y2",
                )
            )
        fig.update_layout(
            title=f"Life Expectancy & GDP per Capita in Selected Countries",
            xaxis_title="Year",
            yaxis=dict(title="Life Expectancy (Years)", side="left"),
            yaxis2=dict(
                title="GDP per Capita", overlaying="y", side="right", showgrid=False
            ),
            legend=dict(
                orientation="h",  # Horizontal legend
                x=0,
                y=-0.4,  # Adjusted legend position
                xanchor="left",
                yanchor="bottom",
            ),
            height=500,  # Increase the height of the chart
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data available for the selected countries.")

    country_year_data = country_data[country_data["year"] == year_selected]
    # Histogram of all countries with the selected countries highlighted
    st.write(f"### Histogram of GDP per Capita for All Countries in {year_selected}")

    # Filter data for the selected year
    year_data = df[df["year"] == year_selected]

    # Create histogram for GDP
    fig = px.histogram(
        year_data,
        x="GDP per capita",
        nbins=50,
        title=f"GDP per Capita Distribution in {year_selected}",
        labels={"GDP per capita": "GDP per Capita"},
        color_discrete_sequence=["rgb(200, 85, 59)"],  # Different custom color in RGB
    )

    # Highlight the selected countries
    for country in selected_countries:
        selected_country_gdp = (
            country_year_data[country_year_data["country"] == country][
                "GDP per capita"
            ].values[0]
            if not country_year_data[country_year_data["country"] == country].empty
            else None
        )
        if selected_country_gdp:
            fig.add_vline(
                x=selected_country_gdp,
                line_dash="solid",
                line_color="yellow",
                line_width=3,
                annotation_text=f"{country}",
                annotation_position="top right",
            )

    st.plotly_chart(fig, use_container_width=True)
    if not country_year_data.empty:
        st.write(f"### Statistics for Selected Countries in {year_selected}")
        st.write(country_year_data)
    else:
        st.warning(f"No data available for the selected countries in {year_selected}.")

    ####
    # Create histogram for life expectancy
    fig = px.histogram(
        year_data,
        x="Life Expectancy (IHME)",
        nbins=50,
        title=f"Life Expectancy (IHME) Distribution in {year_selected}",
        labels={"Life Expectancy (IHME)": "Life Expectancy (IHME)"},
        color_discrete_sequence=["rgb(110, 85, 59)"],  # Different custom color in RGB
    )

    # Highlight the selected countries
    for country in selected_countries:
        selected_country_life_expectancy = (
            country_year_data[country_year_data["country"] == country][
                "Life Expectancy (IHME)"
            ].values[0]
            if not country_year_data[country_year_data["country"] == country].empty
            else None
        )

        if selected_country_life_expectancy:
            fig.add_vline(
                x=selected_country_life_expectancy,
                line_dash="solid",
                line_color="yellow",
                line_width=3,
                annotation_text=f"{country}",
                annotation_position="top right",
            )

    st.plotly_chart(fig, use_container_width=True)


### Tab 3 - Data Explorer ###
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

# Tab 4 - model

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
