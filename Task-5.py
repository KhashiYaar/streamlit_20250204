# task 5 in tab 1: in terminal conda install -c plotly plotly
# create a scatterplot of the dataframe filtered according to the slider: x=GDP per capita, y = Life Expectancy (IHME) with hover, log, size, color, title, labels
# you might store the code in an extra plots.py file


import streamlit as st
import pandas as pd
import plotly.express as px
from plots import scatter_gdp_vs_life_expectancy


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
tab1, tab2, tab3 = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer"])


# Load dataset
@st.cache_data
def load_data():
    global_developement_data = "./data/global_development_data.csv"
    return pd.read_csv(global_developement_data)


# should learn hot ow merge these: but for now, use the ready data from repo, already downloaded in data folder
#  poverty_url = 'https://raw.githubusercontent.com/owid/poverty-data/main/datasets/pip_dataset.csv'
# life_exp_url = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Healthy%20Life%20Expectancy%20-%20IHME/Healthy%20Life%20Expectancy%20-%20IHME.csv"
# gdp_url = 'https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Maddison%20Project%20Database%202020%20(Bolt%20and%20van%20Zanden%20(2020))/Maddison%20Project%20Database%202020%20(Bolt%20and%20van%20Zanden%20(2020)).csv'
# To read csv file directly from a URL

# # Use cached function to load the dataset
df = load_data()

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

    # Scatter plot // call the filtered df here
    scatter_gdp_vs_life_expectancy(filtered_df)

    # # **SLIDER**: Select Year
    # min_year, max_year = int(df["year"].min()), int(df["year"].max())
    # selected_year = st.slider("Select a Year", min_year, max_year, max_year)

    # # **Filter Data** for selected year
    # filtered_df = df[df["year"] == selected_year]

    # **4 Key Metrics in 4 Columns**
    # col1, col2, col3, col4 = st.columns(4)
    # with col1:
    #     mean_life_exp = filtered_df["Life Expectancy (IHME)"].mean()
    #     st.metric(label="📈 Mean Life Expectancy", value=f"{mean_life_exp:.2f} years")
    # with col2:
    #     median_gdp = filtered_df["GDP per capita"].median()
    #     st.metric(label="💰 Median GDP per Capita", value=f"${median_gdp:,.2f}")
    # with col3:
    #     mean_poverty_ratio = filtered_df[
    #         "headcount_ratio_upper_mid_income_povline"
    #     ].mean()
    #     st.metric(
    #         label="📉 Mean Poverty Ratio (Upper Mid Income)",
    #         value=f"{mean_poverty_ratio:.2%}",
    #     )
    # with col4:
    #     num_countries = filtered_df["country"].nunique()
    #     st.metric(label="🌍 Number of Countries", value=f"{num_countries}")

    # **Display Filtered Data**
    # st.write("#### Filtered Dataset Preview:")
    # st.dataframe(filtered_df)

    # # **Make the Filtered Dataset Downloadable**
    # @st.cache_data
    # def convert_df(df):
    #     return df.to_csv(index=False).encode("utf-8")

    # csv = convert_df(filtered_df)
    # st.download_button(
    #     label="📥 Download Filtered Data as CSV",
    #     data=csv,
    #     file_name=f"filtered_data_{selected_year}.csv",
    #     mime="text/csv",
    # )

# Tab 2 - Country Deep Dive
with tab2:
    st.write("### :bar_chart: Country Deep Dive")
    st.write("Analyze specific countries in detail.")

# Tab 3 - Data Explorer
with tab3:
    st.write("### :open_file_folder: Data Explorer")
    st.write("Explore raw data and trends over time.")

    # **MULTI-SELECT BOX**: Select Countries
    country_options = df["country"].unique().tolist()
    selected_countries = st.multiselect(
        "Select Countries", country_options, default=country_options[:3]
    )

    # **SLIDER**: Select Year Range
    min_year, max_year = int(df["year"].min()), int(df["year"].max())
    year_range = st.slider(
        "Select Year Range", min_year, max_year, (min_year, max_year)
    )
    # **FILTER DATA** based on selections
    filtered_df = df[
        (df["country"].isin(selected_countries)) & (df["year"].between(*year_range))
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
