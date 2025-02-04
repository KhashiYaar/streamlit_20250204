# #task 3: deployment: deploy the app on streamlit cloud (see readme: create own github repo with practical.py file and requirements.txt, connect the github to streamlit cloud)
# do later: use link instead of file


import streamlit as st
import pandas as pd


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
# To read csv file directly from a URL:
#  poverty_url = 'https://raw.githubusercontent.com/owid/poverty-data/main/datasets/pip_dataset.csv'
# life_exp_url = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Healthy%20Life%20Expectancy%20-%20IHME/Healthy%20Life%20Expectancy%20-%20IHME.csv"
# gdp_url = 'https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Maddison%20Project%20Database%202020%20(Bolt%20and%20van%20Zanden%20(2020))/Maddison%20Project%20Database%202020%20(Bolt%20and%20van%20Zanden%20(2020)).csv'
# import pandas as pd
df = pd.read_csv(
    "https://raw.githubusercontent.com/JohannaViktor/streamlit_practical/refs/heads/main/global_development_data.csv"
)


# Tab 1 - Global Overview
with tab1:
    st.write("### :earth_americas: Global Overview")
    st.write(
        "This section provides a global perspective on quality of life indicators."
    )

# Tab 2 - Country Deep Dive
with tab2:
    st.write("### :bar_chart: Country Deep Dive")
    st.write("Analyze specific countries in detail.")

# Tab 3 - Data Explorer (Show the dataset)
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

    # **Show Summary Statistics**
    st.write("#### Summary Statistics:")
    st.write(filtered_df.describe())

    # **Make the Filtered Dataset Downloadable**
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode("utf-8")

    csv = convert_df(filtered_df)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv",
    )
