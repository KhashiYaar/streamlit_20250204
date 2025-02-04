import plotly.express as px
import streamlit as st


def scatter_gdp_vs_life_expectancy(df):
    """Creates a scatter plot of GDP per capita vs Life Expectancy"""

    # Ensure required columns exist
    required_cols = ["GDP per capita", "Life Expectancy (IHME)", "country"]
    if not all(col in df.columns for col in required_cols):
        st.error("🚨 Missing necessary columns for the scatter plot!")
        return

    # Create scatter plot
    fig = px.scatter(
        df,
        x="GDP per capita",
        y="Life Expectancy (IHME)",
        hover_name="country",
        size="GDP per capita",
        color="country",
        log_x=True,  # Use log scale for GDP
        title="📊 Life Expectancy vs GDP per Capita",
        labels={
            "GDP per capita": "GDP per Capita (log)",
            "Life Expectancy (IHME)": "Life Expectancy (years)",
        },
    )

    # Display plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
