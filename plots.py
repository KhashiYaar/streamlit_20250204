import plotly.express as px
import streamlit as st


def scatter_gdp_vs_life_expectancy(df):
    """Creates a scatter plot of GDP per capita vs Life Expectancy"""

    # Ensure required columns exist
    required_cols = ["gdp_per_capita", "life_expectancy", "country"]
    if not all(col in df.columns for col in required_cols):
        st.error("🚨 Missing necessary columns for the scatter plot!")
        return None

    # Create scatter plot
    fig = px.scatter(
        df,
        x="gdp_per_capita",
        y="life_expectancy",
        hover_name="country",
        size="gdp_per_capita",
        color="gdp_per_capita",
        log_x=True,  # Use log scale for GDP
        title="📊 Life Expectancy vs GDP per Capita",
        labels={
            "gdp_per_capita": "GDP per Capita (log)",
            "life_expectancy": "Life Expectancy (years)",
        },
    )

    # Display plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)
