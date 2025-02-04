import streamlit as st

# task 1
# write headline as header "Worldwide Analysis of Quality of Life and Economic Factors"
st.set_page_config(layout="wide")
st.title(":earth_africa: Worldwide Analysis of Quality of Life and Economic Factors")

# write subtitle "This app enables you to explore the relationships between poverty, life expectancy, and GDP across various countries and years. Use the panels to select options and interact with the data."
st.subheader(
    "This app enables you to explore the relationships between poverty, "
    "life expectancy, and GDP across various countries and years. "
    "Use the panels to select options and interact with the data."
)


# #use the whole width of the page

# create 3 tabs called ""Global Overview", "Country Deep Dive", "Data Explorer"

tab1, tab2, tab3 = st.tabs(["Global Overview", "Country Deep Dive", "Data Explorer"])
with tab1:
    st.write("### :earth_americas: Global Overview")
    st.write(
        "This section provides a global perspective on quality of life indicators."
    )
with tab2:
    st.write("### :bar_chart: Country Deep Dive")
    st.write("Analyze specific countries in detail.")
with tab3:
    st.write("### :open_file_folder: Data Explorer")
    st.write("Explore raw data and trends over time.")
