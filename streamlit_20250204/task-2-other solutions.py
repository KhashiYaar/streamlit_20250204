# with tab3:
#     st.subheader("Data explorer")
#     st.write("This is the complete dataset:")
#     selected_year = st.slider("Select a year", min_value=min_year, max_value=max_year, value=min_year)
#     # Country multiselect
#     selected_countries = st.multiselect("Select countries", unique_countries, default=unique_countries[:3])  # Default selects first 3
#     # Filter dataset based on selected year and countries
#     filtered_df = df[(df["year"] == selected_year) & (df["country"].isin(selected_countries))]
#     st.dataframe(filtered_df)
#     # Convert filtered DataFrame to CSV
#     csv = filtered_df.to_csv(index=False).encode("utf-8")
#     # Download button
#     st.download_button(
#         label="Download CSV",
#         data=csv,
#         file_name=f"filtered_data_{selected_year}.csv",
#         mime="text/csv"
#     )


# 11:05
# with tab3:
#     st.subheader("Data explorer")
#     st.write("This is the complete dataset:")
#     selected_year = st.slider("Select a year", min_value=min_year, max_value=max_year, value=min_year)
#     # Country multiselect
#     selected_countries = st.multiselect("Select countries", unique_countries, default=unique_countries[:3])  # Default selects first 3
#     # Filter dataset based on selected year and countries
#     filtered_df = df[(df["year"] == selected_year) & (df["country"].isin(selected_countries))]
#     st.dataframe(filtered_df)
#     # Convert filtered DataFrame to CSV
#     csv = filtered_df.to_csv(index=False).encode("utf-8")
#     # Download button
#     st.download_button(
#         label="Download CSV",
#         data=csv,
#         file_name=f"filtered_data_{selected_year}.csv",
#         mime="text/csv"
#     )
# New
# 11:08
# # min_year, max_year = int(df["year"].min()), int(df["year"].max())
