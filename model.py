# Predcitive model task-6
# task 6 in tab 1: create a simple model (conda install scikit-learn -y; Randomforest Regressor): features only 3 columns: ['GDP per capita', 'headcount_ratio_upper_mid_income_povline', 'year']; target: 'Life Expectancy (IHME)'
# you might store the code in an extra model.py file
# make input fields for inference of the features (according to existing values in the dataset) and use the model to predict the life expectancy for the input values
# additional: show the feature importance as a bar plot

# !! hints!!!
# !! need to train the model locally! on app is too heavy!!
# so save:  model.pkl   save in repo folder
# u push this to github, and make it accesible; from there it gets the 3 model feature importance & the mean squarred error;
# and ask for input: for example for the 3 importantn features: GDP per capita; poverty rate; year of prediction (timestamp)
# then u have to train for these, & from there: predict life expectancy

### appraoch:
# made another file for Predictive_models.py , im gonna do my main training there, and call it later from the main
# here are the thigns i should do:
# 1. I wanna find out the top 4 important parameters
# 2.  train and build a predictive model based on these 4,
# 3. output give me the mean squared error and the 4 important parameters
# 4. later include it in tab 4 , so that it shows model feature imporntance boxes: wher eu input the values, and it gives out the predcited life expectancy

import streamlit as st
import pandas as pd
import plotly.express as px
from plots import scatter_gdp_vs_life_expectancy
import numpy as np
import joblib  # For saving/loading the model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def train_model(df):
    df["GDP per capita"] = df["GDP"] / df["Population"]

    # Features and target
    X = df[["GDP per capita", "headcount_ratio_upper_mid_income_povline", "year"]]
    y = df["Life Expectancy (IHME)"]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model trained. MAE: {mae:.2f}")

    # Save the trained model
    joblib.dump(model, "random_forest_model.pkl")

    return model, model.feature_importances_


def load_model():
    """Loads the trained model from file."""
    return joblib.load("random_forest_model.pkl")


def predict_life_expectancy(model, gdp_per_capita, poverty_ratio, year):
    """Uses the trained model to predict Life Expectancy."""
    input_data = np.array([[gdp_per_capita, poverty_ratio, year]])
    return model.predict(input_data)[0]
