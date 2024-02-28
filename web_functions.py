"""This module contains necessary function needed"""

# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder as lb

@st.cache_data()
def load_data():
    """This function returns the preprocessed data"""

    # Load the Diabetes dataset into DataFrame.
    df = pd.read_csv('lungs.csv')

    df['Gender'] = lb().fit_transform(df['Gender'])
    df["Fever"] = lb().fit_transform(df["Fever"])
    df["Cough"] = lb().fit_transform(df["Cough"])
    df["Shortness_of_Breath"] = lb().fit_transform(df["Shortness_of_Breath"])
    df["Fatigue"] = lb().fit_transform(df["Fatigue"])
    df["Chest_Pain"] = lb().fit_transform(df["Chest_Pain"])
    df["Smoking"] = lb().fit_transform(df["Smoking"])
    df["Air_pollution"] = lb().fit_transform(df["Air_pollution"])
    df["Chemical_fumes"] = lb().fit_transform(df["Chemical_fumes"])
    df["Malnutrition_weakness"] = lb().fit_transform(df["Malnutrition_weakness"])
    df["Chronic_lung_disease"] = lb().fit_transform(df["Chronic_lung_disease"])
    df["Pneumonia_Diagnosis"] = lb().fit_transform(df["Pneumonia_Diagnosis"])

    
    # Perform feature and target split
    X = df[["Age","Rpm","Gender","Fever","Cough","Shortness_of_Breath","Fatigue","Chest_Pain","Smoking","Air_pollution","Chemical_fumes","Malnutrition_weakness","Chronic_lung_disease"]]
    y = df['Pneumonia_Diagnosis']
    X=np.array(X)
    y=np.array(y)

    return df, X, y

@st.cache_data()
def train_model(X, y):
    """This function trains the model and return the model and model score"""
    # Create the model
    model = RandomForestClassifier (n_estimators=50, random_state=42)

    # Fit the data on model
    model.fit(X, y)
    # Get the model score
    score = model.score(X, y)
    # Return the values
    return model, score

def predict(X, y, features):
    # Get model and model score
    model, score = train_model(X, y)
    # Predict the value
    prediction = model.predict(np.array(features).reshape(1, -1))

    return prediction, score
