"""This modules contains data about visualisation page"""

# Import necessary modules
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
'''from sklearn.metrics import plot_confusion_matrix'''
from sklearn import tree
import streamlit as st


# Import necessary functions from web_functions
from web_functions import train_model

def app(df, X, y):
    """This function create the visualisation page"""
    
    # Remove the warnings
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Set the page title
    st.title("Visualise the Pneumonia Level")

    # Create a checkbox to show correlation heatmap
    if st.checkbox("Show the correlation heatmap"):
        st.subheader("Correlation Heatmap")

        fig = plt.figure(figsize = (10, 6))
        ax = sns.heatmap(df.iloc[:, 1:].corr(), annot = True)   # Creating an object of seaborn axis and storing it in 'ax' variable
        bottom, top = ax.get_ylim()                             # Getting the top and bottom margin limits.
        ax.set_ylim(bottom + 0.5, top - 0.5)                    # Increasing the bottom and decreasing the top margins respectively.
        st.pyplot(fig)

    if st.checkbox("Show Scatter Plot"):
        
        figure, axis = plt.subplots(2, 2,figsize=(15,10))

        sns.scatterplot(ax=axis[0,0],data=df,x='Age',y='Shortness_of_Breath')
        axis[0, 0].set_title("Breathing complexity with respect to age")
  
        sns.scatterplot(ax=axis[0,1],data=df,x='Age',y='Malnutrition_weakness')
        axis[0, 1].set_title("Patient Age vs Malnutrition_weakness")
  
        sns.scatterplot(ax=axis[1, 0],data=df,x='Smoking',y='Chronic_lung_disease')
        axis[1, 0].set_title("Smoking vs Chronic_lung_disease")
  
        sns.scatterplot(ax=axis[1,1],data=df,x='Rpm',y='Age')
        axis[1, 1].set_title("Respiration Per Minute vs Patient Age")
        st.pyplot()

    if st.checkbox("Display Boxplot"):
        fig, ax = plt.subplots(figsize=(15,5))
        df.boxplot([ "Age","Rpm","Gender","Fever","Cough","Shortness_of_Breath","Fatigue","Chest_Pain","Smoking","Air_pollution","Chemical_fumes","Malnutrition_weakness","Chronic_lung_disease","Pneumonia_Diagnosis"],ax=ax)
        st.pyplot()

  

    

    
    
