"""This modules contains data about prediction page"""

# Import necessary modules
import streamlit as st

# Import necessary functions from web_functions
from web_functions import predict


def app(df, X, y):
    """This function create the prediction page"""

    # Add title to the page
    st.title("Prediction Page")

    # Add a brief description
    st.markdown(
        """
            <p style="font-size:25px">
                This app uses <b style="color:green">Random Forest Classifier</b> for the Prediction of Pneumonia Type and Intensity.
            </p>
        """, unsafe_allow_html=True)
    
    # Take feature input from the user
    # Add a subheader
    st.subheader("Select Values:")

    # Take input of features from the user.
    Age = int(st.number_input ('Insert Age'))
    Rpm = int(st.number_input ('Insert a Respiration per Minute'))
    
    col1,col2, col3= st.columns(3)
    with col1:   
        Gender = st.radio("Gender",("***Male***", "***Female***" ))
        if Gender == 'Female':
            Gender = 0
        else :
            Gender = 1
        Fever = st.radio("Fever",("***Yes***", "***No***" ))
        if Fever == 'Yes':
            Fever = 1
        else :
            Fever = 0
        Cough = st.radio("Cough",("***Yes***", "***No***" ))
        if Cough == 'Yes':
            Cough = 1
        else :
            Cough = 0
        Shortness_of_Breath = st.radio("Shortness_of_Breath",("***Yes***", "***No***" ))
        if Shortness_of_Breath == 'Yes':
            Shortness_of_Breath = 1
        else :
            Shortness_of_Breath = 0
        
    with col2:  
        Fatigue = st.radio("Fatigue",("***Yes***", "***No***" ))
        if Fatigue == 'Yes':
            Fatigue = 1
        else :
            Fatigue = 0
        Chest_Pain = st.radio("Chest_Pain",("***Yes***", "***No***" ))
        if Chest_Pain == 'Yes':
            Chest_Pain = 1
        else :
            Chest_Pain = 0
        Smoking = st.radio("Smoking",("***Yes***", "***No***" ))
        if Smoking == 'Yes':
            Smoking = 1
        else :
            Smoking = 0
        Air_pollution = st.radio("Air_pollution",("***Yes***", "***No***" ))
        if Air_pollution == 'Yes':
            Air_pollution = 1
        else :
            Air_pollution = 0
    with col3:
        Chemical_fumes = st.radio("Chemical_fumes",("***Yes***", "***No***" ))
        if Chemical_fumes == 'Yes':
            Chemical_fumes = 1
        else :
            Chemical_fumes = 0
        Malnutrition_weakness = st.radio("Malnutrition_weakness",("***Yes***", "***No***" ))
        if Malnutrition_weakness == 'Yes':
            Malnutrition_weakness = 1
        else :
            Malnutrition_weakness = 0
        Chronic_lung_disease = st.radio("Chronic_lung_disease",("***Yes***", "***No***" ))
        if Chronic_lung_disease == 'Yes':
            Chronic_lung_disease = 1
        else :
            Chronic_lung_disease = 0
        

    # Create a list to store all the features
    features = [Age,Rpm,Gender,Fever,Cough,Shortness_of_Breath,Fatigue,Chest_Pain,Smoking,Air_pollution,Chemical_fumes,Malnutrition_weakness,Chronic_lung_disease]

    # Create a button to predict
    if st.button("Predict"):
        # Get prediction and model score
        prediction, score = predict(X, y, features)
        
     
        if (Age< 60):
            st.info("Follow clinical procedures and recommendations with 600 mg of paracetamol to keep away fever")

        if (Age >60):
            st.info("Immediate attention needed!")

        # Print the output according to the prediction
        if (prediction == 1):
            st.warning("The person has risk of Aspiration Pneumonia")
            st.info("Severity Level 1: This is a nominal pneumonia and gets cured easily.")
            st.success("Smell some Eucalyptus oil and inhale medicated vapour especially with clove oil")

        elif (prediction == 2):
            st.warning("The person has risk of Bacterial Pneumonia")
            st.info("Severity Level 2: This is a common pneumonia and requires some mild doses of medication.")
            st.success("Requires medical attention and nebulizaton and medication courses like antibiotics and antihismatics. Consult a Physician for more details.")

        elif (prediction == 3):
            st.error("The person has high risk of Viral Pneumonia")
            st.info("Severity Level 3: This is a severe pneumonia and needs good medical attention and proper course of medication")
            st.success("Required ventilation / air purifier and stronger doses of antibiotics. However it gets cured faster than bacterial pneumonia.")

        elif (prediction == 4):
            st.error("The person has Fungal Pneumonia")
            st.info("Severity Level :4 This is a chronic pneumonia and is hard to cure if medications and clinical care are not provided in time.")
            st.success("Require Amycline or similar antibiotics and possible chances of being admitted to ICU with ventilation. Requires hospital level treatment.")
       
        # Print teh score of the model 
        st.sidebar.info("The model used is trusted by doctor and has an accuracy of " + str(score*100) + "%")

        st.sidebar.markdown('''<a href="https://www.drugs.com/medical-answers/antibiotics-treat-pneumonia-3121707/
" target="_blank" style="display: inline-block; padding: 12px 20px; background-color: orange; color: white; text-align: center; text-decoration: none; font-size: 16px; border-radius: 10px;">Best Medication for Pneumonia</a>''',unsafe_allow_html=True)