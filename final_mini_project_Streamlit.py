from ssl import Options
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

st.set_page_config(page_title="Fraud_detection",layout="wide",menu_items={"About":"https://github.com/Vikram2305/Insurance-Fraud-Detection"})
# Load the saved model and scaler
with open(r"fraud_detection_rf_model.pkl", 'rb') as model_file:
    loaded_model = pd.read_pickle(model_file)

# Load the saved model and scaler
with open(r"scaler.pkl", 'rb') as le_file:
    scaler = pd.read_pickle(le_file)

def Fraud_detection(input_data):
    new_data_scaled = scaler.transform(input_data)
    predicted_fraud = loaded_model.predict(new_data_scaled)
    predicted_probabilities = loaded_model.predict_proba(new_data_scaled)
    confidence_level = max(predicted_probabilities[0])
    color = "green" if predicted_fraud[0] == 0 else "red"
    result = "Genuine" if predicted_fraud[0] == 0 else "Fraud"
    st.markdown(f"<p style='color:{color}; font-size:20px;'>Predicted Outcome: {result}</p>", unsafe_allow_html=True)
    st.markdown(f"**Confidence Level:** {confidence_level * 100:.2f}%")
    st.markdown("<p style='font-size:16px;'>Predicted Probabilities:</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{color};'>Likelihood of Not Fraud: {predicted_probabilities[0][0] * 100:.2f}%</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{color};'>Likelihood of Fraud: {predicted_probabilities[0][1] * 100:.2f}%</p>", unsafe_allow_html=True)
    print(predicted_fraud)

    if (predicted_fraud[0] == 0):
      return f'The Model has been successfully predicted that claim may be  GENUINE  with the confidence level of: {predicted_probabilities[0][0] * 100:.2f}'
    else:
      return f'The Model has been successfully predicted that claim may be  FRAUD with the confidence level of: {predicted_probabilities[0][1] * 100:.2f}'

# Streamlit app
def main():
    # st.title("Fraud Detection Web App")
    st.markdown("<h1 style='text-align: center;'>Fraud Detection Web Application</h1>", unsafe_allow_html=True)

    col1,col2,col3 =st.columns(3)
    with col1:
        st.header("Insured Details")

        age = st.number_input("ENTER INSURED AGE : ",step=1,min_value=15)

        insured_sex1 = {'FEMALE': 0, 'MALE': 1}
        insured_sex = st.selectbox("Select Gender :",options= {'FEMALE': 0, 'MALE': 1})
        insured_sex=insured_sex1[insured_sex]


        insured_education_level1 ={'Associate': 0, 'College': 1, 'High School': 2, 'JD': 3, 'MD': 4, 'Masters': 5, 'PhD': 6}
        insured_education_level = st.selectbox("ENTER INSURED EDUCATION LEVEL : ",options={'Associate': 0, 'College': 1, 'High School': 2,
                                                                                                    'JD': 3, 'MD': 4, 'Masters': 5, 'PhD': 6})
        insured_education_level = insured_education_level1[insured_education_level]


        insured_occupation1 = {'adm-clerical': 0, 'armed-forces': 1, 'craft-repair': 2, 'exec-managerial': 3, 'farming-fishing': 4,
                                'handlers-cleaners': 5, 'machine-op-inspct': 6, 'other-service': 7, 'priv-house-serv': 8,
                                'prof-specialty': 9, 'protective-serv': 10, 'sales': 11, 'tech-support': 12, 'transport-moving': 13}
        insured_occupation = st.selectbox("ENTER OCCUPATION : ",options={'adm-clerical': 0, 'armed-forces': 1, 'craft-repair': 2,
                                                                                'exec-managerial': 3, 'farming-fishing': 4,
                                                                                    'handlers-cleaners': 5, 'machine-op-inspct': 6, 'other-service': 7,
                                                                                    'priv-house-serv': 8, 'prof-specialty': 9, 'protective-serv': 10,
                                                                                        'sales': 11, 'tech-support': 12, 'transport-moving': 13})
        insured_occupation = insured_occupation1[insured_occupation]

    with col2:
        st.header("Vehicle Details")


        auto_make1 ={'Accura': 0, 'Audi': 1, 'BMW': 2, 'Chevrolet': 3, 'Dodge': 4, 'Ford': 5, 'Honda': 6, 'Jeep': 7,
                    'Mercedes': 8, 'Nissan': 9, 'Saab': 10, 'Suburu': 11, 'Toyota': 12, 'Volkswagen': 13}
        auto_make = st.selectbox("ENTER AUTO MAKE : ",options={'Accura': 0, 'Audi': 1, 'BMW': 2, 'Chevrolet': 3,
                                                                        'Dodge': 4, 'Ford': 5, 'Honda': 6, 'Jeep': 7,
                                                                        'Mercedes': 8, 'Nissan': 9, 'Saab': 10, 'Suburu': 11,
                                                                            'Toyota': 12, 'Volkswagen': 13})
        auto_make = auto_make1[auto_make]

        vehicle_age = st.number_input("ENTER VEHICLE AGE : ",step=1,min_value=0, max_value=50)

        policy_state1 = {'Northern Region': 0, 'Southern Region': 1, 'Central Region': 2}
        policy_state = st.selectbox("ENTER REGION : ",options={'Northern Region': 0, 'Southern Region': 1, 'Central Region': 2})
        policy_state = policy_state1[policy_state]  

    with col3:
        st.header("Accident Details")
  
        collision_type1 = {'Front Collision': 0, 'Rear Collision': 1, 'Side Collision': 2, 'UNKNOWN': 3}
        collision_type = st.selectbox("ENTER COLLISION TYPE : ",options={'Front Collision': 0, 'Rear Collision': 1, 'Side Collision': 2, 'UNKNOWN': 3})
        collision_type = collision_type1[collision_type]

        incident_severity1 ={'Major Damage': 0, 'Minor Damage': 1, 'Total Loss': 2, 'Trivial Damage': 3}
        incident_severity = st.selectbox("Select Incident Severity :",options= {'Major Damage': 0, 'Minor Damage': 1, 'Total Loss': 2, 'Trivial Damage': 3})
        incident_severity = incident_severity1[incident_severity]

        property_damage1 ={'NO': 0, 'UNKNOWN': 1, 'YES': 2}
        property_damage = st.selectbox("Select Property Damage",options={'NO': 0, 'UNKNOWN': 1, 'YES': 2})
        property_damage = property_damage1[property_damage]

        police_report_available1 ={'NO': 0, 'UNKNOWN': 1, 'YES': 2}
        police_report_available = st.selectbox("IS POLICE REPORT AVAILABLE : ",options={'NO': 0, 'UNKNOWN': 1, 'YES': 2})
        police_report_available = police_report_available1[police_report_available]

    # Prepare user input as a DataFrame
    user_input = pd.DataFrame({
        'age': [age],
        'insured_sex': [insured_sex],
        'policy_state': [policy_state],
        'incident_severity': [incident_severity],
        'collision_type': [collision_type],
        'property_damage': [property_damage],
        'police_report_available': [police_report_available],
        'auto_make': [auto_make],
        'vehicle_age': [vehicle_age],
        'insured_education_level': [insured_education_level],
        'insured_occupation': [insured_occupation]    
    })

    col1, col2,col3= st.columns([17,5,15])

    # with col2:
    
    #     # st.markdown("<button style='display: block; margin: 0 auto;'>Make Predictions</button>", unsafe_allow_html=True)
    #     st.write(" ")
    #     st.write(" ")

    # submitted = st.button('Make Predictions')
  
    # if submitted:
    #     st.markdown("""
    #         <style>
    #             div.stButton > button {
    #                 align
    #                 box-shadow: none;
    #                 border: none;
    #             }
    #         </style>
    #     """, unsafe_allow_html=True)
    #     # Call your prediction function here
    #     pred = Fraud_detection(user_input)
    #     pred = "Prediction result"
    #     # st.success(pred)

    with col2:
        with st.form(key='user_input_form1'):
            submitted = st.form_submit_button('Make Predictions')


    with st.form(key='user_input_form'):
        if submitted==True:
            st.markdown("""
                <style>
                    div.stButton > button {
                        box-shadow: none;
                        border: none;
                    }
                </style>
            """, 
            unsafe_allow_html=True)
            pred = Fraud_detection(user_input)
            submitted1 = st.form_submit_button('Successfully Predicted')
            st.info(pred)


# Run the Streamlit app
if __name__ == "__main__":
    main()
