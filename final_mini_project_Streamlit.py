from ssl import Options
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

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
    st.title("Fraud Detection Web App")
    st.sidebar.header("User Input Features")

    # Collect user input
    age = st.sidebar.number_input("ENTER AGE : ",step=1,min_value=5)
    
    insured_sex1 = {'FEMALE': 0, 'MALE': 1}
    insured_sex = st.sidebar.selectbox("Select Gender :",options= {'FEMALE': 0, 'MALE': 1})
    insured_sex=insured_sex1[insured_sex]

    insured_occupation1 = {'adm-clerical': 0, 'armed-forces': 1, 'craft-repair': 2, 'exec-managerial': 3, 'farming-fishing': 4,
                            'handlers-cleaners': 5, 'machine-op-inspct': 6, 'other-service': 7, 'priv-house-serv': 8,
                            'prof-specialty': 9, 'protective-serv': 10, 'sales': 11, 'tech-support': 12, 'transport-moving': 13}
    insured_occupation = st.sidebar.selectbox("ENTER OCCUPATION : ",options={'adm-clerical': 0, 'armed-forces': 1, 'craft-repair': 2,
                                                                            'exec-managerial': 3, 'farming-fishing': 4,
                                                                                'handlers-cleaners': 5, 'machine-op-inspct': 6, 'other-service': 7,
                                                                                'priv-house-serv': 8, 'prof-specialty': 9, 'protective-serv': 10,
                                                                                    'sales': 11, 'tech-support': 12, 'transport-moving': 13})
    insured_occupation = insured_occupation1[insured_occupation]
    # Add input fields for other features

    incident_severity1 ={'Major Damage': 0, 'Minor Damage': 1, 'Total Loss': 2, 'Trivial Damage': 3}
    incident_severity = st.sidebar.selectbox("Select Incident Severity :",options= {'Major Damage': 0, 'Minor Damage': 1, 'Total Loss': 2, 'Trivial Damage': 3})
    incident_severity = incident_severity1[incident_severity]

    property_damage1 ={'NO': 0, 'UNKNOWN': 1, 'YES': 2}
    property_damage = st.sidebar.selectbox("Select Property Damage",options={'NO': 0, 'UNKNOWN': 1, 'YES': 2})
    property_damage = property_damage1[property_damage]

    police_report_available1 ={'NO': 0, 'UNKNOWN': 1, 'YES': 2}
    police_report_available = st.sidebar.selectbox("IS POLICE REPORT AVAILABLE : ",options={'NO': 0, 'UNKNOWN': 1, 'YES': 2})
    police_report_available = police_report_available1[police_report_available]

    collision_type1 = {'Front Collision': 0, 'Rear Collision': 1, 'Side Collision': 2, 'UNKNOWN': 3}
    collision_type = st.sidebar.selectbox("ENTER COLLISION TYPE : ",options={'Front Collision': 0, 'Rear Collision': 1, 'Side Collision': 2, 'UNKNOWN': 3})
    collision_type = collision_type1[collision_type]

    policy_state1 = {'IL': 0, 'IN': 1, 'OH': 2}
    policy_state = st.sidebar.selectbox("ENTER POLICY STATE : ",options={'IL': 0, 'IN': 1, 'OH': 2})
    policy_state = policy_state1[policy_state]

    insured_education_level1 ={'Associate': 0, 'College': 1, 'High School': 2, 'JD': 3, 'MD': 4, 'Masters': 5, 'PhD': 6}
    insured_education_level = st.sidebar.selectbox("ENTER INSURED EDUCATION LEVEL : ",options={'Associate': 0, 'College': 1, 'High School': 2,
                                                                                                'JD': 3, 'MD': 4, 'Masters': 5, 'PhD': 6})
    insured_education_level = insured_education_level1[insured_education_level]


    auto_make1 ={'Accura': 0, 'Audi': 1, 'BMW': 2, 'Chevrolet': 3, 'Dodge': 4, 'Ford': 5, 'Honda': 6, 'Jeep': 7,
                'Mercedes': 8, 'Nissan': 9, 'Saab': 10, 'Suburu': 11, 'Toyota': 12, 'Volkswagen': 13}
    auto_make = st.sidebar.selectbox("ENTER AUTO MAKE : ",options={'Accura': 0, 'Audi': 1, 'BMW': 2, 'Chevrolet': 3,
                                                                    'Dodge': 4, 'Ford': 5, 'Honda': 6, 'Jeep': 7,
                                                                    'Mercedes': 8, 'Nissan': 9, 'Saab': 10, 'Suburu': 11,
                                                                        'Toyota': 12, 'Volkswagen': 13})
    auto_make = auto_make1[auto_make]

    vehicle_age = st.sidebar.number_input("ENTER Vehicle AGE : ",step=1,min_value=5, max_value=50)


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

    col1, col2 = st.columns(2)

    with col1:
        with st.form(key='user_input_form'):
            
            submitted = st.form_submit_button('Make Predictions')
            if submitted:
                st.markdown("""
                    <style>
                        div.stButton > button {
                            box-shadow: none;
                            border: none;
                        }
                    </style>
                """, unsafe_allow_html=True)
                pred = Fraud_detection(user_input)
                submitted1 = st.form_submit_button('Successfully Predicted')
    with col2:
        if submitted:
            st.success(pred)


# Run the Streamlit app
if __name__ == "__main__":
    main()