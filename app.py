from ssl import Options
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

st.set_page_config(page_title="Fraud_detection",initial_sidebar_state="expanded",layout="wide",menu_items={"About":"https://github.com/Vikram2305/Insurance-Fraud-Detection"})
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

    if predicted_fraud[0] == 0:
        st.success( f'The model has determined the claim to be **VALID** with a confidence level of {confidence_level * 100:.2f}%.')
    else:
        st.warning( f'The model has flagged the claim as **SUSPICIOUS** with a confidence level of {confidence_level * 100:.2f}%. Investigation recommended.')



# Streamlit app
def main():
    with st.sidebar:
        option=option_menu(None,options=["Home","Prediction"], orientation='vertical', icons=["bi bi-house-door", 'bi bi-graph-up'])
    if option=="Home":
        st.markdown("<h1 style='text-align: center;'>Fraud Detection Web Application</h1>", unsafe_allow_html=True)
        st.markdown("<h2>Welcome to the Insurance Fraud Detection Web Application</h2>", unsafe_allow_html=True)
        st.markdown("<p> This web application is designed to help insurance companies detect potential fraud in claims. By leveraging machine learning algorithms, we provide a streamlined process for analyzing claim details and predicting the likelihood of fraud.</p>", unsafe_allow_html=True)
        st.markdown("<h2>How It Works</h2>", unsafe_allow_html=True)
        st.markdown("<p>Simply enter the required details about the insured individual, vehicle, and accident information. Our sophisticated model will then process the data and provide a prediction regarding the authenticity of the claim.</p>", unsafe_allow_html=True)
        st.markdown("<h2>Why This?</h2>", unsafe_allow_html=True)
        st.markdown("<ul><li>Advanced Machine Learning: Our model is built using state-of-the-art machine learning techniques for accurate predictions.</li><li>Efficient Fraud Detection: Save time and resources by quickly identifying suspicious claims.</li><li>User-Friendly Interface: Our web application offers an intuitive and easy-to-use interface for seamless interaction.</li><li>Data Privacy: We prioritize data security and privacy, ensuring that your information remains confidential and secure.</li></ul>", unsafe_allow_html=True)
        st.markdown("<h2>Get Started</h2>", unsafe_allow_html=True)
        st.markdown("<p>Experience the power of predictive analytics in fraud detection. Start using our web application today!</p>", unsafe_allow_html=True)
    elif option=="Prediction" :
        st.header("Enter the claim Details")
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

        col1, col2,col3= st.columns([15,6,15])

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
                st.form_submit_button('Enter Another......!')
                # st.info(pred)
# Run the Streamlit app
if __name__ == "__main__":
    main()
