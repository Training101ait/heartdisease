import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# Load the saved scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
model = tf.keras.models.load_model('heart_model.h5')

st.title("Heart Disease Prediction App")
st.write(
    """
    This app predicts the risk of heart disease based on several clinical features.
    Please provide the values for each feature below.
    """
)

# Detailed explanation for each feature:
st.subheader("Feature Explanations:")
st.markdown(
    """
    - **Age**: Patient's age in years.
    - **Sex**: Patient's gender (0 = Female, 1 = Male).
    - **Chest Pain Type (cp)**: Type of chest pain experienced.
      - **0**: Asymptomatic  
      - **1**: Atypical Angina  
      - **2**: Non-anginal Pain  
      - **3**: Typical Angina  
    - **Resting Blood Pressure (trestbps)**: Blood pressure (mm Hg) measured at rest.
    - **Serum Cholesterol (chol)**: Cholesterol level in mg/dl.
    - **Fasting Blood Sugar (fbs)**: Fasting blood sugar level.
      - **0**: ≤ 120 mg/dl  
      - **1**: > 120 mg/dl  
    - **Resting ECG Results (restecg)**: Results of the resting electrocardiogram.
      - **0**: Normal  
      - **1**: ST-T wave abnormality  
      - **2**: Left ventricular hypertrophy  
    - **Maximum Heart Rate Achieved (thalach)**: Maximum heart rate during a stress test.
    - **Exercise Induced Angina (exang)**: Indicates if exercise causes chest pain.
      - **0**: No  
      - **1**: Yes  
    - **ST Depression (oldpeak)**: ST depression induced by exercise relative to rest.
    - **Slope of the Peak Exercise ST Segment (slope)**: Slope of the ST segment during exercise.
      - **0**: Downsloping  
      - **1**: Flat  
      - **2**: Upsloping  
    - **Number of Major Vessels (ca)**: Number of major vessels (0-3) colored by fluoroscopy.
    - **Thalassemia (thal)**: Blood disorder status.
      - **0**: Normal  
      - **1**: Fixed Defect  
      - **2**: Reversible Defect  
    """
)

# Create input widgets with explanations as tooltips

# Age in years
age = st.number_input(
    "Age (years)", 
    min_value=1, max_value=120, 
    value=50, 
    help="Age of the patient in years."
)

# Sex: 0 = Female, 1 = Male
sex = st.selectbox(
    "Sex (0 = Female, 1 = Male)", 
    options=[0, 1], 
    help="Gender of the patient: 0 for Female, 1 for Male."
)

# Chest pain type: 0 (asymptomatic), 1 (atypical angina), 2 (non-anginal pain), 3 (typical angina)
cp = st.number_input(
    "Chest Pain Type (cp)", 
    min_value=0, max_value=3, 
    value=1, 
    help="Type of chest pain: 0=asymptomatic, 1=atypical angina, 2=non-anginal pain, 3=typical angina."
)

# Resting blood pressure in mm Hg
trestbps = st.number_input(
    "Resting Blood Pressure (trestbps)", 
    min_value=50, max_value=250, 
    value=130, 
    help="Resting blood pressure in mm Hg."
)

# Serum cholesterol in mg/dl
chol = st.number_input(
    "Serum Cholesterol (chol)", 
    min_value=100, max_value=600, 
    value=250, 
    help="Serum cholesterol in mg/dl."
)

# Fasting blood sugar: 0 (≤120 mg/dl) or 1 (>120 mg/dl)
fbs = st.selectbox(
    "Fasting Blood Sugar (fbs) (0 = ≤120 mg/dl, 1 = >120 mg/dl)", 
    options=[0, 1], 
    help="Fasting blood sugar level: 0 for ≤120 mg/dl, 1 for >120 mg/dl."
)

# Resting ECG results: 0 (normal), 1 (ST-T wave abnormality), 2 (left ventricular hypertrophy)
restecg = st.number_input(
    "Resting ECG Results (restecg)", 
    min_value=0, max_value=2, 
    value=1, 
    help="Resting electrocardiogram results: 0=normal, 1=ST-T wave abnormality, 2=left ventricular hypertrophy."
)

# Maximum heart rate achieved during stress test
thalach = st.number_input(
    "Maximum Heart Rate Achieved (thalach)", 
    min_value=60, max_value=250, 
    value=150, 
    help="Maximum heart rate achieved during a stress test."
)

# Exercise induced angina: 0 (No) or 1 (Yes)
exang = st.selectbox(
    "Exercise Induced Angina (exang) (0 = No, 1 = Yes)", 
    options=[0, 1], 
    help="Does the patient experience chest pain during exercise? 0 for No, 1 for Yes."
)

# ST depression induced by exercise relative to rest
oldpeak = st.number_input(
    "ST Depression (oldpeak)", 
    min_value=0.0, max_value=10.0, 
    value=1.0, step=0.1, 
    help="ST depression induced by exercise relative to rest."
)

# Slope of the peak exercise ST segment: 0 (downsloping), 1 (flat), 2 (upsloping)
slope = st.number_input(
    "Slope of the Peak Exercise ST Segment (slope)", 
    min_value=0, max_value=2, 
    value=1, 
    help="Slope of the ST segment during exercise: 0=downsloping, 1=flat, 2=upsloping."
)

# Number of major vessels colored by fluoroscopy (0-3)
ca = st.number_input(
    "Number of Major Vessels (ca)", 
    min_value=0, max_value=3, 
    value=0, 
    help="Number of major vessels (0-3) colored by fluoroscopy."
)

# Thalassemia status: 0 (normal), 1 (fixed defect), 2 (reversible defect)
thal = st.number_input(
    "Thalassemia (thal)", 
    min_value=0, max_value=2, 
    value=1, 
    help="Thalassemia status: 0=normal, 1=fixed defect, 2=reversible defect."
)

# When the button is pressed, arrange the inputs and predict
if st.button("Predict Heart Disease"):
    # Arrange the input features in the same order as the training data
    input_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])
    
    # Scale the features using the saved scaler
    input_scaled = scaler.transform(input_features)
    
    # Get the probability prediction from the model
    prediction_prob = model.predict(input_scaled)[0][0]
    
    # Use a threshold of 0.5 to determine the class label
    prediction = 1 if prediction_prob >= 0.5 else 0
    
    st.write("### Prediction:")
    if prediction == 1:
        st.error(f"High risk of heart disease (Probability: {prediction_prob:.2f})")
    else:
        st.success(f"Low risk of heart disease (Probability: {prediction_prob:.2f})")
