import streamlit as st
import joblib
import pandas as pd

# Load model
try:
    model = joblib.load('models/heart_disease_model.pkl')
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    model = None

# App title
st.title("Heart Disease Risk Prediction")

# Input form
with st.form("prediction_form"):
    st.header("Patient Information")
    
    # Demographic
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
    
    # Medical history
    cp = st.selectbox(
        "Chest Pain Type",
        options=[("Typical angina", 0), ("Atypical angina", 1), 
                 ("Non-anginal pain", 2), ("Asymptomatic", 3)],
        format_func=lambda x: x[0]
    )[1]
    
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[("No", 0), ("Yes", 1)])[1]
    
    # ECG and exercise
    restecg = st.selectbox(
        "Resting ECG Results",
        options=[("Normal", 0), ("ST-T wave abnormality", 1), 
                 ("Left ventricular hypertrophy", 2)],
        format_func=lambda x: x[0]
    )[1]
    
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)])[1]
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    
    # Additional parameters
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=[("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
        format_func=lambda x: x[0]
    )[1]
    
    ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 0)
    thal = st.selectbox(
        "Thalassemia",
        options=[("Normal", 0), ("Fixed defect", 1), ("Reversible defect", 2)],
        format_func=lambda x: x[0]
    )[1]
    
    # Submit button
    submitted = st.form_submit_button("Predict Risk")
    
    if submitted and model is not None:
        # Prepare input data
        input_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Feature engineering (must match training)
        df['bp_hr_ratio'] = df['trestbps'] / df['thalach']
        df['high_chol'] = (df['chol'] > 240).astype(int)
        
        # One-hot encode categoricals
        for col in ['cp', 'restecg', 'slope', 'thal']:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
        
        # Ensure all expected columns are present
        expected_cols = model.feature_names_in_
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns
        df = df[expected_cols]
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1] * 100
        
        # Display results
        st.subheader("Prediction Results")
        
        if prediction:
            st.error(f"🚨 High Risk ({probability:.2f}% probability)")
            st.warning("Recommendation: Consult a cardiologist immediately")
        else:
            st.success(f"✅ Low Risk ({probability:.2f}% probability)")
            st.info("Recommendation: No significant risk detected")
        
        # Show interpretation guide
        with st.expander("Feature Interpretation Guide"):
            st.markdown("""
            - **Chest Pain Type (cp)**: 0=Typical angina, 1=Atypical angina, 2=Non-anginal pain, 3=Asymptomatic
            - **Resting ECG (restecg)**: 0=Normal, 1=ST-T wave abnormality, 2=Left ventricular hypertrophy
            - **Slope of Peak Exercise ST Segment (slope)**: 0=Upsloping, 1=Flat, 2=Downsloping
            """)

if __name__ == '__main__':
    st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
    st.sidebar.markdown("## About")
    st.sidebar.info("This app predicts the risk of heart disease using machine learning.")