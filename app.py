
import streamlit as st
import numpy as np
import joblib

# Load models
dia_path = "models/diabetes_model.pkl"
heart_path = "models/heart_model.pkl"

# Check if models exist, if not create dummy ones
import os
if not os.path.exists("models"):
    os.makedirs("models")

if not os.path.exists(dia_path):
    # Create a dummy diabetes model if it doesn't exist
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Assuming a dummy dataset or placeholder logic for diabetes model
    # In a real scenario, you'd train and save this model properly
    dummy_data = pd.DataFrame(np.random.rand(100, 4), columns=['sugar', 'bmi', 'age', 'target'])
    dummy_data['target'] = np.random.randint(0, 2, 100)
    X_dummy = dummy_data.drop('target', axis=1)
    y_dummy = dummy_data['target']
    X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(X_dummy, y_dummy, test_size=0.2)
    dummy_diabetes_model = RandomForestClassifier()
    dummy_diabetes_model.fit(X_train_dummy, y_train_dummy)
    joblib.dump(dummy_diabetes_model, dia_path)

if not os.path.exists(heart_path):
    # Create a dummy heart model if it doesn't exist
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Assuming a dummy dataset or placeholder logic for heart model
    # In a real scenario, you'd train and save this model properly
    dummy_data_heart = pd.DataFrame(np.random.rand(100, 4), columns=['age', 'chest_pain', 'breathing', 'fatigue'])
    dummy_data_heart['target'] = np.random.randint(0, 2, 100)
    X_dummy_heart = dummy_data_heart.drop('target', axis=1)
    y_dummy_heart = dummy_data_heart['target']
    X_train_dummy_heart, X_test_dummy_heart, y_train_dummy_heart, y_test_dummy_heart = train_test_split(X_dummy_heart, y_dummy_heart, test_size=0.2)
    dummy_heart_model = RandomForestClassifier()
    dummy_heart_model.fit(X_train_dummy_heart, y_train_dummy_heart)
    joblib.dump(dummy_heart_model, heart_path)

diabetes_model = joblib.load(dia_path)
heart_model = joblib.load(heart_path)

st.set_page_config(page_title="AI Health System", layout="centered")

st.title("🏥 AI Health Prediction System (Live Deployment)")

menu = st.sidebar.radio("Select Module", ["Diabetes", "Heart Disease"])

# =========================
# DIABETES
# =========================
if menu == "Diabetes":
    st.subheader("🩸 Diabetes Risk Check")

    sugar = st.slider("Blood Sugar Level", 0, 200, 100)
    bmi = st.slider("BMI", 10, 50, 25)
    age = st.slider("Age", 10, 100, 30)

    if st.button("Predict Risk"):
        data = np.array([[sugar, bmi, age, 1]])

        prob = diabetes_model.predict_proba(data)[0][1] * 100

        st.metric("Risk %", f"{prob:.2f}%")
        st.progress(int(prob))

        if prob > 70:
            st.error("⚠️ High Risk")
        elif prob > 30:
            st.warning("⚠️ Medium Risk")
        else:
            st.success("✅ Low Risk")

# =========================
# HEART
# =========================
if menu == "Heart Disease":
    st.subheader("❤️ Heart Risk Check")

    age = st.slider("Age", 10, 100, 40)
    chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])
    breathing = st.selectbox("Breathing Issue", ["No", "Yes"])
    fatigue = st.selectbox("Fatigue", ["No", "Yes"])

    if st.button("Predict Risk"):
        data = np.array([[
            age,
            1 if chest_pain == "Yes" else 0,
            1 if breathing == "Yes" else 0,
            1 if fatigue == "Yes" else 0
        ]])

        prob = heart_model.predict_proba(data)[0][1] * 100

        st.metric("Risk %", f"{prob:.2f}%")
        st.progress(int(prob))

        if prob > 70:
            st.error("⚠️ High Risk")
        elif prob > 30:
            st.warning("⚠️ Medium Risk")
        else:
            st.success("✅ Low Risk")
