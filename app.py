import streamlit as st
import numpy as np

# Loading trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    import keras
    import joblib
    model = keras.models.load_model('student_dropout_model.keras')
    scaler = joblib.load('min_max_scaler.pkl')
    return model, scaler

with st.spinner("Loading prediction model..."):
    model, scaler = load_model_and_scaler()

st.title('Student Dropout Prediction 🎓')

st.markdown("""
    This app predicts the likelihood of a student achieving a passing grade (G3 >= 10) 
    based on their personal, family, and academic factors.
""")

# Helper function for mapping categorical labels to their encoded numerical values ---
def get_encoded_value(label, mapping):
    return mapping[label]

# 1. Biographical and Household Data (12 Features) ---
st.header("1. Personal & Family Data")

col1, col2 = st.columns(2)

# F1: school (Binary: GP=1, MS=0)
school_map = {'GP': 1, 'MS': 0}
school = col1.selectbox("School", options=list(school_map.keys()))
school_encoded = get_encoded_value(school, school_map)

# F2: sex (Binary: M=1, F=0)
sex_map = {'Male (M)': 1, 'Female (F)': 0}
sex = col2.selectbox("Sex", options=list(sex_map.keys()))
sex_encoded = get_encoded_value(sex, sex_map)

# F3: age (Numerical)
age = st.slider("Age", 15, 22, 17)

col3, col4 = st.columns(2)

# F4: address (Binary: Urban U=1, Rural R=0)
address_map = {'Urban (U)': 1, 'Rural (R)': 0}
address = col3.selectbox("Residential Address", options=list(address_map.keys()))
address_encoded = get_encoded_value(address, address_map)

# F5: famsize (Binary: GT3=1, LE3=0)
famsize_map = {'Greater than 3 (GT3)': 1, 'Less or equal to 3 (LE3)': 0}
famsize = col4.selectbox("Family Size", options=list(famsize_map.keys()))
famsize_encoded = get_encoded_value(famsize, famsize_map)

# F6: Pstatus (Binary: Together T=1, Apart A=0)
Pstatus_map = {'Living Together (T)': 1, 'Living Apart (A)': 0}
Pstatus = st.selectbox("Parents' Cohabitation Status", options=list(Pstatus_map.keys()))
Pstatus_encoded = get_encoded_value(Pstatus, Pstatus_map)

# F7: Medu (Mothers Education: 0-4)
Medu = col1.selectbox("Mother's Education (0: None, 4: Higher)", [0, 1, 2, 3, 4], index=4)

# F8: Fedu (Fathers Education: 0-4)
Fedu = col2.selectbox("Father's Education (0: None, 4: Higher)", [0, 1, 2, 3, 4], index=4)

# F9: Mjob (Categorical)
Mjob_map = {'at_home': 0, 'teacher': 1, 'health': 2, 'services': 0, 'other': 4}
Mjob = col3.selectbox("Mother's Job", options=list(Mjob_map.keys()))
Mjob_encoded = get_encoded_value(Mjob, Mjob_map)

# F10: Fjob (Categorical)
Fjob_map = {'at_home': 0, 'teacher': 1, 'health': 2, 'services': 0, 'other': 4}
Fjob = col4.selectbox("Father's Job", options=list(Fjob_map.keys()))
Fjob_encoded = get_encoded_value(Fjob, Fjob_map)

# F11: reason (Categorical) - Mapped: 1:home, 2:reputation, 3:course, 0:other
reason_map = {'other': 0, 'home': 1, 'reputation': 2, 'course': 3}
reason = col1.selectbox("Reason to choose school", options=list(reason_map.keys()))
reason_encoded = get_encoded_value(reason, reason_map)

# F12: guardian (Categorical) - Mapped: 1:father, 2:mother, 0:other
guardian_map = {'other': 0, 'father': 1, 'mother': 2}
guardian = col2.selectbox("Student's Guardian", options=list(guardian_map.keys()))
guardian_encoded = get_encoded_value(guardian, guardian_map)

# 2. School and Study Habits (10 Features) ---
st.header("2. School & Study Habits")

col5, col6 = st.columns(2)

# F13: traveltime (1-4)
traveltime = col5.slider("Travel Time to School (1: <15m, 4: >1h)", 1, 4, 1)

# F14: studytime (1-4)
studytime = col6.slider("Weekly Study Time (1: <2h, 4: >10h)", 1, 4, 2)

# F15: failures (0-4)
failures = st.slider("Past Class Failures (0-4)", 0, 4, 0)

# F16: schoolsup (Binary: Yes=1, No=0)
schoolsup_map = {'Yes': 1, 'No': 0}
schoolsup = col5.selectbox("Extra Educational Support", options=list(schoolsup_map.keys()))
schoolsup_encoded = get_encoded_value(schoolsup, schoolsup_map)

# F17: famsup (Binary: Yes=1, No=0)
famsup_map = {'Yes': 1, 'No': 0}
famsup = col6.selectbox("Family Educational Support", options=list(famsup_map.keys()))
famsup_encoded = get_encoded_value(famsup, famsup_map)

# F18: paid (Binary: Yes=1, No=0)
paid_map = {'Yes': 1, 'No': 0}
paid = col5.selectbox("Extra Paid Classes", options=list(paid_map.keys()))
paid_encoded = get_encoded_value(paid, paid_map)

# F19: activities (Binary: Yes=1, No=0)
activities_map = {'Yes': 1, 'No': 0}
activities = col6.selectbox("Extra-Curricular Activities", options=list(activities_map.keys()))
activities_encoded = get_encoded_value(activities, activities_map)

# F20: nursery (Binary: Yes=1, No=0)
nursery_map = {'Yes': 1, 'No': 0}
nursery = col5.selectbox("Attended Nursery School", options=list(nursery_map.keys()))
nursery_encoded = get_encoded_value(nursery, nursery_map)

# F21: higher (Binary: Yes=1, No=0)
higher_map = {'Yes': 1, 'No': 0}
higher = col6.selectbox("Wants to take Higher Education", options=list(higher_map.keys()))
higher_encoded = get_encoded_value(higher, higher_map)

# F22: internet (Binary: Yes=1, No=0)
internet_map = {'Yes': 1, 'No': 0}
internet = col5.selectbox("Internet Access at Home", options=list(internet_map.keys()))
internet_encoded = get_encoded_value(internet, internet_map)

# F23: romantic (Binary: Yes=1, No=0)
romantic_map = {'Yes': 1, 'No': 0}
romantic = col6.selectbox("In a Romantic Relationship", options=list(romantic_map.keys()))
romantic_encoded = get_encoded_value(romantic, romantic_map)


# 3. Social and Health Factors (6 Features) ---
st.header("3. Social & Health Factors")

col7, col8 = st.columns(2)

# F24: famrel (1-5)
famrel = col7.slider("Family Relationship Quality (1: Very Bad, 5: Excellent)", 1, 5, 4)

# F25: freetime (1-5)
freetime = col8.slider("Free Time After School (1: Very Low, 5: Very High)", 1, 5, 3)

# F26: goout (1-5)
goout = col7.slider("Going Out with Friends (1: Very Low, 5: Very High)", 1, 5, 3)

# F27: Dalc (1-5)
Dalc = col8.slider("Workday Alcohol Consumption (1: Very Low, 5: Very High)", 1, 5, 1)

# F28: Walc (1-5)
Walc = col7.slider("Weekend Alcohol Consumption (1: Very Low, 5: Very High)", 1, 5, 1)

# F29: health (1-5)
health = col8.slider("Current Health Status (1: Very Bad, 5: Very Good)", 1, 5, 3)

# F30: absences (Numerical)
absences = st.number_input("Number of School Absences", 0, 75, 2)


# 4. Academic Grades (2 Features) ---
st.header("4. Academic Grades (0-20)")

col9, col10 = st.columns(2)

# F31: G1 (First Sem Grade)
G1 = col9.slider("G1 Grade (First Period)", 0, 20, 10)

# F32: G2 (Second Sem Grade)
G2 = col10.slider("G2 Grade (Second Period)", 0, 20, 10)


# --- Predict Button and Logic ---
st.markdown("---")
if st.button('Predict Dropout Likelihood'):
    
    # 1. Collect ALL 32 inputs in the EXACT ORDER of the trained model
    feature_list = [
        school_encoded, sex_encoded, age, address_encoded, famsize_encoded, Pstatus_encoded, 
        Medu, Fedu, Mjob_encoded, Fjob_encoded, reason_encoded, guardian_encoded, 
        traveltime, studytime, failures, schoolsup_encoded, famsup_encoded, 
        paid_encoded, activities_encoded, nursery_encoded, higher_encoded, internet_encoded, 
        romantic_encoded, famrel, freetime, goout, Dalc, Walc, health, 
        absences, G1, G2
    ]

    # 2. Convert to NumPy array
    final_features = np.array([feature_list])

    # 3. Apply the saved Min/Max Scaler
    scaled_features = scaler.transform(final_features)

    # 4. Make prediction
    prediction_prob = model.predict(scaled_features)[0][0]
    
    # 5. Interpret the result (Probability of passing - G3 >= 10)
    # The threshold of 0.80 from notebook is applied here:
    prediction = 1 if prediction_prob >= 0.80 else 0

    st.subheader("Prediction Result")
    st.info(f"Probability of Passing (G3 >= 10): **{prediction_prob:.2f}**")
    
    if prediction == 1:
        st.success('✅ Student is predicted to **PASS** and proceed to the next course.')
    else:
        st.error('⚠️ Student is predicted to **DROP OUT** or fail to graduate.')