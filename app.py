import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
from fpdf import FPDF

model = joblib.load('endometrial_risk_model.pkl')
le_dict = joblib.load('label_encoders.pkl')

st.set_page_config(page_title="Endometrial Cancer Risk Predictor", layout="centered")

# --- Language selection ---
lang = st.selectbox("Language / மொழி", ["English", "Tamil"])

text = {
    "English": {
        "title": "🩺 AI-Based Endometrial Cancer Risk Predictor",
        "subtitle": "Enter your menstrual and lifestyle details below to check your risk level.",
        "warning": "⚠️ This is a screening/awareness tool only, NOT a medical diagnosis. Please consult a doctor for proper evaluation.",
        "header": "Enter Your Details",
        "name": "Full Name",
        "height": "Height (in cm)",
        "weight": "Weight (in kg)",
        "age": "Age",
        "menarche": "Age at Menarche (first period)",
        "cycle_length": "Menstrual Cycle Length (days)",
        "cycle_reg": "Cycle Regularity",
        "menopause": "Menopause Status",
        "bleed_post": "Do you experience Postmenopausal Bleeding?",
        "bleed_ab": "Do you experience Abnormal Bleeding between periods?",
        "parity": "Number of Children (Parity)",
        "pcos": "PCOS (Polycystic Ovary Syndrome)",
        "diabetes": "Diabetes",
        "hypertension": "Hypertension (High BP)",
        "family": "Family History of Cancer",
        "hormone": "Hormone Therapy Use",
        "button": "🔍 Check My Risk",
        "name_error": "Please enter your name before checking your risk.",
        "result_for": "Result for",
        "confidence": "### Prediction Confidence:",
        "factors": "### 📊 Key Risk Factors Contributing to This Result:",
        "recommendation": "### 💡 Recommendation:",
        "low_msg": "- Continue regular menstrual cycle tracking every month\n- Maintain a healthy BMI through balanced diet and regular exercise\n- Go for routine annual gynecological check-ups\n- Manage existing conditions like diabetes or blood pressure if present\n- Avoid unsupervised hormone therapy or medications\n- Stay alert to any new symptoms like unusual bleeding",
        "medium_msg": "- Consult a gynecologist for a proper clinical evaluation soon\n- Monitor and manage BMI/weight if above the healthy range\n- Track and report any irregular bleeding patterns promptly\n- Manage PCOS, diabetes, or hypertension under medical guidance\n- Avoid self-medication with hormone therapy without supervision\n- Go for an ultrasound or check-up within the next few months\n- Re-check your risk level after lifestyle improvements",
        "high_msg": "- Consult a gynecologist/oncologist as soon as possible\n- Get clinical tests done (ultrasound, endometrial biopsy) without delay\n- Do not ignore abnormal or postmenopausal bleeding\n- Bring this report as a reference during your doctor consultation\n- Follow all diagnostic advice given by your healthcare provider\n- Involve a family member for support during medical evaluation",
        "final_note": "Please consult a gynecologist/oncologist for proper medical evaluation regardless of this result.",
        "saved": "✅ Your assessment has been saved to records.",
        "download": "📄 Download Report as PDF"
    },
    "Tamil": {
        "title": "🩺 AI அடிப்படையிலான கருப்பை புற்றுநோய் ஆபத்து கணிப்பான்",
        "subtitle": "உங்கள் மாதவிடாய் மற்றும் வாழ்க்கை முறை விவரங்களை கீழே உள்ளிடவும்.",
        "warning": "⚠️ இது ஒரு awareness tool மட்டுமே, மருத்துவ கண்டறிதல் அல்ல. மருத்துவரை அணுகவும்.",
        "header": "உங்கள் விவரங்களை உள்ளிடவும்",
        "name": "முழு பெயர்",
        "height": "உயரம் (செ.மீ)",
        "weight": "எடை (கிலோ)",
        "age": "வயது",
        "menarche": "முதல் மாதவிடாய் ஏற்பட்ட வயது",
        "cycle_length": "மாதவிடாய் சுழற்சி நீளம் (நாட்கள்)",
        "cycle_reg": "சுழற்சி முறை",
        "menopause": "மாதவிடாய் நிறுத்த நிலை",
        "bleed_post": "மாதவிடாய் நிறுத்தத்திற்குப் பிறகு இரத்தப்போக்கு உள்ளதா?",
        "bleed_ab": "மாதவிடாய் இடையில் அசாதாரண இரத்தப்போக்கு உள்ளதா?",
        "parity": "குழந்தைகள் எண்ணிக்கை",
        "pcos": "PCOS (கருமுட்டை பை நோய்)",
        "diabetes": "நீரிழிவு",
        "hypertension": "உயர் இரத்த அழுத்தம்",
        "family": "குடும்ப புற்றுநோய் வரலாறு",
        "hormone": "ஹார்மோன் சிகிச்சை",
        "button": "🔍 எனது ஆபத்தை சரிபார்க்கவும்",
        "name_error": "தயவுசெய்து உங்கள் பெயரை உள்ளிடவும்.",
        "result_for": "முடிவு",
        "confidence": "### கணிப்பு நம்பகத்தன்மை:",
        "factors": "### 📊 முக்கிய ஆபத்து காரணிகள்:",
        "recommendation": "### 💡 பரிந்துரை:",
        "low_msg": "உங்கள் ஆபத்து நிலை குறைவு. வழக்கமான பரிசோதனைகளை தொடரவும்.",
        "medium_msg": "உங்கள் ஆபத்து நிலை நடுத்தரம். விரைவில் மருத்துவரை அணுகவும்.",
        "high_msg": "உங்கள் ஆபத்து நிலை அதிகம். உடனடியாக மருத்துவரை அணுகவும்.",
        "final_note": "இந்த முடிவைப் பொருட்படுத்தாமல் மருத்துவரை அணுகவும்.",
        "saved": "✅ உங்கள் தகவல் பதிவு செய்யப்பட்டது.",
        "download": "📄 அறிக்கையை PDF ஆக பதிவிறக்கவும்"
    }
}

t = text[lang]
def yn_format(option):
    if lang == "Tamil":
        return "ஆம்" if option == "Yes" else "இல்லை"
    return option

def cycle_format(option):
    if lang == "Tamil":
        return "வழக்கமான" if option == "Regular" else "ஒழுங்கற்ற"
    return option

st.title(t["title"])
st.write(t["subtitle"])
st.warning(t["warning"])

st.header(t["header"])

name = st.text_input(t["name"])

col1, col2 = st.columns(2)
with col1:
    height_cm = st.number_input(t["height"], min_value=100.0, max_value=220.0, value=160.0)
with col2:
    weight_kg = st.number_input(t["weight"], min_value=30.0, max_value=200.0, value=60.0)

height_m = height_cm / 100
bmi = weight_kg / (height_m ** 2)
st.info(f"📊 BMI: *{bmi:.1f}*")

if bmi < 18.5:
    bmi_category = "Underweight"
elif bmi < 25:
    bmi_category = "Normal"
elif bmi < 30:
    bmi_category = "Overweight"
else:
    bmi_category = "Obese"
st.write(f"BMI Category: *{bmi_category}*")

age = st.number_input(t["age"], min_value=18, max_value=100, value=35)
age_at_menarche = st.number_input(t["menarche"], min_value=8, max_value=20, value=13)
cycle_length = st.number_input(t["cycle_length"], min_value=15, max_value=60, value=28)
cycle_regularity = st.selectbox(t["cycle_reg"], ["Regular", "Irregular"], format_func=cycle_format)
menopause_status = st.selectbox(t["menopause"], ["Yes", "No"], format_func=yn_format)

if menopause_status == "Yes":
    abnormal_bleeding = st.selectbox(t["bleed_post"], ["Yes", "No"], format_func=yn_format)
else:
    abnormal_bleeding = st.selectbox(t["bleed_ab"], ["Yes", "No"], format_func=yn_format)

parity = st.number_input(t["parity"], min_value=0, max_value=10, value=1)
pcos = st.selectbox(t["pcos"], ["Yes", "No"], format_func=yn_format)
diabetes = st.selectbox(t["diabetes"], ["Yes", "No"], format_func=yn_format)
hypertension = st.selectbox(t["hypertension"], ["Yes", "No"], format_func=yn_format)
family_history = st.selectbox(t["family"], ["Yes", "No"], format_func=yn_format)
hormone_therapy = st.selectbox(t["hormone"], ["Yes", "No"], format_func=yn_format)
if st.button(t["button"]):

    if name.strip() == "":
        st.error(t["name_error"])
    else:
        input_data = pd.DataFrame({
            'Age': [age], 'BMI': [bmi], 'Age_at_Menarche': [age_at_menarche],
            'Cycle_Length': [cycle_length], 'Cycle_Regularity': [cycle_regularity],
            'Menopause_Status': [menopause_status], 'Parity': [parity],
            'PCOS': [pcos], 'Diabetes': [diabetes], 'Hypertension': [hypertension],
            'Family_History': [family_history], 'Abnormal_Bleeding': [abnormal_bleeding],
            'Hormone_Therapy': [hormone_therapy]
        })

        categorical_cols = ['Cycle_Regularity', 'Menopause_Status', 'PCOS', 'Diabetes',
                             'Hypertension', 'Family_History', 'Abnormal_Bleeding', 'Hormone_Therapy']

        encoded_data = input_data.copy()
        for col in categorical_cols:
            le = le_dict[col]
            encoded_data[col] = le.transform(encoded_data[col])

        prediction = model.predict(encoded_data)[0]
        probabilities = model.predict_proba(encoded_data)[0]

        st.subheader(f"{t['result_for']}: {name}")

        if prediction == "High":
            st.error(f"⚠️ Risk Level: *{prediction}*")
        elif prediction == "Medium":
            st.warning(f"⚠️ Risk Level: *{prediction}*")
        else:
            st.success(f"✅ Risk Level: *{prediction}*")

        st.write(t["confidence"])
        classes = model.classes_
        for cls, prob in zip(classes, probabilities):
            st.write(f"{cls}: {prob*100:.1f}%")

        st.write(t["factors"])
        feature_names = encoded_data.columns.tolist()
        importances = model.feature_importances_
        importance_df = pd.DataFrame({'Factor': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Factor'))

        st.write(t["recommendation"])
        if prediction == "Low":
            st.info(t["low_msg"])
        elif prediction == "Medium":
            st.warning(t["medium_msg"])
        else:
            st.error(t["high_msg"])

        st.info(t["final_note"])

        record = input_data.copy()
        record.insert(0, "Name", name)
        record["Predicted_Risk"] = prediction
        record["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        file_path = "user_records.csv"
        if os.path.exists(file_path):
            record.to_csv(file_path, mode='a', header=False, index=False)
        else:
            record.to_csv(file_path, mode='w', header=True, index=False)

        st.success(t["saved"])

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="Endometrial Cancer Risk Assessment Report", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 8, txt=f"Name: {name}", ln=True)
        pdf.cell(200, 8, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 8, txt="Entered Details:", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.cell(200, 7, txt=f"Age: {age}   BMI: {bmi:.1f} ({bmi_category})", ln=True)
        pdf.cell(200, 7, txt=f"Age at Menarche: {age_at_menarche}   Cycle Length: {cycle_length} days", ln=True)
        pdf.cell(200, 7, txt=f"Cycle Regularity: {cycle_regularity}   Menopause: {menopause_status}", ln=True)
        pdf.cell(200, 7, txt=f"Bleeding Issue: {abnormal_bleeding}   Parity: {parity}", ln=True)
        pdf.cell(200, 7, txt=f"PCOS: {pcos}   Diabetes: {diabetes}   Hypertension: {hypertension}", ln=True)
        pdf.cell(200, 7, txt=f"Family History: {family_history}   Hormone Therapy: {hormone_therapy}", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 13)
        pdf.cell(200, 8, txt=f"Predicted Risk Level: {prediction}", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 6, txt="Disclaimer: This is a screening tool only, NOT a medical diagnosis. Please consult a doctor.")
        pdf_output = pdf.output(dest='S').encode('latin-1')

        st.download_button(label=t["download"], data=pdf_output, file_name=f"{name}_risk_report.pdf", mime="application/pdf")