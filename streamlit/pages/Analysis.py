import streamlit as st

st.set_page_config(
    page_title="EDA Presentation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Exploratory Data Analysis Presentation")
st.write("This is a curated presentation of key EDA findings.")

# Show the first image
st.subheader("1. Class imbalance")
st.image("../assets/EDA/class_imbalance.png", caption="Target feature imbalance", use_container_width=True)

# Show the second image
st.subheader("2. Correlation numerical")
st.image("../assets/EDA/biserial_target.png", caption="Correlation numerical", use_container_width=True)


# Show the third image
st.subheader("3. Attrition by age")
st.image("../assets/EDA/age_target.png", caption="Attrtition by employee age", use_container_width=True)


# Show the third image
st.subheader("4. Attrition by years of experience")
st.image("../assets/EDA/totalworkingyears_target.png", caption="Attrtition by total working years", use_container_width=True)


# Show the second image
st.subheader("5. Correlation categorical")
st.image("../assets/EDA/cramers_target.png", caption="Correlation categorical", use_container_width=True)


# Show the second image
st.subheader("6. Environment and job satisfaction influence")
st.image("../assets/EDA/environment_target.png", caption="Environment", use_container_width=True)
st.image("../assets/EDA/jobsatisfaction_target.png", caption="Job satisfaction", use_container_width=True)

st.subheader("7. Model comparisons")
st.image("../assets/EDA/model_comparison.png", caption="Model metric comparison", use_container_width=True)

st.subheader("8. Chosen Model feature contributions")
st.image("../assets/EDA/shap_forest.png", caption="SHAP values Random Forest", use_container_width=True)
st.image("../assets/EDA/shap_catboost.png", caption="SHAP values Catboost", use_container_width=True)


st.write("Thank you for viewing!")
