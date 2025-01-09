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
st.image("../assets/EDA/class_imbalance.png", caption="Target feature imbalance", use_container_width=False)

# Show the second image
st.subheader("2. Correlation numerical")
st.image("../assets/EDA/biserial_target.png", caption="Correlation numerical", use_container_width=False)


# Show the third image
st.subheader("3. Attrition by age")
st.image("../assets/EDA/age_target.png", caption="Attrtition by employee age", use_container_width=False)


# Show the third image
st.subheader("4. Attrition by years of experience")
st.image("../assets/EDA/totalworkingyears_target.png", caption="Attrtition by total working years", use_container_width=False)


st.subheader("5. Correlation between numerical features")
st.image("../assets/EDA/corr_numerical.png", caption="Correlation between numerical features", use_container_width=False)


# Show the second image
st.subheader("6. Correlation categorical")
st.image("../assets/EDA/cramers_target.png", caption="Correlation categorical", use_container_width=False)


# Show the second image
st.subheader("7. Environment and job satisfaction influence")
st.image("../assets/EDA/environment_target.png", caption="Environment", use_container_width=False)
st.image("../assets/EDA/jobsatisafaction_target.png", caption="Job satisfaction", use_container_width=False)

# Show the second image
st.subheader("8. Marital status influence")
st.image("../assets/EDA/marital_target.png", caption="Marital status inluence", use_container_width=False)

# Show the second image
st.subheader("9. Business travel influence")
st.image("../assets/EDA/businesstravel_target.png", caption="Business travel status inluence", use_container_width=False)

# Show the second image
st.subheader("10. Interactions: Travel + age vs attrititon")
st.image("../assets/EDA/interaction_travel_age.png", caption="Business travel + age interaction impact", use_container_width=False)


st.subheader("11. Interactions: Travel + marital status vs attrititon")
st.image("../assets/EDA/interaction_travel_marital.png", caption="Business travel + marital status interaction impact", use_container_width=False)



st.subheader("12. Trained models")
st.image("../assets/EDA/model_comparison_bar.png", caption="Trained models metric comparison", use_container_width=False)

st.subheader("12. Random Forest Vs. Catboost")
st.image("../assets/EDA/model_comparison.png", caption="Model metric comparison", use_container_width=False)


st.subheader("13. Chosen Model feature contributions")
st.image("../assets/EDA/shap_forest.png", caption="SHAP values Random Forest", use_container_width=False)
st.image("../assets/EDA/shap_catboost.png", caption="SHAP values Catboost", use_container_width=False)


st.write("Thank you for viewing!")
