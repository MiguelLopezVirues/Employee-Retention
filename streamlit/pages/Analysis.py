import streamlit as st

st.set_page_config(
    page_title="EDA Presentation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Exploratory Data Analysis Presentation")
st.write("This is a curated presentation of key EDA findings.")

# Show the first image
st.subheader("1. Distribution of Age")
st.image("images/age_distribution.png", caption="Distribution of Employee Ages", use_container_width=True)

# Show the second image
st.subheader("2. Correlation Matrix")
st.image("images/correlation_matrix.png", caption="Correlation Between Features", use_container_width=True)

# Show the third image
st.subheader("3. Job Satisfaction Levels")
st.image("images/job_satisfaction.png", caption="Job Satisfaction Across Employees", use_container_width=True)

st.write("Thank you for viewing!")
