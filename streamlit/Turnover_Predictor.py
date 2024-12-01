import streamlit as st
import pandas as pd
import pickle
import requests

# Streamlit config
st.set_page_config(
    page_title="Employee Turnover Prediction",
    page_icon="👔",
    layout="centered",
)

# Title description
st.title("👔 Employee Turnover Prediction")
st.write(
    "Introduce the details of an employee to predict the likelihood of attrition. "
    "Make informed decisions to retain your top talent!"
)

# Mostrar una imagen llamativa
st.image(
    "https://plus.unsplash.com/premium_photo-1663011639930-de993311fd95?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", 
    caption="""Make data-driven decisions to reduce employee turnover... 
    Illo illo! No lea tanto que se va! (Boy, boy, don't read so much, he's leaving.)""",
    use_container_width=True,
)

# Input form
st.header("🔧 Employee Details")

# Split input form into two columns for a cleaner layout
col1, col2 = st.columns(2)

# Inputs for employee details
with col1:
    age = st.number_input("Age", min_value=18, max_value=60, value=37, step=1)
    businesstravel = st.selectbox(
        "Business Travel Frequency", 
        ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'], 
        index=0  # Default: 'Travel_Rarely'
    )
    department = st.selectbox(
        "Department", 
        ['Sales', 'Research & Development', 'Human Resources'], 
        index=1  # Default: 'Research & Development'
    )
    distancefromhome = st.number_input("Distance From Home (km)", min_value=1, max_value=29, value=9, step=1)
    education = st.selectbox(
        "Education Level", 
        [1, 2, 3, 4, 5], 
        index=4  # Default: 5
    )
    educationfield = st.selectbox(
        "Education Field", 
        ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'], 
        index=4  # Default: 'Life Sciences'
    )
    gender = st.selectbox(
        "Gender", 
        ['Female', 'Male'], 
        index=1  # Default: 'Male'
    )
    joblevel = st.selectbox(
        "Job Level", 
        [1, 2, 3, 4, 5], 
        index=3  # Default: 4
    )
    yearssincelastpromotion = st.selectbox(
        "Years Since Last Promotion", 
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 
        index=3  # Default: 3
    )
    yearswithcurrmanager = st.selectbox(
        "Years with Current Manager", 
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], 
        index=5  # Default: 5
    )
    worklifebalance = st.selectbox(
        "Work-Life Balance", 
        [1, 2, 3, 4], 
        index=2  # Default: 3
    )
    jobinvolvement = st.selectbox(
        "Job Involvement", 
        [1, 2, 3, 4], 
        index=2  # Default: 3
    )

with col2:
    jobrole = st.selectbox(
        "Job Role",
        ['Healthcare Representative', 'Research Scientist', 'Sales Executive', 'Human Resources',
         'Research Director', 'Laboratory Technician', 'Manufacturing Director', 'Sales Representative', 'Manager'],
        index=6  # Default: 'Manufacturing Director'
    )
    maritalstatus = st.selectbox(
        "Marital Status", 
        ['Married', 'Single', 'Divorced'], 
        index=0  # Default: 'MArried'
    )
    monthlyincome = st.number_input("Monthly Income (₹)", min_value=10090, max_value=199990, value=65000, step=1000)
    numcompaniesworked = st.selectbox(
        "Number of Companies Worked", 
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 
        index=2 # Default: 2
    )
    percentsalaryhike = st.selectbox(
        "Percent Salary Hike (%)", 
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25], 
        index=8  # Default: 19
    )
    stockoptionlevel = st.selectbox(
        "Stock Option Level", 
        [0, 1, 2, 3], 
        index=2  # Default: 2
    )
    totalworkingyears = st.number_input("Total Working Years", min_value=0, max_value=40, value=11, step=1)
    trainingtimeslastyear = st.selectbox(
        "Training Times Last Year", 
        [0, 1, 2, 3, 4, 5, 6], 
        index=2  # Default: 2
    )
    yearsatcompany = st.number_input("Years at Company", min_value=0, max_value=40, value=7, step=1)
    environmentsatisfaction = st.selectbox(
        "Environment Satisfaction", 
        [1, 2, 3, 4], 
        index=2  # Default: 3
    )
    jobsatisfaction = st.selectbox(
        "Job Satisfaction", 
        [1, 2, 3, 4], 
        index=2  # Default: 3
    )
    performancerating = st.selectbox(
        "Performance Rating", 
        [3, 4], 
        index=0  # Default: 3
    )




# Collect the input data into a dictionary
input_data = {
    "age": age,
    "businesstravel": businesstravel,
    "department": department,
    "distancefromhome": distancefromhome,
    "education": education,
    "educationfield": educationfield,
    "gender": gender,
    "joblevel": joblevel,
    "jobrole": jobrole,
    "maritalstatus": maritalstatus,
    "monthlyincome": monthlyincome,
    "numcompaniesworked": numcompaniesworked,
    "percentsalaryhike": percentsalaryhike,
    "stockoptionlevel": stockoptionlevel,
    "totalworkingyears": totalworkingyears,
    "trainingtimeslastyear": trainingtimeslastyear,
    "yearsatcompany": yearsatcompany,
    "yearssincelastpromotion": yearssincelastpromotion,
    "yearswithcurrmanager": yearswithcurrmanager,
    "environmentsatisfaction": environmentsatisfaction,
    "jobsatisfaction": jobsatisfaction,
    "worklifebalance": worklifebalance,
    "jobinvolvement": jobinvolvement,
    "performancerating": performancerating,
}


# Custom CSS for styling the button and prediction result
st.markdown("""
    <style>
    .center-button {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }
    .prediction-box {
        margin-top: 20px;
        padding: 20px;
        border-radius: 10px;
        font-size: 18px;
        text-align: center;
        font-weight: bold;
    }
    .attrition-yes {
        background-color: #ffcccc;
        color: #cc0000;
    }
    .attrition-no {
        background-color: #ccffcc;
        color: #008000;
    }
    </style>
""", unsafe_allow_html=True)

# Button to make predictions
st.markdown('<div class="center-button">', unsafe_allow_html=True)
if st.button("Predict Attrition"):

    # Add a spinner while waiting for the response
    with st.spinner("Predicting attrition..."):
        res = requests.post("http://127.0.0.1:5000/predict", json=input_data)
        attrition = res.json()["prediction"]

    # Display prediction result with styled feedback
    if attrition:
        st.markdown(
            '<div class="prediction-box attrition-yes">'
            'The employee will probably leave the company. 🚨💼🚪'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="prediction-box attrition-no">'
            "We're safe with this one. 🍀😌👍"
            '</div>',
            unsafe_allow_html=True
        )
st.markdown('</div>', unsafe_allow_html=True)


# Pie de página
st.markdown(
    """









    ---
    **Project powered by Data Magic 🎩✨🦄.**  
    """,
    unsafe_allow_html=True,
)