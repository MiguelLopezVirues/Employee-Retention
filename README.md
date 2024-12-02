[Response #1]  

# 🌟 Employee Turnover Prediction

<div style="text-align: center;">
  <img src="assets/employee_turnover.jpg" alt="Project Cover" />
</div>

## 📝 Project Overview

**Employee Turnover Prediction** is a machine learning classification project designed to address one of the critical challenges in human resources: employee retention. 

By analyzing various factors such as satisfaction, performance, and workload, this project identifies patterns and predicts whether an employee is likely to stay or leave a company. The goal of the project is to extract actionable insights for organizations to improve employee engagement and reduce turnover rates.

---

### 🚀 Features

This project follows a structured approach through the following steps:

1. **Data preliminary exploration and cleaning:**
   - Initial analysis to understand the dataset's structure and variables.
   - Handling null values, duplicates, and inconsistencies.
      - Ensuring data quality for accurate modeling.
2. **Feature Engineering evaluation with sklearn pipelines:**
   - Encoding categorical variables and scaling numeric features.
   - Addressing class imbalance through resampling techniques.
4. **Model evaluation:**
   - Testing multiple classification algorithms, mainly Random Forests, Catboost, XGBoost and Decision trees..
   - Evaluating models on AUC-PR to account for target imbalance.
5. **Model Optimization:**
   - Fine-tuning hyperparameters using techniques like GridSearchCV.
6. **Model selection**
   - Final model selection based on best performing classification threshold.
7. **Interpretability and insights:**
   - Interpretation of developed model via SHAP values.
   - Extraction of actionable insights for business and future  models.

---

## 🏢 Context and Problem Statement

Employee turnover can disrupt organizational efficiency and incur high costs. Understanding the factors that influence turnover is crucial for developing strategies to retain valuable talent. This project uses a fictional dataset containing information from employee and manager surveys, alongside general demographic and performance data, to build a predictive model that identifies at-risk employees and the key factors driving their decisions.

---

### 📊 Dataset Description

The dataset consists of three key files:

- **Employee Survey Data:**
  - Satisfaction levels, work-life balance, and other subjective metrics.
- **Manager Survey Data:**
  - Supervisor evaluations of performance and satisfaction.
- **General Employee Data:**
  - Demographics, job role, salary, and other general information.

Each dataset has detailed documentation available in the `data_dictionary.csv` file for reference.

---

## 📁 Project Structure

```bash
Employee-Retention/
├── assets/                     # Images or visual assets
├── data/                       # Raw data files and dictionaries
├── deployment/                 # Deployment-related files
├── notebooks/                  # Jupyter notebooks for analysis and modeling
│   ├── 1_exploration_cleaning.ipynb
│   ├── 2_EDA.ipynb
│   ├── 3_preprocessing.ipynb
│   ├── 4_model_evaluation.ipynb
│   ├── 5_EDA_v2.ipynb
│   ├── drafts.ipynb
├── results/                    # Pickle files for models evaluated
├── src/                        # Python scripts for support functions
│   ├── ab_testing_support.py
│   ├── association_metrics.py
│   ├── data_preparation.py
│   ├── data_visualization_support.py
│   ├── model_evaluation_support.py
│   ├── soporte_ajuste_clasificacion.py
│   ├── soporte_eda.py
│   ├── soporte_outliers.py
│   ├── soporte_preprocesamiento.py
├── streamlit/                  # Streamlit app files
├── Pipfile                     # Pipenv configuration file
├── Pipfile.lock                # Locked dependencies
└── README.md                   # Project documentation
```

---

## 🛠️ Installation and Requirements

### Prerequisites

- Python 3.11+
- The following libraries:
[Response #4]

### Libraries and Tools

The project relies on the following Python libraries:

- [streamlit](https://streamlit.io/)  
- [seaborn](https://seaborn.pydata.org/)  
- [pandas](https://pandas.pydata.org/docs/)  
- [numpy](https://numpy.org/doc/)  
- [scipy](https://docs.scipy.org/doc/scipy/)  
- [statsmodels](https://www.statsmodels.org/stable/index.html)  
- [association-metrics](https://pypi.org/project/association-metrics/)  
- [scikit-learn](https://scikit-learn.org/stable/documentation.html)  
- [scikit-posthocs](https://scikit-posthocs.readthedocs.io/)  
- [pingouin](https://pingouin-stats.org/)  
- [category-encoders](https://contrib.scikit-learn.org/category_encoders/)  
- [xgboost](https://xgboost.readthedocs.io/)  
- [catboost](https://catboost.ai/)  
- [shap](https://shap.readthedocs.io/)  
- [imbalanced-learn](https://imbalanced-learn.org/)  
- [matplotlib](https://matplotlib.org/stable/users/index.html)  
- [flask](https://flask.palletsprojects.com/)  

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/username/employee-turnover-prediction
   cd employee-turnover-prediction
   ```

2. Install dependencies:

   - Using Pipenv:

     ```bash
     pipenv install
     pipenv shell
     ```
   - Using pip:
   
     ```bash
     pip install -r requirements.txt
     ```


---
### Usage

For streamlit application:
1. Go to ``deployment/`` folder, activate environment and run to activate the Flask API serving:

     ```bash
     python main.py
     ```
2. Go to ``streamlit/`` folder, activate environment and run 

     ```bash
     streamlit run turnover_predictor.py
     ```
---

## 🔄 Next Steps

- Tune Random Forest further, as high
- [Explain a bit more] Explore how GridSearchCV can give worse results for a single parameter optimized, with options included that give better performance when alone. Assuming correct seed control. 
- Different employees might represent different ROIs of employee retention, thus, it might be a good a idea to either:
    - Develop different models for different profiles based on risk (costs vs. probability)
    - Apply different classification thresholds for different profiles
- Use ordinal or custom encodings to replace target ones, to allow for SHAP values interpretability.

---

## 🤝 Contributions

Contributions are welcome! Feel free to fork the repository, create pull requests, or raise issues for discussion.

---

## ✒️ Authors

- **Miguel López Virués** - [GitHub Profile](https://github.com/MiguelLopezVirues)

---

## 📜 License

This project is licensed under the MIT License.