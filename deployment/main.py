import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
import pickle

import sys
sys.path.append("..")
import src.data_preparation as dp
import sklearn


app = Flask(__name__)

pipeline_path = "../results/imp. simple, OHE + TE/random_forest/mejor_modelo.pkl"

# load pipeline
with open(pipeline_path, 'rb') as f:
    pipeline = pickle.load(f)

variables_one = ['age',
 'businesstravel',
 'department',
 'distancefromhome',
 'education',
 'educationfield',
 'gender',
 'joblevel',
 'jobrole',
 'maritalstatus',
 'monthlyincome',
 'numcompaniesworked',
 'percentsalaryhike',
 'stockoptionlevel',
 'totalworkingyears',
 'trainingtimeslastyear',
 'yearsatcompany',
 'yearssincelastpromotion',
 'yearswithcurrmanager',
 'environmentsatisfaction',
 'jobsatisfaction',
 'worklifebalance',
 'jobinvolvement',
 'performancerating']



@app.route("/")
def home():
    return jsonify({"mensaje":"API de prediccion en funcionamiento",
                    "endpoints":{"/predict":"Usa este endpoint para realizar predicciones"}})

@app.route("/predict", methods= ["POST"])
def predict():
    # try:
    json_data = request.get_json()

    X_pred = dp.load_and_clean_json(json_data)

    # ensure the pipeline outputs dataframes between steps
    sklearn.set_config(transform_output="pandas")

    probability = pipeline.predict_proba(X_pred)[:,1]
    prediction = 1 if probability[0] > 0.101 else 0

    return jsonify({"prediction":prediction,
                    "probability": probability[0]})

    # except:
    #     return jsonify({"respuesta":"ha habido un error en la recepción de la información."})

if __name__ == "__main__":
    app.run(debug=True)
