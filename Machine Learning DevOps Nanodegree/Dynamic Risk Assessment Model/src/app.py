from flask import Flask, session, jsonify, request, abort
import pandas as pd
import numpy as np
import pickle
import scoring
import diagnostics
import json
import os


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'
 
######### ##########Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    if not request.json or not 'dataset_path' in request.json:
        abort(400)
    dataset_path = request.json['dataset_path'] 

    df = pd.read_csv(dataset_path)
    dataset = df.drop(['corporation', 'exited'], axis = 1)
    result = diagnostics.model_predictions(dataset)
    
    return jsonify(result), 200  #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring_stats():        
    #check the score of the deployed model
    score = scoring.score_model()
    return jsonify(score), 200  #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary_stats():        
    #check means, medians, and modes for each column
    summary = diagnostics.dataframe_summary()
    
    return jsonify(summary), 200  #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics_stats():        
    #check timing and percent NA values
    timing = diagnostics.execution_time()
    dependency_check = diagnostics.outdated_packages_list()
    missing_data = diagnostics.missing_data_stats()
    
    response = {
        "timing": timing,
        "missing_data": missing_data,
        "dependepncy_check": dependency_check
    }

    return jsonify(response), 200  #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
