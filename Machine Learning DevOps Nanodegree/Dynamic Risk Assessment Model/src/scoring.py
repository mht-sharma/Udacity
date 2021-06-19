from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'], 'testdata.csv') 
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl') 
score_path = os.path.join(config['output_model_path'], 'latestscore.txt') 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    df = pd.read_csv(test_data_path)
     
    X_test = df.drop(['corporation', 'exited'], axis = 1)
    y_true = df.exited
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    
    f1_score = metrics.f1_score(y_true, y_pred)
  
    with open(score_path, 'w') as f:
            f.write(str(f1_score))
            
    return f1_score