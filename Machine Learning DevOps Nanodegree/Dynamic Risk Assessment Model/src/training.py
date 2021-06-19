from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
import json
from sklearn.ensemble import RandomForestClassifier

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
model_path = os.path.join(config['output_model_path'], 'trainedmodel.pkl')


#################Function for training the model
def train_model():
    #use this random forest classifier for training
    model = RandomForestClassifier(max_depth=5, random_state=0)

    #fit the logistic regression to your data
    df = pd.read_csv(dataset_csv_path)
    
    X_train = df.drop(['corporation', 'exited'], axis = 1)
    y_train = df.exited

    model.fit(X_train, y_train)

    #write the trained model to your workspace in a file called trainedmodel.pkl
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    return model_path
