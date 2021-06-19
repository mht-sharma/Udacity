import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
from importlib_metadata import version
import requests


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path'], 'finaldata.csv') 
test_data_path = os.path.join(config['test_data_path'], 'testdata.csv') 
model_path = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl') 

##################Function to get model predictions
def model_predictions(dataset):
    #read the deployed model and a test dataset, calculate predictions
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    result = model.predict(dataset)
        
    return result.tolist()  #return value should be a list containing all predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    df = pd.read_csv(dataset_csv_path)
    data = df.drop(['corporation', 'exited'], axis = 1)

    mean = data.mean()
    median = data.median()
    std = data.std()

    stats = [mean, median, std]
    result = pd.concat(stats, axis=1, keys=['mean', 'median', 'std'])
    
    return [result.columns.values.tolist()] + result.values.tolist() #return value should be a list containing all summary statistics

##################Function to get summary statistics
def missing_data_stats():
    #calculate summary statistics here
    df = pd.read_csv(dataset_csv_path)
    data = df.drop(['corporation', 'exited'], axis = 1)

    missing_per = data.isnull().sum() * 100 / len(df)
    
    return missing_per.values.tolist()  #return value should be a list containing all summary statistics

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    #run each function for 10 runs
    num_runs = 10

    #calculate the ingestion time
    ingestion_time = timeit.timeit(
        'ingestion.merge_multiple_dataframe',
        setup='import ingestion',
        number=num_runs) / num_runs

    
    #calculate the model training time
    train_time = timeit.timeit(
        'training.train_model',
        setup='import training',
        number=num_runs) / num_runs
    
    return [ingestion_time, train_time]  #return a list of 2 timing values in seconds

##################Function to return latest version of the package
def get_latest_version(package):
    response = requests.get(f'https://pypi.org/pypi/{package}/json')
    return response.json()['info']['version']

##################Function to check dependencies
def outdated_packages_list():
    #get a list of 
    dependencies = []
    
    with open('requirements.txt', 'r') as f:
        deps = map(lambda x : x.split('==')[0], f.read().split('\n'))

    for dep in deps:
         dependencies.append([dep, version(dep), get_latest_version(dep)])
        
    return dependencies

if __name__ == '__main__':
    df = pd.read_csv(test_data_path)
    X_test = df.drop(['corporation', 'exited'], axis = 1)
  
    model_predictions(X_test)
    dataframe_summary()
    execution_time()
    outdated_packages_list()
    missing_data_stats()


    