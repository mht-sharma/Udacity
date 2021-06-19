import requests
import pickle
import json
import os

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

#Call each API endpoint and store the responses
response1 = requests.post(f'{URL}/prediction',  headers={'Content-Type': 'application/json'}, data=json.dumps({'dataset_path':'testdata/testdata.csv'}))
response2 = requests.get(f'{URL}/scoring')
response3 = requests.get(f'{URL}/summarystats')
response4 = requests.get(f'{URL}/diagnostics')

#combine all API responses
responses = {
    "prediction": response1.text,
    "scoring": response2.text,
    "summarystats": response3.text,
    "diagnostics": response4.text
}

#write the responses to your workspace
with open('config.json','r') as f:
    config = json.load(f) 

api_returns_path = os.path.join(config['output_model_path'], 'apireturns.txt') 
                                     
with open(api_returns_path, 'w') as f:
    for response in responses:
        f.write(f'{response}: {responses[response]}\n\n')
