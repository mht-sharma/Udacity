import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion
import sys
import json
import glob
import os
import subprocess


with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
prod_deployment_path = os.path.join(config["prod_deployment_path"])

#################Check and read new data
# first, read ingestedfiles.txt
dataset_filenames_path = os.path.join(prod_deployment_path, "ingestedfiles.txt")
with open(dataset_filenames_path, "r") as f:
    ingested_files = f.read().split("\n")

# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
filenames = glob.glob(input_folder_path + "/*.csv")
filenames = list(filter(lambda filename: filename not in ingested_files, filenames))

##################Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
# completed in above line
if len(filenames) == 0:
    print("No new data found. Exiting!")
    sys.exit()

print(f"Number of new data files: {len(filenames)}")
print("Merging the files...")
ingestion.merge_multiple_dataframe()

##################Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
score_path = os.path.join(prod_deployment_path, "latestscore.txt")
with open(score_path, "r") as f:
    prev_score = float(f.read())

print("Training the model on new data...")
training.train_model()
new_score = scoring.score_model()

print("Checking model drift...")
model_drift = new_score > prev_score

##################Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here
if not model_drift:
    print("No model drift detected. Exiting!")
    sys.exit()

##################Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
print("Model drift detected. Model deployment in progress!")
deployment.store_model_into_pickle()

##################Diagnostics and reporting
# run diagnostics.py (apicalls.py will run the necessary methods in diagonostics) and reporting.py for the re-deployed model
print("Reporting the results...")
subprocess.run("python src/reporting.py", shell=True, check=False)
subprocess.run("python src/apicalls.py", shell=True, check=False)

print("Completed :)")
