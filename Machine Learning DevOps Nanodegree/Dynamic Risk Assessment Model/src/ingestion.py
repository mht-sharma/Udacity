import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import glob


#############Load config.json and get input and output paths
with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]
prod_deployment_path = os.path.join(config["prod_deployment_path"])

#############Function for data ingestion
def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file
    # get a list of all the csv filenames in the input path
    filenames = glob.glob(input_folder_path + "/*.csv")

    # filter files which are already used in deployment
    dataset_filenames_path = os.path.join(output_folder_path, "ingestedfiles.txt")
    if os.path.exists(dataset_filenames_path):
        with open(dataset_filenames_path, "r") as f:
            ingested_files = f.read().split("\n")
        filenames = list(
            filter(lambda filename: filename not in ingested_files, filenames)
        )

    if len(filenames) > 0:
        print(f"Number of new data files: {len(filenames)}")
        df = []
        df = [pd.read_csv(filename) for filename in filenames]

        # concatenate the datasets in a single dataframe
        df = pd.concat(df, ignore_index=True).drop_duplicates()

        # create the output directory if not exists
        os.makedirs(output_folder_path, exist_ok=True)

        # path to save the dataset
        dataset_csv_path = os.path.join(output_folder_path, "finaldata.csv")

        print(f"Saving the data in {dataset_csv_path}")
        if os.path.exists(dataset_filenames_path):
            df.to_csv(dataset_csv_path, index=False, mode="a", header=False)
        else:
            df.to_csv(dataset_csv_path, index=False)

        with open(dataset_filenames_path, "a") as f:
            for filename in filenames:
                f.write(f"{filename}\n")

        return

    print("No new data found. Exiting!")


if __name__ == "__main__":
    merge_multiple_dataframe()
