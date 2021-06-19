import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import diagnostics


###############Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

plot_save_path = os.path.join(config["output_model_path"], "confusionmatrix.png")
test_data_path = os.path.join(config["test_data_path"], "testdata.csv")


##############Function for reporting
def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    df = pd.read_csv(test_data_path)
    X_test = df.drop(["corporation", "exited"], axis=1)
    y_true = df.exited

    y_pred = diagnostics.model_predictions(X_test)

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    confusion_matrix_plot = sns.heatmap(confusion_matrix, annot=True)
    confusion_matrix_plot.figure.savefig(plot_save_path)


if __name__ == "__main__":
    score_model()
