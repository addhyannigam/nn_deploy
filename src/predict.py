import pandas as pd
import numpy as np

from src.config import config
import src.preprocessing.preprocessors as pp
from src.preprocessing.data_management import load_dataset, save_model, load_model

import pipeline as pl
import train_pipeline as tt

import pickle

X_test = pd.DataFrame(data={"x1": [0, 0, 1, 1], "x2": [0, 1, 0, 1]})

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_pass(sample):
    z = {}
    h = {}
    h[0] = sample.values.reshape(1, sample.shape[0])

    for l in range(1, config.NUM_LAYERS):
        z[l] = tt.layer_neurons_weighted_sum(h[l-1], pl.theta0[l], pl.theta[l])
        h[l] = tt.layer_neurons_output(z[l], config.f[l])

    final_output = sigmoid(h[config.NUM_LAYERS-1])

    if final_output[0, 0] >= 0.5:
        binary_output = 1
    else:
        binary_output = 0

    return binary_output

if __name__ == "__main__":
    training_data = load_dataset("train.csv")
    obj = pp.preprocess_data()
    obj.fit(training_data.iloc[:, 0:2], training_data.iloc[:, 2])
    X_train, Y_train = obj.transform(training_data.iloc[:, 0:2], training_data.iloc[:, 2])

    pl.initialize_parameters()

    for sample_index in range(len(X_test)):
        sample = X_test.iloc[sample_index]
        binary_output = forward_pass(sample)
        print(f"Binary output for sample {sample_index}: {binary_output}")

    