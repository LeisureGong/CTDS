import os
import numpy as np
import tslearn.utils.utils as tuu


def load_ucr(dataset_name):
    dataset_path = './datasets'
    train_file = os.path.join(dataset_path, dataset_name, dataset_name + "_TRAIN.tsv")
    test_file = os.path.join(dataset_path, dataset_name, dataset_name + '_TEST.tsv')
    train_data = np.loadtxt(train_file)
    test_data = np.loadtxt(test_file)
    X_train = tuu.to_time_series_dataset(train_data[:, 1:])
    y_train = train_data[:, 0].astype(np.int)
    X_test = tuu.to_time_series_dataset(test_data[:, 1:])
    y_test = test_data[:, 0].astype(np.int)
    return X_train, y_train, X_test, y_test
