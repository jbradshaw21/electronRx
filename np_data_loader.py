import numpy as np


np.random.seed(42)


def np_data_loader(train_data, test_data, h_features):

    np_train_data, np_test_data, load_train = [None, None, True]
    for data_set in (train_data, test_data):
        np_empty = True
        for feature in h_features:
            np_row = []
            for data in data_set[feature]:
                np_data = data.numpy()
                np_row.append(np.average(np_data))

            np_row = np.reshape(np.array(np_row), newshape=(len(np_row), 1))
            if np_empty and load_train:
                np_train_data = np_row
                np_empty = False
            elif not np_empty and load_train:
                np_train_data = np.concatenate((np_train_data, np_row), axis=1)
            elif np_empty:
                np_test_data = np_row
                np_empty = False
            else:
                np_test_data = np.concatenate((np_test_data, np_row), axis=1)

        load_train = False

    return np_train_data, np_test_data
