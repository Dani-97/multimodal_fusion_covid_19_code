import numpy as np
from sklearn import preprocessing

class No_Preprocessing():

    def __init__(self, **kwargs):
        pass

    def execute_preprocessing(self, input_data, attrs_headers, params=None):
        return input_data, attrs_headers

class Normalization_Preprocessing(No_Preprocessing):

    def __init__(self, **kwargs):
        pass

    # If params is None, then the min and the max from the input data will be
    # used. Otherwise, those values will be the ones specified by this variable.
    # The used params will be returned.
    def execute_preprocessing(self, input_data, attrs_headers, params=None):
        preprocessed_data = np.copy(input_data)
        nofcolumns = preprocessed_data.shape[1]
        max_values, min_values = [], []
        for current_column_aux in range(0, nofcolumns):
            if (params is None):
                current_min_value = np.min(input_data[:, current_column_aux])
                current_max_value = np.max(input_data[:, current_column_aux])
                min_values.append(current_min_value)
                max_values.append(current_max_value)
            else:
                min_values, max_values = params
                current_max_value = max_values[current_column_aux]
                current_min_value = min_values[current_column_aux]

            if ((current_max_value - current_min_value)>0):
                preprocessed_data[:, current_column_aux] = \
                    (preprocessed_data[:, current_column_aux] - current_min_value)/(current_max_value - current_min_value)
            else:
                preprocessed_data[:, current_column_aux] = np.array([0]*len(preprocessed_data[:, current_column_aux]))

        params = (np.array(min_values), np.array(max_values))

        return preprocessed_data, attrs_headers, params
