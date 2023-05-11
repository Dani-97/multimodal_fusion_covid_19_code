import numpy as np
from sklearn import preprocessing

class No_Preprocessing():

    def __init__(self, **kwargs):
        pass

    def execute_preprocessing(self, input_data, attrs_headers):
        return input_data, attrs_headers

class Standardization_Preprocessing(No_Preprocessing):

    def __init__(self, **kwargs):
        # self.preprocessing_module = preprocessing.StandardScaler()
        pass

    def execute_preprocessing(self, input_data, attrs_headers):
        # self.preprocessing_module.fit(input_data)
        # preprocessed_data = self.preprocessing_module.transform(input_data)
        preprocessed_data = (input_data - input_data.mean(axis=0))/input_data.std(axis=0)
        valid_idxs = np.where((~np.isnan(preprocessed_data).any(axis=0))+(np.std(preprocessed_data, axis=0)>0))[0]
        preprocessed_data = preprocessed_data[:, valid_idxs]
        attrs_headers = np.array(attrs_headers)[valid_idxs].tolist()

        return preprocessed_data, attrs_headers

class Normalization_Preprocessing(No_Preprocessing):

    def __init__(self, **kwargs):
        # self.preprocessing_module = preprocessing.Normalizer()
        pass

    def execute_preprocessing(self, input_data, attrs_headers):
        # self.preprocessing_module.fit(input_data)
        # preprocessed_data = self.preprocessing_module.transform(input_data)
        preprocessed_data = (input_data - input_data.min(axis=0))/(input_data.max(axis=0) - input_data.min(axis=0))
        valid_idxs = np.where((~np.isnan(preprocessed_data).any(axis=0))+(np.std(preprocessed_data, axis=0)>0))[0]
        preprocessed_data = preprocessed_data[:, valid_idxs]
        attrs_headers = np.array(attrs_headers)[valid_idxs].tolist()

        return preprocessed_data, attrs_headers
