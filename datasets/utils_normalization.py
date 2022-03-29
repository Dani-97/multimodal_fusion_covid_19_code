from sklearn import preprocessing

class No_Preprocessing():

    def __init__(self, **kwargs):
        pass

    def execute_preprocessing(self, input_data):
        return input_data

class Standardization_Preprocessing():

    def __init__(self, **kwargs):
        self.preprocessing_module = preprocessing.StandardScaler()

    def execute_preprocessing(self, input_data):
        self.preprocessing_module.fit(input_data)
        preprocessed_data = self.preprocessing_module.transform(input_data)

        return preprocessed_data

class Normalization_Preprocessing():

    def __init__(self, **kwargs):
        self.preprocessing_module = preprocessing.Normalizer()

    def execute_preprocessing(self, input_data):
        self.preprocessing_module.fit(input_data)
        preprocessed_data = self.preprocessing_module.transform(input_data)

        return preprocessed_data
