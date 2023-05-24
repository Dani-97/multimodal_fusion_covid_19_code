import numpy as np
from sklearn.impute import KNNImputer

# This acts as an abstract class.
class Super_Values_Imputation_Model():

    def __init__(self, **kwargs):
        pass

    def execute_imputation(self, input_data, output_data):
        raise NotImplementedError("++++ ERROR: The execute_imputation method has not been implemented!")

class No_Imputation_Model(Super_Values_Imputation_Model):

    def __init__(self, **kwargs):
        pass

    def execute_imputation(self, input_data, output_data):
        return input_data, output_data

class kNN_Values_Imputation_Model(Super_Values_Imputation_Model):

    def __init__(self, **kwargs):
        print('++++ The imputation of the missing values will be performed with the kNN algorithm.')

    def execute_imputation(self, input_data, output_data):
        imputer = KNNImputer(missing_values=-1.0, n_neighbors=2, weights="uniform")
        transformed_input_data = imputer.fit_transform(input_data)

        return transformed_input_data, output_data
