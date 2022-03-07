import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# An object of this class will be instantiated if no feature selection was
# chosen.
class No_Feature_Retrieval():

    def __init__(self, **kwargs):
        print('++++ No feature selection or extraction will be performed')

    # If no feature selection is chosen, then the output of this function will
    # be exactly the same as the input data of the dataset.
    def execute_feature_retrieval(self, input_data, output_data):
        return input_data

class SelectKBest_Feature_Retrieval():

    def __init__(self, **kwargs):
        print('++++ The SelectKBest has been chosen for feature selection')
        self.noftopfeatures = kwargs['noftopfeatures']
        print('---- Number of top features: %d'%self.noftopfeatures)

    def execute_feature_retrieval(self, input_data, output_data):
        output_selection = SelectKBest(chi2, \
                                  k=self.noftopfeatures).fit_transform(\
                                                      input_data, output_data)

        return output_selection

class PCA_Feature_Retrieval():

    def __init__(self, **kwargs):
        print('++++ The PCA algorithm has ben chosen for feature selection')

    def execute_feature_retrieval(self, input_data, output_data):
        pca = PCA(n_components=2)

        pca.fit(input_data, output_data)
        transformed_data = pca.transform(input_data)

        return transformed_data
