import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt

def plot_2D_distribution(input_data, output_data):
    plt.figure()
    plt.scatter(input_data[output_data=='0', 0], input_data[output_data=='0', 1], color='red')
    plt.scatter(input_data[output_data=='1', 0], input_data[output_data=='1', 1], color='blue')
    plt.show()

def plot_3D_distribution(input_data, output_data):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter3D(input_data[output_data=='0', 0], input_data[output_data=='0', 1], input_data[output_data=='0', 2], c=input_data[output_data=='0', 2], cmap='Greens')
    ax.scatter3D(input_data[output_data=='1', 0], input_data[output_data=='1', 1], input_data[output_data=='1', 2], c=input_data[output_data=='1', 2], cmap='Reds')
    plt.show()

def plot_features_distribution(input_data, output_data):
    input_data_shape = np.shape(input_data)
    # Number of attributes of the dataset.
    nofattributes = input_data_shape[1]

    if (nofattributes==2):
        plot_2D_distribution(input_data, output_data)
    elif (nofattributes==3):
        plot_3D_distribution(input_data, output_data)
    else:
        print('+++++ This dataset has %d attributes, so we '%nofattributes + \
                'cannot plot its data. Only 2D and 3D datasets are allowed')

class Super_Feature_Retrieval():

    def __init__(self, **kwargs):
        pass

    # If plot_data=True, this function will plot the data. By default, it does
    # not return anything. Therefore, if you want to perform feature retrieval,
    # this method must be overloaded by their children classes.
    def execute_feature_retrieval(self, input_data, output_data, plot_data=False):
        if (plot_data):
            plot_features_distribution(input_data, output_data)

# An object of this class will be instantiated if no feature selection was
# chosen.
class No_Feature_Retrieval(Super_Feature_Retrieval):

    def __init__(self, **kwargs):
        print('++++ No feature selection or extraction will be performed')

    # If no feature selection is chosen, then the output of this function will
    # be exactly the same as the input data of the dataset. In case it is
    # desired and possible, the data will be plotted.
    def execute_feature_retrieval(self, input_data, output_data, plot_data=False):
        super().execute_feature_retrieval(input_data, output_data, plot_data)

        return input_data

class SelectKBest_Feature_Retrieval(Super_Feature_Retrieval):

    def __init__(self, **kwargs):
        print('++++ The SelectKBest has been chosen for feature selection')
        self.noftopfeatures = kwargs['noftopfeatures']
        print('---- Number of top features: %d'%self.noftopfeatures)

    def execute_feature_retrieval(self, input_data, output_data, plot_data=False):
        super().execute_feature_retrieval(input_data, output_data, plot_data)
        output_selection = SelectKBest(chi2, \
                                  k=self.noftopfeatures).fit_transform(\
                                                      input_data, output_data)

        return output_selection

class PCA_Feature_Retrieval(Super_Feature_Retrieval):

    def __init__(self, **kwargs):
        print('++++ The PCA algorithm has ben chosen for feature selection')
        self.nofcomponents = kwargs['nofcomponents']
        print('---- Number of components: %d'%self.nofcomponents)

    def execute_feature_retrieval(self, input_data, output_data, plot_data=False):
        pca = PCA(n_components=self.nofcomponents)

        pca.fit(input_data, output_data)
        transformed_data = pca.transform(input_data)
        super().execute_feature_retrieval(transformed_data, output_data, plot_data)

        return transformed_data
