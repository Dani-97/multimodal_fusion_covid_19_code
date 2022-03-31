import csv
import numpy as np
from ReliefF import ReliefF
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
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

    # This function stores in a csv file the report of the feature selection
    # or feature extraction process. By default, in this super class, it does
    # not do anything, so it works like an abstract method that must be
    # overwritten by the child class.
    def store_report(self, csv_file_path, attrs_headers, append=True):
        pass

    def set_dir_to_store_results(self, dir_to_store_results):
        self.dir_to_store_results = dir_to_store_results

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

    def store_report(self, csv_file_path, attrs_headers, append=True):
        pass

class ReliefF_Feature_Retrieval(Super_Feature_Retrieval):

    def __init__(self, **kwargs):
        print('++++ The ReliefF algorithm has been chosen for feature selection')
        self.noftopfeatures = kwargs['noftopfeatures']
        print('---- Number of top features: %d'%self.noftopfeatures)

    def execute_feature_retrieval(self, input_data, output_data, plot_data=False):
        super().execute_feature_retrieval(input_data, output_data, plot_data)
        reliefF_algorithm = ReliefF(n_neighbors=5, n_features_to_keep=self.noftopfeatures)
        input_data = input_data.astype(np.float64)
        output_data = output_data.astype(np.float64)
        output_selection = reliefF_algorithm.fit_transform(input_data[:, :], output_data)
        self.reliefF_report = reliefF_algorithm.__dict__

        return output_selection

    def __give_names_to_features(self, attrs_headers, \
                                   top_features_list, features_scores_list):
        top_features_with_names_list = []

        for item_aux in top_features_list:
            item_with_name_aux = item_aux + ': ' + \
                      attrs_headers[int(item_aux)] + \
                        ' (score = %d)'%features_scores_list[int(item_aux)]
            top_features_with_names_list.append(item_with_name_aux)

        return top_features_with_names_list

    def __store_report_to_csv__(self, csv_file_path, attrs_headers, append=True):
        if (append):
            file_mode = 'a'
        else:
            file_mode = 'w'

        with open(csv_file_path, file_mode) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')

            top_features = np.array(self.reliefF_report['top_features']).astype(str)
            features_scores = np.array(self.reliefF_report['feature_scores']).astype(np.float64)
            top_features_with_names = \
                    self.__give_names_to_features(attrs_headers, top_features, features_scores)
            csv_writer.writerow(['top_features'])
            csv_writer.writerow(top_features_with_names)
            csv_writer.writerow([])

    # This function plots the report of the scores and the ranking of the
    # attributes after the feature selection algorithm is applied.
    def __plot_report__(self, attrs_headers, dir_to_store_results):
        top_features_idx = np.flip(np.array(self.reliefF_report['top_features']))
        features_scores = np.array(self.reliefF_report['feature_scores']).astype(np.float64)
        features_scores = features_scores[top_features_idx]

        fig, ax = plt.subplots(figsize=(27.5, 10))
        y_pos = list(range(len(features_scores)))
        ax.barh(y_pos, features_scores, align='center')

        plt.title('Top features scores ranking')
        plt.tick_params(labeltop=True, labelright=True)
        plt.yticks(y_pos, np.array(attrs_headers)[top_features_idx])
        plt.xlabel('Feature score')
        plt.ylabel('Feature')

        # It is important to note that x and y are flipped, because we are
        # using barh.
        for x_coordinate, y_coordinate in enumerate(features_scores):
            text_to_show = str(int(y_coordinate))
            if (y_coordinate<0):
                displacement = -(len(text_to_show)+20)
            else:
                displacement = len(text_to_show)
            ax.text(y_coordinate + displacement, x_coordinate - 0.25, text_to_show, color='black', fontweight='bold')

        # ax.grid(True)

        output_filename = '%s/%s'%(dir_to_store_results, 'reliefF_report.pdf')
        plt.savefig(output_filename)
        print('++++ The report of ReliefF top features ranking has been stored at %s'%output_filename)

    def store_report(self, csv_file_path, attrs_headers, append=True):
        self.__store_report_to_csv__(csv_file_path, attrs_headers, append)
        self.__plot_report__(attrs_headers, self.dir_to_store_results)

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

    def store_report(self, csv_file_path, attrs_headers, append=True):
        print('+++++ WARNING: the method store_report is not implemented for PCA!')
