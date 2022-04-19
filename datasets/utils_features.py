import configparser
import csv
import numpy as np
from scipy.sparse import *
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector, VarianceThreshold
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
import math
import matplotlib.pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import os
from skfeature.utility import construct_W

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

    def set_attributes_types(self, attrs_types):
        self.attrs_types = attrs_types

    def translate_fields(self, attrs_headers, config):
        translated_attrs_headers = []

        for attr_header_aux in attrs_headers:
            translated_attrs_headers.append(config['TRANSLATIONS'][attr_header_aux])

        return translated_attrs_headers

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

class SequentialSelector_Feature_Retrieval(Super_Feature_Retrieval):

    def __init__(self, **kwargs):
        print('++++ A Sequential Selector has been chosen for feature selection')
        self.noftopfeatures = kwargs['noftopfeatures']
        print('---- Number of top features: %d'%self.noftopfeatures)

    def execute_feature_retrieval(self, input_data, output_data, plot_data=False):
        knn = KNeighborsClassifier()
        tmp_selected_features = []
        # Total number of features that are included in the dataset.
        total_noffeatures = np.shape(input_data)[1]

        self.ordered_features_idxs = []
        self.subsets_scores_list = []
        self.feature_selector = SFS(knn,
                                    k_features=np.shape(input_data)[1],
                                    forward=True,
                                    floating=False,
                                    verbose=0,
                                    scoring='roc_auc',
                                    cv=0)

        self.feature_selector.fit(input_data.astype(np.float64), output_data)

        for noffeatures_aux in range(1, total_noffeatures+1):
            # This line retrieves the feature that was selected in the current
            # step of the algorithm.
            current_selected_feature = list(set(list(self.feature_selector.subsets_[noffeatures_aux]['feature_idx'])) - \
                     set(tmp_selected_features))
            tmp_selected_features+=current_selected_feature
            self.ordered_features_idxs.append(current_selected_feature[0])
            current_subset_score_aux = self.feature_selector.subsets_[noffeatures_aux]['avg_score']
            self.subsets_scores_list.append(current_subset_score_aux)

        noftopfeatures_idxs = self.ordered_features_idxs[0:self.noftopfeatures]
        transformed_data = input_data[:, noftopfeatures_idxs]

        return transformed_data

    def __plot_report__(self, attrs_headers, dir_to_store_results):
        top_features_idx = self.ordered_features_idxs
        features_scores = np.array(self.subsets_scores_list).astype(np.float64)
        features_scores = features_scores[top_features_idx]

        fig, ax = plt.subplots(figsize=(30, 30))
        y_pos = list(range(len(features_scores)))
        ax.barh(y_pos, features_scores, align='center')

        plt.title('Top features scores ranking')
        plt.tick_params(labeltop=True, labelright=True)
        plt.yticks(y_pos, np.array(attrs_headers)[top_features_idx])
        # plt.xlabel('Feature score')
        plt.ylabel('Feature')

        # It is important to note that x and y are flipped, because we are
        # using barh.
        for x_coordinate, y_coordinate in enumerate(features_scores):
            if (math.isnan(y_coordinate)):
                y_coordinate = 0
                text_to_show = 'No variability'
            else:
                text_to_show = str(int(y_coordinate))
            ax.text(y_coordinate + 0.5, x_coordinate - 0.25, text_to_show, color='black', fontweight='bold')

        output_filename = '%s/%s'%(dir_to_store_results, 'seq_selector_report.pdf')
        plt.savefig(output_filename)
        print('++++ The report of the Sequential Selector top features ranking has been stored at %s'%output_filename)

    def __store_report_to_csv__(self, csv_file_path, attrs_headers, append=True):
        report_list = []
        if (append):
            file_mode = 'a'
        else:
            file_mode = 'w'

        attrs_headers = np.array(attrs_headers)
        it = 0
        for feature_idx_aux in self.ordered_features_idxs:
            current_line_str_aux = '%d: %s (score: %.4f)'%(feature_idx_aux, \
                              attrs_headers[feature_idx_aux], self.subsets_scores_list[it])
            report_list.append(current_line_str_aux)
            it+=1

        with open(csv_file_path, file_mode) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')

            csv_writer.writerow(['top_features'])
            csv_writer.writerow(report_list)
            csv_writer.writerow([])

    def store_report(self, csv_file_path, attrs_headers, append=True):
        self.__store_report_to_csv__(csv_file_path, attrs_headers)
        self.__plot_report__(attrs_headers, self.dir_to_store_results)

class VarianceThreshold_Feature_Retrieval(Super_Feature_Retrieval):

    def __init__(self, **kwargs):
        print('++++ The Variance Threshold algorithm has been chosen for feature selection')
        self.noftopfeatures = kwargs['noftopfeatures']
        print('---- Number of top features: %d'%self.noftopfeatures)

    def execute_feature_retrieval(self, input_data, output_data, plot_data=False):
        self.feature_selector = VarianceThreshold()
        self.feature_selector.fit(input_data)

        self.attrs_variances = self.feature_selector.variances_
        self.ordered_features_idxs = np.flip(np.argsort(self.attrs_variances))
        top_features_idxs = self.ordered_features_idxs[list(range(0, self.noftopfeatures))]

        transformed_data = input_data[:, top_features_idxs]

        return transformed_data

    def __plot_report__(self, attrs_headers, dir_to_store_results):
        top_features_idx = self.ordered_features_idxs
        features_scores = np.array(self.attrs_variances).astype(np.float64)
        features_scores = np.flip(features_scores[top_features_idx])

        config = configparser.ConfigParser()
        config.read('./datasets/translations.cfg')
        translated_attrs_headers = super().translate_fields(attrs_headers, config)

        plt.rcParams['font.size'] = '50'
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
        fig, ax = plt.subplots(figsize=(30, 30))
        y_pos = list(range(len(features_scores)))

        my_cmap = plt.get_cmap("Purples")
        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
        ax.barh(y_pos, features_scores, edgecolor='black', \
                                 align='center', color=my_cmap(rescale(y_pos)))

        # plt.title('Top features scores ranking')
        plt.yticks(np.flip(y_pos), \
                        np.array(translated_attrs_headers)[top_features_idx])
        plt.xscale('log')
        # plt.xlabel('Feature score')
        # plt.ylabel('Feature')

        # It is important to note that x and y are flipped, because we are
        # using barh.

        '''
            NOTE: UNCOMMENT THESE LINES TO SHOW THE SCORE NEXT TO EACH BAR!
        '''
        # for x_coordinate, y_coordinate in enumerate(features_scores):
        #     if (math.isnan(y_coordinate)):
        #         y_coordinate = 0
        #         text_to_show = 'No variability'
        #     else:
        #         text_to_show = '%.2E'%(float(y_coordinate))
        #     ax.text(y_coordinate, x_coordinate - 0.25, text_to_show, color='black', fontweight='bold')

        plt.xlim([10**(-3), 10**(6)])
        plt.subplots_adjust(left=0.25, bottom=0.05, right=0.70, top=0.97)
        output_filename = '%s/%s'%(dir_to_store_results, 'var_thresh_selector_report.pdf')
        # plt.yticks(rotation=30)
        plt.savefig(output_filename)
        print('++++ The report of the Variance Threshold Selector top features ranking has been stored at %s'%output_filename)

    def __store_report_to_csv__(self, csv_file_path, attrs_headers, append=True):
        report_list = []
        if (append):
            file_mode = 'a'
        else:
            file_mode = 'w'

        attrs_headers = np.array(attrs_headers)
        it = 0
        for feature_idx_aux in self.ordered_features_idxs:
            current_line_str_aux = '%d: %s (score: %.4f)'%(feature_idx_aux, \
                            np.array(attrs_headers)[feature_idx_aux], \
                                        self.attrs_variances[feature_idx_aux])
            report_list.append(current_line_str_aux)
            it+=1

        with open(csv_file_path, file_mode) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')

            csv_writer.writerow(['top_features'])
            csv_writer.writerow(report_list)
            csv_writer.writerow([])

    def store_report(self, csv_file_path, attrs_headers, append=True):
        self.__store_report_to_csv__(csv_file_path, attrs_headers, append)
        self.__plot_report__(attrs_headers, self.dir_to_store_results)

class Fisher_Feature_Retrieval(Super_Feature_Retrieval):

    def __init__(self, **kwargs):
        print('++++ The Fisher Score algorithm has been chosen for feature selection')
        self.noftopfeatures = kwargs['noftopfeatures']
        print('---- Number of top features: %d'%self.noftopfeatures)

    def __compute_fisher_score__(self, input_data, output_data):
        input_data = np.array(input_data).astype(np.float64)
        output_data = np.array(output_data).astype(np.float64)

        nofsamples, nofattrs = np.shape(input_data)
        labels = np.unique(output_data)
        nofclasses = len(labels)
        total_mean = np.mean(input_data, axis=0)
        '''
        [mean(a1), mean(a2), ... , mean(am)]
        '''
        class_num = np.zeros(nofclasses)
        class_mean = np.zeros((nofclasses, nofattrs))
        class_std = np.zeros((nofclasses, nofattrs))
        for i, lab in enumerate(labels):
            idx_list = np.where(output_data == lab)[0]
            # print(idx_list[0])
            class_num[i] = len(idx_list)
            class_mean[i] = np.mean(input_data[idx_list], axis=0)
            class_std[i] = np.std(input_data[idx_list], axis=0)
        '''
        std(c1_a1), std(c1_a2), ..., std(c1_am)
        std(c2_a1), std(c2_a2), ..., std(c2_am)
        std(c3_a1), std(c3_a2), ..., std(c3_am)
        '''
        fisher_scores_list = []

        for current_attr_idx_aux in range(nofattrs):
            Sb_j = 0.0
            Sw_j = 0.0
            for current_class_idx_aux in range(nofclasses):
                Sb_j += class_num[current_class_idx_aux] * \
                    (class_mean[current_class_idx_aux, current_attr_idx_aux] - \
                        total_mean[current_attr_idx_aux])** 2
                Sw_j += class_num[current_class_idx_aux] * \
                    class_std[current_class_idx_aux, current_attr_idx_aux] ** 2
                ratio = Sb_j / Sw_j

            fisher_scores_list.append(ratio)

        fisher_idxs_list = np.flip(np.argsort(fisher_scores_list))

        return fisher_scores_list, fisher_idxs_list

    def execute_feature_retrieval(self, input_data, output_data, plot_data=False):
        self.features_scores, self.ordered_features_idxs = \
            self.__compute_fisher_score__(input_data, output_data)

        top_features_idxs = self.ordered_features_idxs[list(range(0, self.noftopfeatures))]
        transformed_data = input_data[:, top_features_idxs]

        return transformed_data

    def __store_report_to_csv__(self, csv_file_path, attrs_headers, append=True):
        report_list = []
        if (append):
            file_mode = 'a'
        else:
            file_mode = 'w'

        attrs_headers = np.array(attrs_headers)
        it = 0
        for feature_idx_aux in self.ordered_features_idxs:
            current_line_str_aux = '%d: %s (score: %.4f)'%(feature_idx_aux, \
                            np.array(attrs_headers)[feature_idx_aux], \
                                self.features_scores[feature_idx_aux])
            report_list.append(current_line_str_aux)
            it+=1

        with open(csv_file_path, file_mode) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')

            csv_writer.writerow(['top_features'])
            csv_writer.writerow(report_list)
            csv_writer.writerow([])

    # This function plots the report of the scores and the ranking of the
    # attributes after the feature selection algorithm is applied.
    def __plot_report__(self, attrs_headers, dir_to_store_results):
        top_features_idx = np.flip(np.array(self.ordered_features_idxs))
        features_scores = np.array(self.features_scores).astype(np.float64)
        features_scores = features_scores[top_features_idx]

        config = configparser.ConfigParser()
        config.read('./datasets/translations.cfg')
        translated_attrs_headers = super().translate_fields(attrs_headers, config)

        plt.rcParams['font.size'] = '50'
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
        fig, ax = plt.subplots(figsize=(30, 30))
        y_pos = list(range(len(features_scores)))

        my_cmap = plt.get_cmap("Purples")
        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
        ax.barh(y_pos, features_scores, edgecolor='black', align='center', color=my_cmap(rescale(y_pos)))

        # plt.title('Top features scores ranking')
        plt.yticks(y_pos, np.array(translated_attrs_headers)[top_features_idx])
        plt.xscale('log')
        # plt.xlabel('Feature score')
        # plt.ylabel('Feature')

        # It is important to note that x and y are flipped, because we are
        # using barh.
        '''
        for x_coordinate, y_coordinate in enumerate(features_scores):
            text_to_show = '%.2E'%float(y_coordinate)
            ax.text(y_coordinate, x_coordinate - 0.25, text_to_show, color='black', fontweight='bold')
        '''

        plt.xlim([10**(-5), 10**(0)])
        plt.subplots_adjust(left=0.25, bottom=0.05, right=0.60, top=0.97)
        output_filename = '%s/%s'%(dir_to_store_results, 'fisher_report.pdf')
        # plt.yticks(rotation=30)
        plt.savefig(output_filename)
        print('++++ The report of Fisher score top features ranking has been stored at %s'%output_filename)

    def store_report(self, csv_file_path, attrs_headers, append=True):
        self.__store_report_to_csv__(csv_file_path, attrs_headers, append)
        self.__plot_report__(attrs_headers, self.dir_to_store_results)

class MutualInformation_Feature_Retrieval(Super_Feature_Retrieval):

    def __init__(self, **kwargs):
        print('++++ The Mutual Information algorithm has been chosen for feature selection')
        self.noftopfeatures = kwargs['noftopfeatures']
        print('---- Number of top features: %d'%self.noftopfeatures)

    def execute_feature_retrieval(self, input_data, output_data, plot_data=False):
        self.features_scores = mutual_info_classif(input_data, output_data, random_state=5)

        self.ordered_features_idxs = np.flip(np.argsort(self.features_scores))
        top_features_idxs = self.ordered_features_idxs[list(range(0, self.noftopfeatures))]

        transformed_data = input_data[:, top_features_idxs]

        return transformed_data

    def __store_report_to_csv__(self, csv_file_path, attrs_headers, append=True):
        report_list = []
        if (append):
            file_mode = 'a'
        else:
            file_mode = 'w'

        attrs_headers = np.array(attrs_headers)
        it = 0
        for feature_idx_aux in self.ordered_features_idxs:
            current_line_str_aux = '%d: %s (score: %.4f)'%(feature_idx_aux, \
                            np.array(attrs_headers)[feature_idx_aux], \
                                self.features_scores[feature_idx_aux])
            report_list.append(current_line_str_aux)
            it+=1

        with open(csv_file_path, file_mode) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')

            csv_writer.writerow(['top_features'])
            csv_writer.writerow(report_list)
            csv_writer.writerow([])

    # This function plots the report of the scores and the ranking of the
    # attributes after the feature selection algorithm is applied.
    def __plot_report__(self, attrs_headers, dir_to_store_results):
        top_features_idx = np.flip(np.array(self.ordered_features_idxs))
        features_scores = np.array(self.features_scores).astype(np.float64)
        features_scores = features_scores[top_features_idx]

        config = configparser.ConfigParser()
        config.read('./datasets/translations.cfg')
        translated_attrs_headers = super().translate_fields(attrs_headers, config)

        plt.rcParams['font.size'] = '50'
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

        fig, ax = plt.subplots(figsize=(30, 30))
        y_pos = list(range(len(features_scores)))

        my_cmap = plt.get_cmap("Purples")
        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
        ax.tick_params(labelbottom=False,labeltop=True)
        ax.barh(y_pos, features_scores, edgecolor='black', align='center', color=my_cmap(rescale(y_pos)))

        # plt.title('Top features scores ranking')
        plt.yticks(y_pos, np.array(translated_attrs_headers)[top_features_idx])
        plt.xscale('log')
        # plt.xlabel('Feature score')
        # plt.ylabel('Feature')

        # It is important to note that x and y are flipped, because we are
        # using barh.
        '''
        for x_coordinate, y_coordinate in enumerate(features_scores):
            text_to_show = '%.2E'%float(y_coordinate)
            if (y_coordinate<(10**(-4))):
                y_coordinate = 10**(-4)
            ax.text(y_coordinate, x_coordinate - 0.25, text_to_show, color='black', fontweight='bold')
        '''

        # plt.yticks(rotation=30)
        plt.xlim([10**(-4), 10**(0)])
        plt.subplots_adjust(left=0.25, bottom=0.05, right=0.70, top=0.97)
        output_filename = '%s/%s'%(dir_to_store_results, 'mutual_info_report.pdf')
        plt.savefig(output_filename)
        print('++++ The report of Mutual Information score top features ranking has been stored at %s'%output_filename)

    def store_report(self, csv_file_path, attrs_headers, append=True):
        self.__store_report_to_csv__(csv_file_path, attrs_headers, append)
        self.__plot_report__(attrs_headers, self.dir_to_store_results)

class PCA_Feature_Retrieval(Super_Feature_Retrieval):

    def __init__(self, **kwargs):
        print('++++ The PCA algorithm has been chosen for feature selection')
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
