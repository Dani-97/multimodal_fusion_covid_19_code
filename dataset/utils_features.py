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
import matplotlib.ticker as plticker
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import os
import pandas as pd
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

    def get_ordered_features_idxs(self):
        if (self.noftopfeatures=='all'):
            top_features_idxs = self.ordered_features_idxs
        else:
            top_features_idxs = self.ordered_features_idxs[list(range(0, int(self.noftopfeatures)))]

        return top_features_idxs

    def set_dir_to_store_results(self, dir_to_store_results):
        self.dir_to_store_results = dir_to_store_results

    def set_attributes_types(self, attrs_types):
        self.attrs_types = attrs_types

    def __get_global_idxs_clinical_features__(self, attrs_names_list):
        global_idxs_list = []
        idx = 0
        for current_attr_name_aux in attrs_names_list:
            if (current_attr_name_aux.find('feature_')==-1):
                global_idxs_list.append(idx)

            idx+=1

        return global_idxs_list

    def __associate_global_idxs_to_headers__(self, attrs_headers_to_disp, clinical_features_global_idxs):
        attrs_headers_to_disp_global_idxs_added = []

        idx = 0
        for current_attr_header_aux in attrs_headers_to_disp:
            current_item_to_add = '%d. '%(clinical_features_global_idxs[idx]+1) + current_attr_header_aux
            attrs_headers_to_disp_global_idxs_added.append(current_item_to_add)
            idx+=1

        return attrs_headers_to_disp_global_idxs_added

    # PRECONDITION: this function assumess that scores_list and attrs_names_list
    # are in the same order.
    def __remove_unnecessary_features_to_plot__(self, scores_list, attrs_names_list):
        # This function allows to display the attributes without score as
        # empty strings.
        filtered_features_list = []
        filtered_scores_list = []

        it = 0
        for current_attr_name_aux in attrs_names_list:
            if ((len(attrs_names_list[it])>0) and (attrs_names_list[it].find('feature_')==-1)):
                filtered_features_list.append(current_attr_name_aux)
                filtered_scores_list.append(scores_list[it])

            it+=1

        return filtered_features_list, filtered_scores_list

    def get_ordered_top_features(self, attrs_headers, noftopfeatures):
        if (noftopfeatures=='all'):
            ordered_top_features = np.array(attrs_headers)[self.ordered_features_idxs]
        else:
            ordered_top_features = np.array(attrs_headers)[self.ordered_features_idxs[:int(noftopfeatures)]]

        return ordered_top_features

    def get_ordered_categorical_and_continuous_top_features(self, attrs_headers, noftopfeatures, csv_path_with_attrs_types):

        def get_attrs_types_array(attrs_types_df, ordered_top_features):
            types_list = []
            for current_top_feature_aux in ordered_top_features:
                current_type_aux = attrs_types_df.query("name=='%s'"%current_top_feature_aux)['type'].values.tolist()
                if (len(current_type_aux)>0):
                    types_list+=current_type_aux
                else:
                    types_list+=['continuous']

            return types_list

        ordered_top_features = self.get_ordered_top_features(attrs_headers, noftopfeatures)
        attrs_types_df = pd.read_csv(csv_path_with_attrs_types, delimiter=',')

        types_list = get_attrs_types_array(attrs_types_df, ordered_top_features)
        categorical_attrs_idxs = np.where(np.array(types_list)=='categorical')[0]
        continuous_attrs_idxs = np.where(np.array(types_list)=='continuous')[0]

        return categorical_attrs_idxs, continuous_attrs_idxs
    
    def translate_headers(self, attrs_headers, translations_csv_path):
        translated_attrs_headers = attrs_headers
        
        # Only of the CSV with the translations is specified, the 
        # translations will be actually performed. 
        if (translations_csv_path!=None):
            translated_attrs_headers = []
            for current_attr_header_aux in attrs_headers:
                translations_df = pd.read_csv(translations_csv_path)
                current_translated_attr_header = \
                    translations_df.query("original_name=='%s'"%current_attr_header_aux)['translated_name'].values
                if (len(current_translated_attr_header)>0):
                    translated_attrs_headers.append(current_translated_attr_header[0])
                else:
                    translated_attrs_headers.append(current_attr_header_aux)

        return translated_attrs_headers
        
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
    def __plot_report__(self, attrs_headers, dir_to_store_results, csv_file_path):
        top_features_idx = np.array(self.ordered_features_idxs)
        features_scores = np.array(self.features_scores).astype(np.float64)
        # In this line, we make sure that the features are ordered by importance
        # (and, therefore, they go from the highest to the lowest).
        features_scores = features_scores[top_features_idx]

        # In this line, we make sure that all the features are ordered properly given their importance.
        translated_attrs_headers = np.array(attrs_headers)[top_features_idx]

        # Here, we obtain only the indexes of the clinical features. Moreover, those features are ordered
        # by importance, both globally and specifically for only this set of clinical features.
        clinical_features_global_idxs = self.__get_global_idxs_clinical_features__(translated_attrs_headers)
        attrs_headers_to_disp = translated_attrs_headers[clinical_features_global_idxs]
        features_scores_to_disp = features_scores[clinical_features_global_idxs]

        attrs_headers_to_disp = self.__associate_global_idxs_to_headers__(attrs_headers_to_disp, clinical_features_global_idxs)

        plt.rcParams['font.size'] = '50'
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

        _, ax = plt.subplots(figsize=(30, 30))
        y_pos = list(range(len(features_scores_to_disp)))

        my_cmap = plt.get_cmap("Purples")
        rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
        ax.tick_params(labelbottom=False,labeltop=True)
        ax.barh(y_pos, features_scores_to_disp, edgecolor='black', align='center', color='lightblue')
        ax.invert_yaxis()

        plt.yticks(y_pos, np.array(attrs_headers_to_disp))
        plt.xscale('log')

        xlim_left, xlim_right = 10**(-5), 10**(-1)
        plt.xlim([xlim_left, xlim_right])
        plt.subplots_adjust(left=0.45, bottom=0.05, right=0.90, top=0.97)

        output_pdf_filename = csv_file_path.replace('.csv', '') + '%s.pdf'%self.output_pdf_suffix
        output_filename = '%s/%s'%(dir_to_store_results, output_pdf_filename)
        plt.savefig(output_filename)
        print('++++ The report of %s score top features ranking has been stored at %s'%(self.approach_name, output_filename))

    def store_report(self, csv_file_path, attrs_headers, append=True):
        self.__store_report_to_csv__(csv_file_path, attrs_headers, append)
        self.__plot_report__(attrs_headers, self.dir_to_store_results, csv_file_path)

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

class VarianceThreshold_Feature_Retrieval(Super_Feature_Retrieval):

    def __init__(self, **kwargs):
        self.approach_name = 'Variance Threshold'
        self.output_pdf_suffix = '_var_threshold'
        print('++++ The %s algorithm has been chosen for feature selection'%self.approach_name)
        self.noftopfeatures = kwargs['noftopfeatures']
        print('---- Number of top features: %s'%self.noftopfeatures)

    def execute_feature_retrieval(self, input_data, output_data, plot_data=False):
        self.feature_selector = VarianceThreshold()
        self.feature_selector.fit(input_data)

        self.features_scores = self.feature_selector.variances_
        self.ordered_features_idxs = np.flip(np.argsort(self.features_scores))
        top_features_idxs = super().get_ordered_features_idxs()

        transformed_data = input_data[:, top_features_idxs]

        return transformed_data

class Fisher_Feature_Retrieval(Super_Feature_Retrieval):

    def __init__(self, **kwargs):
        self.approach_name = 'Fisher Score'
        self.output_pdf_suffix = '_fisher_score'
        print('++++ The %s algorithm has been chosen for feature selection'%self.approach_name)
        self.noftopfeatures = kwargs['noftopfeatures']
        print('---- Number of top features: %s'%self.noftopfeatures)

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
        self.features_scores, self.ordered_features_idxs = self.__compute_fisher_score__(input_data, output_data)

        top_features_idxs = super().get_ordered_features_idxs()
        transformed_data = input_data[:, top_features_idxs]

        return transformed_data

class MutualInformation_Feature_Retrieval(Super_Feature_Retrieval):

    def __init__(self, **kwargs):
        self.approach_name = 'Mutual Information'
        self.output_pdf_suffix = '_mutual_information'
        print('++++ The %s algorithm has been chosen for feature selection'%self.approach_name)
        self.noftopfeatures = kwargs['noftopfeatures']
        print('---- Number of top features: %s'%self.noftopfeatures)

    def execute_feature_retrieval(self, input_data, output_data, plot_data=False):
        self.features_scores = mutual_info_classif(input_data, output_data, random_state=5)

        self.ordered_features_idxs = np.flip(np.argsort(self.features_scores))
        top_features_idxs = super().get_ordered_features_idxs()

        transformed_data = input_data[:, top_features_idxs]

        return transformed_data

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
