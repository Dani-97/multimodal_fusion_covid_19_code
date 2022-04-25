import csv
from datetime import datetime
import sys
sys.path.append('../')
import numpy as np
import os
import pandas as pd
import pickle
from utils import read_csv_file, read_headers_file
from utils import write_csv_file, convert_si_no_to_int
from deep_features.utils_architectures import *

class UniversalFactory():

    def __init__(self):
        pass

    def create_object(self, namespace, classname, kwargs):
        ClassName = namespace[classname]
        universal_obj = ClassName(**kwargs)
        return universal_obj

class Super_Class_Build_Dataset():

    def __init__(self):
        pass

    def __preprocess_discrete_attrs__(self):
        for column_aux in self.df_discrete_attrs_headers:
            self.df_unique_values[column_aux] = self.df_unique_values[column_aux].fillna(0)
            self.df_unique_values[column_aux] = self.df_unique_values[column_aux].astype(float).astype(int)

    def __preprocess_numerical_attrs__(self):
        for column_aux in self.df_numerical_attrs_headers:
            self.df_unique_values[column_aux] = self.df_unique_values[column_aux].fillna(-1)

            self.df_unique_values[column_aux] = self.df_unique_values[column_aux].astype(str).str. \
                replace(r'([0-9]+),([0-9]+)', r'\1.\2', regex=True). \
                    replace(r'[a-zA-ZÀ-ÿ\u00f1\u00d1]+', r'', regex=True). \
                        replace(r'(,)*', r'', regex=True).str. \
                            replace(r'<([0-9]+)', r'\1', regex=True). \
                                replace(r'>([0-9])', r'\1', regex=True). \
                                    replace(r'[#¡!+]+', r'', regex=True).str.replace(' ', ''). \
                                        replace(r'^$', r'-1', regex=True)

            if (column_aux in self.original_dataset_df.columns[self.int_attrs_idxs]):
                self.df_unique_values[column_aux] = self.df_unique_values[column_aux].astype(float).astype(int)
            else:
                self.df_unique_values[column_aux] = self.df_unique_values[column_aux].astype(float)

    # This function converts all the values of the dataset to numerical.
    def __preprocess_dataset__(self, original_input_dataset, headers_file):
        self.original_dataset_df = pd.read_csv(original_input_dataset, delimiter=';')
        self.code_idx = [0]
        self.cohort_idx = [7]
        self.exitus_idx = [5]
        self.numerical_attrs_idxs = [10, 11, 12, 13, 28, 29, 30, 31, 32, 33, 34, 35, 36]
        # This variable is created if it is necessary to know which are the integer variables.
        self.int_attrs_idxs = [10, 11]
        self.discrete_attrs_idxs = [8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27]
        self.dates_idxs = [1, 3, 4, 6]
        self.other_fields_idxs = [2, 26]
        # This list does not include the unused variables (fields of free text)
        # for the first 2 experiments with this dataset.
        self.attrs_order_idxs = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, \
                      23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 5]

        headers_dict_file = open(headers_file, 'r')
        headers_renaming_dict = eval(headers_dict_file.read())
        self.original_dataset_df = self.original_dataset_df.rename(columns=headers_renaming_dict)

        self.df_columns_in_order = self.original_dataset_df.columns[self.attrs_order_idxs]
        self.df_numerical_attrs_headers = self.original_dataset_df.columns[self.numerical_attrs_idxs]
        self.df_discrete_attrs_headers = self.original_dataset_df.columns[self.discrete_attrs_idxs]

        self.df_unique_values = self.original_dataset_df[self.df_columns_in_order].drop_duplicates()
        self.df_unique_values = self.df_unique_values.replace('Urgencias', 0)
        self.df_unique_values = self.df_unique_values.replace('Hospitalizados', 1)
        self.df_unique_values = self.df_unique_values.replace('Residencias', 2)
        self.df_unique_values = self.df_unique_values.replace('Controles', 3)

        self.df_unique_values = self.df_unique_values.replace('Hombre', 0)
        self.df_unique_values = self.df_unique_values.replace('Mujer', 1)

        self.df_unique_values = self.df_unique_values.replace('<65', 0)
        self.df_unique_values = self.df_unique_values.replace('[65,80]', 1)
        self.df_unique_values = self.df_unique_values.replace('>80', 2)

        self.df_unique_values = self.df_unique_values.replace('No', 0)
        self.df_unique_values = self.df_unique_values.replace('Desconocido', 0)
        self.df_unique_values = self.df_unique_values.replace('Si', 1)

        self.__preprocess_numerical_attrs__()
        self.__preprocess_discrete_attrs__()

        return self.df_unique_values

    # This function retrieves some basic statistics of the dataset.
    def check_dataset_statistics(self):
        print('***** Total number of attributes = %d\n'%len(self.original_dataset_df.columns), '\n')
        print('***** List of numerical attributes (%d):\n\n'%len(self.original_dataset_df.columns[self.numerical_attrs_idxs]), \
                                              self.original_dataset_df.columns[self.numerical_attrs_idxs], '\n')
        print('***** List of numerical integer attributes (%d):\n\n'%len(self.original_dataset_df.columns[self.int_attrs_idxs]), \
                                              self.original_dataset_df.columns[self.int_attrs_idxs], '\n')
        print('***** List of discrete attributes (%d):\n\n'%len(self.original_dataset_df.columns[self.discrete_attrs_idxs]), \
                                              self.original_dataset_df.columns[self.discrete_attrs_idxs], '\n')
        print('***** Assigned code (%d):\n\n'%len(self.original_dataset_df.columns[self.code_idx]), \
                                              self.original_dataset_df.columns[self.code_idx], '\n')
        print('***** Cohort (%d):\n\n'%len(self.original_dataset_df.columns[self.cohort_idx]), \
                                         self.original_dataset_df.columns[self.cohort_idx], '\n')
        print('***** Exitus (%d):\n\n'%len(self.original_dataset_df.columns[self.exitus_idx]), \
                                        self.original_dataset_df.columns[self.exitus_idx], '\n')
        print('***** List of dates attributes (%d):\n\n'%len(self.original_dataset_df.columns[self.dates_idxs]), \
                                              self.original_dataset_df.columns[self.dates_idxs], '\n')
        print('***** List of other attributes (%d):\n\n'%len(self.original_dataset_df.columns[self.other_fields_idxs]), \
                                              self.original_dataset_df.columns[self.other_fields_idxs], '\n')

        print('Total number of rows = ', len(self.original_dataset_df[self.df_columns_in_order]))
        print('Total number of rows without duplicates = ', \
                                           len(self.original_dataset_df[self.df_columns_in_order].drop_duplicates()))

    def build_dataset(self, input_filename, headers_file, \
                            padding_for_missing_values=-1, discretize=False):
        print('++++ WARNING: the method build_dataset has not been implemented for this approach!')

    def build_dataset_with_deep_features(self):
        print('++++ WARNING: the method build_dataset_with_deep_features has not been implemented for this approach!')

    def store_dataset_in_csv_file(self, built_dataset_to_store, output_csv_file_path):
        built_dataset_to_store.to_csv(output_csv_file_path, index=False)
        print('**** The version of the built dataset has been stored at %s'%output_csv_file_path)

class Build_Dataset_Hospitalized_And_Urgencies(Super_Class_Build_Dataset):

    def __init__(self, **kwargs):
        pass

    def check_dataset_statistics(self):
        super().check_dataset_statistics()

        non_hospitalized_patients = self.df_unique_values.query("(output==0)")
        hospitalized_patients = self.df_unique_values.query("(output==1)")
        print('Total number of non-hospitalized patients = %d'%len(non_hospitalized_patients))
        print('Total number of hospitalized patients = %d'%len(hospitalized_patients))

    def build_dataset(self, input_filename, headers_file, \
                            padding_for_missing_values=-1, discretize=False):
        super().__preprocess_dataset__(input_filename, headers_file)
        self.df_unique_values = \
            self.df_unique_values.query("(cohorte==0) or (cohorte==1)")
        # The cohorte is placed as the last column of the dataset and exitus is
        # removed.
        idxs_list = list(range(1, len(self.df_unique_values.columns)-1)) + [0]
        df_columns = self.df_unique_values.columns[idxs_list]
        self.df_unique_values = self.df_unique_values[df_columns]
        self.df_unique_values = self.df_unique_values.rename(columns={'cohorte':'output'})

        return self.df_unique_values.columns, self.df_unique_values

class Build_Dataset_Only_Hospitalized(Super_Class_Build_Dataset):

    def __init__(self, **kwargs):
        pass

    def check_dataset_statistics(self):
        super().check_dataset_statistics()

        death_patients = self.df_unique_values.query("(output==1)")
        survival_patients = self.df_unique_values.query("(output==0)")
        print('Total number of deaths = %d'%len(death_patients))
        print('Total number of survivals = %d'%len(survival_patients))

    def build_dataset(self, input_filename, headers_file, \
                            padding_for_missing_values=-1, discretize=False):
        super().__preprocess_dataset__(input_filename, headers_file)
        self.df_unique_values = self.df_unique_values.query("cohorte==1")
        # The cohorte is placed as the last column of the dataset and exitus is
        # removed.
        idxs_list = list(range(1, len(self.df_unique_values.columns)))
        df_columns = self.df_unique_values.columns[idxs_list]
        self.df_unique_values = self.df_unique_values[df_columns]
        self.df_unique_values = self.df_unique_values.rename(columns={'exitus':'output'})

        return self.df_unique_values.columns, self.df_unique_values
