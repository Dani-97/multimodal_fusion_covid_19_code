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
from imaging_features.utils_deep_features import *
from imaging_features.utils_radiomics_features import *
from imaging_features.utils_images import read_image

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
    def __preprocess_dataset__(self, original_input_dataset, headers_file, include_patients_ids=False):
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
        self.original_dataset_df.columns=headers_renaming_dict

        self.df_columns_in_order = self.original_dataset_df.columns[self.attrs_order_idxs]
        if (include_patients_ids):
            self.df_columns_in_order = self.df_columns_in_order.values.tolist()
            code_column = self.original_dataset_df.columns[self.code_idx].values.tolist()
            self.df_columns_in_order = code_column + self.df_columns_in_order
            self.df_columns_in_order = pd.Index(self.df_columns_in_order)

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

    def __get_images_features__(self, chosen_model_str, associations_df, \
                                    input_dir_root_path, masks_dir_root_path, \
                                        selected_approach_obj, device, include_image_name=False):
        features_list = []

        for image_name in associations_df['image_name'].values:
            input_image_full_path = '%s/%s'%(input_dir_root_path, image_name)
            mask_image_full_path = '%s/%s'%(masks_dir_root_path, image_name.replace('.png', '_mask.png'))
            if (os.path.exists(input_image_full_path)):
                input_image = read_image(input_image_full_path)
                mask_image = None
                if (os.path.exists(mask_image_full_path)):
                    mask_image = read_image(mask_image_full_path)
                current_image_features_list_aux = \
                    selected_approach_obj.extract_features(input_image, input_image_full_path, \
                                                                             mask_image, mask_image_full_path)
                if (include_image_name):
                    features_list.append([image_name] + current_image_features_list_aux)
                else:
                    features_list.append(current_image_features_list_aux)
            else:
                print('++++ NOTE: The image %s has been discarded (Not found)...'%image_name)

        headers_list = selected_approach_obj.get_headers()
        if (include_image_name):
            headers_list = ['image_name'] + headers_list
        features_df = pd.DataFrame(features_list)

        return headers_list, features_df

    def __associate_clinical_and_imaging_datasets__(self, associations_df, clinical_data_df, imaging_data_df):
        headers_list = imaging_data_df.columns[1:].values.tolist() + clinical_data_df.columns[1:].values.tolist()
        global_merged_features_list = []
        for patient_id in clinical_data_df['codigo'].values:
            current_image_name_df_rows = associations_df.query("patient_id==%d"%patient_id)['image_name']
            if (len(current_image_name_df_rows)>0):
                current_image_name = current_image_name_df_rows.values[0]
                current_image_features_df_rows = imaging_data_df.query("image_name=='%s'"%current_image_name)
                if (len(current_image_features_df_rows)>0):
                    current_image_features = current_image_features_df_rows.values[0][1:].tolist()
                    current_patient_clinical_data = clinical_data_df.query("codigo==%d"%patient_id).values[0][1:].tolist()
                    current_merged_features_list = current_image_features + current_patient_clinical_data
                    global_merged_features_list.append(current_merged_features_list)
                else:
                    print('++++ The features of image %s have not been obtained.'%current_image_name)
            else:
                print('++++ The patient %d has not any associated image, so it has been discarded.'%patient_id)

        global_merged_features_df = pd.DataFrame(global_merged_features_list)
        global_merged_features_df.columns = headers_list

        return headers_list, global_merged_features_df

    def __build_dataset_with_imaging_data_aux__(self, chosen_approach, headers_file, input_dataset_path, \
                                    masks_dataset_path, input_table_file, associations_file, output_path, device):
        _, clinical_data_df = self.build_dataset(input_table_file, headers_file, include_patients_ids=True)

        universal_factory = UniversalFactory()
        kwargs = {'device': device}
        selected_approach_obj = universal_factory.create_object(globals(), chosen_approach, kwargs)

        associations_df = pd.read_csv(associations_file)
        imaging_data_columns, imaging_features_df = self.__get_images_features__(chosen_approach, associations_df, \
                                    input_dataset_path, masks_dataset_path, selected_approach_obj, device, include_image_name=True)
        imaging_features_df.columns = imaging_data_columns

        merged_features_columns, global_merged_features_df = \
            self.__associate_clinical_and_imaging_datasets__(associations_df, clinical_data_df, imaging_features_df)

        return merged_features_columns, global_merged_features_df

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

    def build_dataset_with_imaging_data(self, chosen_model_str, headers_file, input_dataset_path, \
                                masks_dataset_path, input_table_file, associations_file, output_path, device):
        print('++++ WARNING: the method build_dataset_with_imaging_data has not been implemented for this approach!')

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
                          padding_for_missing_values=-1, discretize=False, \
                              include_patients_ids=False):
        super().__preprocess_dataset__(input_filename, headers_file, \
                                     include_patients_ids=include_patients_ids)
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
                          padding_for_missing_values=-1, discretize=False, \
                              include_patients_ids=False):
        super().__preprocess_dataset__(input_filename, headers_file, \
                                     include_patients_ids=include_patients_ids)
        self.df_unique_values = self.df_unique_values.query("cohorte==1")
        # The cohorte is placed as the last column of the dataset and exitus is
        # removed.
        idxs_list = list(range(1, len(self.df_unique_values.columns)))
        if (include_patients_ids):
            idxs_list = [0] + list(range(2, len(self.df_unique_values.columns)))

        df_columns = self.df_unique_values.columns[idxs_list]
        self.df_unique_values = self.df_unique_values[df_columns]
        self.df_unique_values = self.df_unique_values.rename(columns={'exitus':'output'})

        return self.df_unique_values.columns, self.df_unique_values

class Build_Dataset_Debugging_Only_Deep_Features(Build_Dataset_Only_Hospitalized):

    def __init__(self, **kwargs):
        pass

    def check_dataset_statistics(self):
        pass

    def build_dataset_with_imaging_data(self, chosen_approach, headers_file, input_dataset_path, \
                                masks_dataset_path, input_table_file, associations_file, output_path, device):
        _, clinical_data_df = super().build_dataset(input_table_file, headers_file, include_patients_ids=True)

        universal_factory = UniversalFactory()
        kwargs = {'device': device}
        selected_approach_obj = universal_factory.create_object(globals(), \
                             chosen_model_str + '_Deep_Features_Model', kwargs)

        associations_df = pd.read_csv(associations_file, sep=';')
        headers_list, imaging_features_df = self.__get_images_features__(chosen_model_str, associations_df, \
                        input_dataset_path, masks_dataset_path, selected_approach_obj, device, include_image_name=True)
        imaging_features_df.columns = headers_list

        self.__associate_clinical_and_imaging_datasets__(associations_df, clinical_data_df, imaging_features_df)

        return headers_list, features_df

class Build_Dataset_Debugging_Radiomics_Features(Build_Dataset_Only_Hospitalized):

    def __init__(self, **kwargs):
        pass

    def check_dataset_statistics(self):
        pass

    def build_dataset_with_imaging_data(self, headers_file, input_dataset_path, \
                                masks_dataset_path, input_table_file, associations_file, output_path, device):
        chosen_approach = 'Radiomics_Features'
        headers_list, features_df = \
          super().__build_dataset_with_imaging_data_aux__(chosen_approach, headers_file, input_dataset_path, \
                                    masks_dataset_path, input_table_file, associations_file, output_path, device)

        return headers_list, features_df
