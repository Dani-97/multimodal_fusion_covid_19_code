import csv
from datetime import datetime
import sys
sys.path.append('../')
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import read_csv_file, read_headers_file
from utils import write_csv_file, convert_si_no_to_int
from feature_extraction.utils_deep_features import *
from feature_extraction.utils_radiomics_features import *
from feature_extraction.utils_images import read_image

class UniversalFactory():

    def __init__(self):
        pass

    def create_object(self, namespace, classname, kwargs):
        ClassName = namespace[classname]
        universal_obj = ClassName(**kwargs)
        return universal_obj

class Super_Class_Build_Dataset_Without_Patients_Ids():

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
    def __preprocess_dataset__(self, original_input_dataset, headers_file, include_rx_code=False):
        try:
            self.original_dataset_df = pd.read_csv(original_input_dataset, delimiter=';')
        except pd.errors.ParserError:
            self.original_dataset_df = pd.read_csv(original_input_dataset, delimiter=',')
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
        self.original_dataset_df.columns=headers_renaming_dict.values()

        self.df_columns_in_order = self.original_dataset_df.columns[self.attrs_order_idxs]
        if (include_rx_code):
            self.df_columns_in_order = self.df_columns_in_order.values.tolist()
            code_column = self.original_dataset_df.columns[self.code_idx].values.tolist()
            self.df_columns_in_order = code_column + self.df_columns_in_order
            self.df_columns_in_order = pd.Index(self.df_columns_in_order)

        self.df_numerical_attrs_headers = self.original_dataset_df.columns[self.numerical_attrs_idxs]
        self.df_discrete_attrs_headers = self.original_dataset_df.columns[self.discrete_attrs_idxs]

        columns_without_code = self.df_columns_in_order[1:]

        self.df_unique_values = self.original_dataset_df[self.df_columns_in_order].drop_duplicates(subset=columns_without_code)
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

    # This function converts all the values of the dataset to numerical.
    def __preprocess_dataset_with_patients_ids__(self, original_input_dataset, \
                          headers_file, include_rx_code=True, include_patient_id=True):
        try:
            self.original_dataset_df = pd.read_csv(original_input_dataset, delimiter=';')
        except pd.errors.ParserError:
            self.original_dataset_df = pd.read_csv(original_input_dataset, delimiter=',')
        self.patient_id_idx = [0]
        self.rxs_codes_list = self.original_dataset_df['Código asignado'].tolist()
        self.patient_text_reports = self.original_dataset_df['Hallazgos']
        self.code_idx = [1]
        self.cohort_idx = [7]
        self.exitus_idx = [5]

        self.numerical_attrs_idxs = [10, 11, 12, 13, 28, 29, 30, 31, 32, 33, 34, 35, 36]
        # This variable is created if it is necessary to know which are the integer variables.
        self.int_attrs_idxs = [10, 11]
        self.discrete_attrs_idxs = [8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27]
        self.dates_idxs = [2, 4, 6]
        self.other_fields_idxs = [2, 26]

        # This list does not include the unused variables (fields of free text)
        # for the first 2 experiments with this dataset.
        self.attrs_order_idxs = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, \
                      23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 2, 5]

        headers_dict_file = open(headers_file, 'r')
        headers_renaming_dict = eval(headers_dict_file.read())
        self.original_dataset_df.columns=headers_renaming_dict.values()

        self.df_columns_in_order = self.original_dataset_df.columns[self.attrs_order_idxs]
        if (include_rx_code):
            self.df_columns_in_order = self.df_columns_in_order.values.tolist()
            code_column = self.original_dataset_df.columns[self.code_idx].values.tolist()
            self.df_columns_in_order = code_column + self.df_columns_in_order
            self.df_columns_in_order = pd.Index(self.df_columns_in_order)

        if (include_patient_id):
            self.df_columns_in_order = self.df_columns_in_order.values.tolist()
            patient_id_column = self.original_dataset_df.columns[self.patient_id_idx].values.tolist()
            self.df_columns_in_order = patient_id_column + self.df_columns_in_order
            self.df_columns_in_order = pd.Index(self.df_columns_in_order)

        self.df_numerical_attrs_headers = self.original_dataset_df.columns[self.numerical_attrs_idxs]
        self.df_discrete_attrs_headers = self.original_dataset_df.columns[self.discrete_attrs_idxs]

        # columns_without_code = self.df_columns_in_order[1:]

        self.df_unique_values = self.original_dataset_df[self.df_columns_in_order]
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

    def __get_images_features__(self, chosen_model_str, clinical_data_df, \
                                    input_dir_root_path, masks_dir_root_path, \
                                        selected_approach_obj, device, layer, include_image_name=False):
        features_list = []
        unique_patients_ids_list = np.unique(clinical_data_df['patient_id'])

        total_nofpatients = len(unique_patients_ids_list)
        for current_patient_id_aux in unique_patients_ids_list:
            current_patient_fst_rx_code = \
                clinical_data_df.query("patient_id==%d"%current_patient_id_aux).sort_values('fecha_seguimiento').iloc[0]['codigo']

            image_to_read_name = 'patient_%d_rxcode_%d.jpg'%(current_patient_id_aux, current_patient_fst_rx_code)
            input_image_full_path = '%s/%s'%(input_dir_root_path, image_to_read_name)
            mask_image_full_path = '%s/%s'%(masks_dir_root_path, image_to_read_name.replace('.png', '_mask.png'))
            if (os.path.exists(input_image_full_path)):
                input_image = read_image(input_image_full_path)
                mask_image = None
                if (os.path.exists(mask_image_full_path)):
                    mask_image = read_image(mask_image_full_path)
                current_image_features_list_aux = \
                    selected_approach_obj.extract_features(input_image, input_image_full_path, \
                                                                             mask_image, mask_image_full_path, layer)
                if (include_image_name):
                    features_list.append([image_to_read_name] + current_image_features_list_aux)
                else:
                    features_list.append(current_image_features_list_aux)
            else:
                print('++++ NOTE: The image %s has been discarded (Not found)...'%image_to_read_name)

        features_df = pd.DataFrame(features_list)

        headers_list = selected_approach_obj.get_headers()
        if (include_image_name):
            headers_list = ['image_name'] + headers_list

        return headers_list, features_df

    def __get_text_reports_features__(self, selected_approach_obj, device):
        sentences_list = self.patient_text_reports.values.tolist()
        sentences_embeddings = selected_approach_obj.extract_deep_features(sentences_list)

        headers_list = ['rx_code']
        sentenc_embeds_df = pd.DataFrame([])
        if (len(sentences_embeddings)!=0):
            for idx in range(0, sentences_embeddings.shape[1]):
                headers_list.append('sentenc_embeds_%d'%idx)

            self.rxs_codes_list = np.expand_dims(np.array(self.rxs_codes_list), axis=1)
            sentenc_embeds_np = np.concatenate([self.rxs_codes_list, sentences_embeddings], axis=1)
            sentenc_embeds_df = pd.DataFrame(sentenc_embeds_np)
            sentenc_embeds_df.columns = headers_list
            sentenc_embeds_df['rx_code'] = sentenc_embeds_df['rx_code'].astype(int)

        return headers_list, sentenc_embeds_df

    def __associate_clinical_and_imaging_datasets__(self, associations_df, clinical_data_df, imaging_data_df, text_reports_features_df):
        # Drop the unnecessary columns from the clinical data.
        clinical_data_df = clinical_data_df.drop(columns=['fecha_seguimiento'])

        sentenc_embeds_headers_list = text_reports_features_df.columns[1:].tolist()

        headers_list = imaging_data_df.columns[1:].values.tolist() + sentenc_embeds_headers_list + clinical_data_df.columns[3:].values.tolist()
        global_merged_features_list = []
        full_images_names_list = imaging_data_df['image_name']

        for current_rx_code_aux in clinical_data_df['codigo'].values:
            current_rx_info = clinical_data_df.query("codigo==%d"%current_rx_code_aux)
            if (len(current_rx_info)>0):
                current_patient_id_aux = current_rx_info.iloc[0]['patient_id']
                current_image_name = 'patient_%d_rxcode_%d.jpg'%(current_patient_id_aux, current_rx_code_aux)
                current_rx_code = current_image_name.split('_')[3].split('.')[0]
                if (current_image_name in full_images_names_list.tolist()):
                    current_image_features_df_rows = imaging_data_df.query("image_name=='%s'"%current_image_name)
                    current_image_features = current_image_features_df_rows.values[0][1:].tolist()
                    current_patient_clinical_data = current_rx_info.values[0][3:]
                    text_reports_features_list = []
                    if (not text_reports_features_df.empty):
                        text_reports_features_list = text_reports_features_df.query('rx_code==%s'%current_rx_code).values[0, 1:].tolist()
                    current_merged_features_list = current_image_features + text_reports_features_list + current_patient_clinical_data.tolist()
                    global_merged_features_list.append(current_merged_features_list)
                else:
                    print('++++ The features of image %s have not been obtained.'%current_image_name)
            else:
                print('++++ The code %d has not any associated image, so it has been discarded.'%current_rx_code_aux)

        global_merged_features_df = pd.DataFrame(global_merged_features_list)
        global_merged_features_df.columns = headers_list

        return headers_list, global_merged_features_df

    def __build_dataset_with_imaging_data_aux__(self, chosen_approach, headers_file, input_dataset_path, \
                                    masks_dataset_path, input_table_file, associations_file, output_path, \
                                        device, layer, text_reports_embeds_method):
        _, clinical_data_df = self.build_dataset_with_patients_ids(input_table_file, headers_file)
        self.clinical_data_df = clinical_data_df

        universal_factory = UniversalFactory()
        kwargs = {'device': device}
        selected_approach_obj = universal_factory.create_object(globals(), chosen_approach, kwargs)

        sentences_embeds_retrieval_obj = \
            universal_factory.create_object(globals(), text_reports_embeds_method + '_Deep_Features_Model', kwargs)

        text_reports_data_columns, text_reports_features_df = \
                            self.__get_text_reports_features__(sentences_embeds_retrieval_obj, device)

        associations_df = pd.read_csv(associations_file)
        imaging_data_columns, imaging_features_df = self.__get_images_features__(chosen_approach, clinical_data_df, \
                                    input_dataset_path, masks_dataset_path, selected_approach_obj, device, layer, include_image_name=True)
        imaging_features_df.columns = imaging_data_columns
        # Removing the features whose values are all 0.
        idxs_to_keep = np.where((imaging_features_df==0).sum()<len(imaging_features_df))[0]
        imaging_features_df = imaging_features_df[imaging_features_df.columns[idxs_to_keep]]

        merged_features_columns, global_merged_features_df = \
            self.__associate_clinical_and_imaging_datasets__(associations_df, clinical_data_df, imaging_features_df, text_reports_features_df)

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

        columns_without_code = self.df_columns_in_order[1:]
        print('Total number of rows = ', len(self.original_dataset_df[self.df_columns_in_order]))
        print('Total number of rows without duplicates = ', \
                                           len(self.original_dataset_df[self.df_columns_in_order].drop_duplicates(subset=columns_without_code)))

    def build_dataset(self, input_filename, headers_file, \
                            padding_for_missing_values=-1, discretize=False):
        print('++++ WARNING: the method build_dataset has not been implemented for this approach!')

    def build_dataset_with_patients_ids(self, input_filename, headers_file, \
                            padding_for_missing_values=-1, discretize=False):
        print('++++ WARNING: the method build_dataset_with_patients_ids has not been implemented for this approach!')

    def build_dataset_with_imaging_data(self, chosen_model_str, headers_file, input_dataset_path, \
                                masks_dataset_path, input_table_file, associations_file, output_path, \
                                    device, layer, text_reports_embeds_method):
        print('++++ WARNING: the method build_dataset_with_imaging_data has not been implemented for this approach!')

    def store_dataset_in_csv_file(self, built_dataset_to_store, output_csv_file_path):
        built_dataset_to_store.to_csv(output_csv_file_path, index=False)
        print('**** The version of the built dataset has been stored at %s'%output_csv_file_path)

class Build_Dataset_Hospitalized_And_Urgencies(Super_Class_Build_Dataset_Without_Patients_Ids):

    def __init__(self, **kwargs):
        pass

    def check_dataset_statistics(self):
        super().check_dataset_statistics()

        non_hospitalized_patients = self.df_unique_values.query("(output==0)")
        hospitalized_patients = self.df_unique_values.query("(output==1)")
        print('Total number of non-hospitalized patients = %d'%len(non_hospitalized_patients))
        print('Total number of hospitalized patients = %d'%len(hospitalized_patients))

    def build_dataset_with_patients_ids(self, input_filename, headers_file, \
                          padding_for_missing_values=-1, discretize=False):
        super().__preprocess_dataset_with_patients_ids__(input_filename, headers_file)
        self.df_unique_values = \
            self.df_unique_values.query("(cohorte==0) or (cohorte==1)")
        # The cohorte is placed as the last column of the dataset and exitus is
        # removed
        idxs_list = [0, 1] + list(range(3, len(self.df_unique_values.columns)-1)) + [2]
        df_columns = self.df_unique_values.columns[idxs_list]
        self.df_unique_values = self.df_unique_values[df_columns]
        self.df_unique_values = self.df_unique_values.rename(columns={'cohorte':'output'})

        return self.df_unique_values.columns, self.df_unique_values

    def build_dataset(self, input_filename, headers_file, \
                          padding_for_missing_values=-1, discretize=False, \
                              include_rx_code=False):
        super().__preprocess_dataset__(input_filename, headers_file, \
                                     include_rx_code=include_rx_code)
        self.df_unique_values = \
            self.df_unique_values.query("(cohorte==0) or (cohorte==1)")
        # The cohorte is placed as the last column of the dataset and exitus is
        # removed.
        idxs_list = list(range(1, len(self.df_unique_values.columns)-1)) + [0]
        df_columns = self.df_unique_values.columns[idxs_list]
        self.df_unique_values = self.df_unique_values[df_columns]
        self.df_unique_values = self.df_unique_values.rename(columns={'cohorte':'output'})

        return self.df_unique_values.columns, self.df_unique_values

class Build_Dataset_Only_Hospitalized(Super_Class_Build_Dataset_Without_Patients_Ids):

    def __init__(self, **kwargs):
        pass

    def check_dataset_statistics(self):
        super().check_dataset_statistics()

        death_patients = self.df_unique_values.query("(output==1)")
        survival_patients = self.df_unique_values.query("(output==0)")
        print('Total number of deaths = %d'%len(death_patients))
        print('Total number of survivals = %d'%len(survival_patients))

    def build_dataset_with_patients_ids(self, input_filename, headers_file, \
                          padding_for_missing_values=-1, discretize=False):
        super().__preprocess_dataset_with_patients_ids__(input_filename, headers_file)
        self.df_unique_values = self.df_unique_values.query("cohorte==1")
        # The cohorte is placed as the last column of the dataset and exitus is
        # removed
        idxs_list = [0] + list(range(1, len(self.df_unique_values.columns)))

        df_columns = self.df_unique_values.columns[idxs_list]
        self.df_unique_values = self.df_unique_values[df_columns]
        self.df_unique_values = self.df_unique_values.rename(columns={'exitus':'output'})

        return self.df_unique_values.columns, self.df_unique_values

    def build_dataset(self, input_filename, headers_file, \
                          padding_for_missing_values=-1, discretize=False, \
                              include_rx_code=False):
        super().__preprocess_dataset__(input_filename, headers_file, \
                                     include_rx_code=include_rx_code)
        self.df_unique_values = self.df_unique_values.query("cohorte==1")
        # The cohorte is placed as the last column of the dataset and exitus is
        # removed.
        idxs_list = list(range(1, len(self.df_unique_values.columns)))
        if (include_rx_code):
            idxs_list = [0] + list(range(2, len(self.df_unique_values.columns)))

        df_columns = self.df_unique_values.columns[idxs_list]
        self.df_unique_values = self.df_unique_values[df_columns]
        self.df_unique_values = self.df_unique_values.rename(columns={'exitus':'output'})

        return self.df_unique_values.columns, self.df_unique_values

class Build_Dataset_DPN_Model_Only_Hospitalized(Build_Dataset_Only_Hospitalized):

    def __init__(self, **kwargs):
        pass

    def check_dataset_statistics(self):
        pass

    def build_dataset_with_imaging_data(self, headers_file, input_dataset_path, \
                                masks_dataset_path, input_table_file, associations_file, \
                                    output_path, text_reports_embeds_method_str, device, layer):
        chosen_approach = 'DPN_TIMM_Deep_Features_Model'
        headers_list, features_df = \
          super().__build_dataset_with_imaging_data_aux__(chosen_approach, headers_file, input_dataset_path, \
                                    masks_dataset_path, input_table_file, associations_file, output_path, \
                                        device, layer, text_reports_embeds_method_str)

        return headers_list, features_df

class Build_Dataset_DPN_Model_Hospitalized_And_Urgencies(Build_Dataset_Hospitalized_And_Urgencies):

    def __init__(self, **kwargs):
        pass

    def check_dataset_statistics(self):
        pass

    def build_dataset_with_imaging_data(self, headers_file, input_dataset_path, \
                                masks_dataset_path, input_table_file, associations_file, \
                                    output_path, text_reports_embeds_method_str, device, layer):
        chosen_approach = 'DPN_TIMM_Deep_Features_Model'
        headers_list, features_df = \
          super().__build_dataset_with_imaging_data_aux__(chosen_approach, headers_file, input_dataset_path, \
                                    masks_dataset_path, input_table_file, associations_file, output_path, \
                                        device, layer, text_reports_embeds_method_str)

        return headers_list, features_df

class Build_Dataset_DeiT_Model_Only_Hospitalized(Build_Dataset_Only_Hospitalized):

    def __init__(self, **kwargs):
        pass

    def check_dataset_statistics(self):
        pass

    def build_dataset_with_imaging_data(self, headers_file, input_dataset_path, \
                                masks_dataset_path, input_table_file, associations_file, \
                                    output_path, text_reports_embeds_method_str, device, layer):
        chosen_approach = 'DeiT_Transformer_Model'
        headers_list, features_df = \
          super().__build_dataset_with_imaging_data_aux__(chosen_approach, headers_file, input_dataset_path, \
                                    masks_dataset_path, input_table_file, associations_file, output_path, \
                                        device, layer, text_reports_embeds_method_str)

        return headers_list, features_df

class Build_Dataset_DeiT_Model_Hospitalized_And_Urgencies(Build_Dataset_Hospitalized_And_Urgencies):

    def __init__(self, **kwargs):
        pass

    def check_dataset_statistics(self):
        pass

    def build_dataset_with_imaging_data(self, headers_file, input_dataset_path, \
                                masks_dataset_path, input_table_file, associations_file, \
                                    output_path, text_reports_embeds_method_str, device, layer):
        chosen_approach = 'DeiT_Transformer_Model'
        headers_list, features_df = \
          super().__build_dataset_with_imaging_data_aux__(chosen_approach, headers_file, input_dataset_path, \
                                    masks_dataset_path, input_table_file, associations_file, output_path, \
                                        device, layer, text_reports_embeds_method_str)

        return headers_list, features_df

class Build_Dataset_Mixed_Vision_Transformer_Only_Hospitalized(Build_Dataset_Only_Hospitalized):

    def __init__(self, **kwargs):
        pass

    def check_dataset_statistics(self):
        pass

    def build_dataset_with_imaging_data(self, headers_file, input_dataset_path, \
                                masks_dataset_path, input_table_file, associations_file, \
                                    output_path, text_reports_embeds_method_str, device, layer):
        chosen_approach = 'Mixed_Vision_Transformer_Model'
        headers_list, features_df = \
          super().__build_dataset_with_imaging_data_aux__(chosen_approach, headers_file, input_dataset_path, \
                                    masks_dataset_path, input_table_file, associations_file, output_path, \
                                        device, layer, text_reports_embeds_method_str)

        return headers_list, features_df

class Build_Dataset_Mixed_Vision_Transformer_Hospitalized_And_Urgencies(Build_Dataset_Hospitalized_And_Urgencies):

    def __init__(self, **kwargs):
        pass

    def check_dataset_statistics(self):
        pass

    def build_dataset_with_imaging_data(self, headers_file, input_dataset_path, \
                                masks_dataset_path, input_table_file, associations_file, \
                                    output_path, text_reports_embeds_method_str, device, layer):
        chosen_approach = 'Mixed_Vision_Transformer_Model'
        headers_list, features_df = \
          super().__build_dataset_with_imaging_data_aux__(chosen_approach, headers_file, input_dataset_path, \
                                    masks_dataset_path, input_table_file, associations_file, output_path, \
                                        device, layer, text_reports_embeds_method_str)

        return headers_list, features_df

class Build_Dataset_VGG_16_Model_Only_Hospitalized(Build_Dataset_Only_Hospitalized):

    def __init__(self, **kwargs):
        pass

    def check_dataset_statistics(self):
        pass

    def build_dataset_with_imaging_data(self, headers_file, input_dataset_path, \
                                masks_dataset_path, input_table_file, associations_file, \
                                    output_path, text_reports_embeds_method_str, device, layer):
        chosen_approach = 'VGG_16_Deep_Features_Model'
        headers_list, features_df = \
          super().__build_dataset_with_imaging_data_aux__(chosen_approach, headers_file, input_dataset_path, \
                                    masks_dataset_path, input_table_file, associations_file, output_path, \
                                        device, layer, text_reports_embeds_method_str)

        return headers_list, features_df

class Build_Dataset_VGG_16_Model_Hospitalized_And_Urgencies(Build_Dataset_Hospitalized_And_Urgencies):

    def __init__(self, **kwargs):
        pass

    def check_dataset_statistics(self):
        pass

    def build_dataset_with_imaging_data(self, headers_file, input_dataset_path, \
                                masks_dataset_path, input_table_file, associations_file, \
                                    output_path, text_reports_embeds_method_str, device, layer):
        chosen_approach = 'VGG_16_Deep_Features_Model'
        headers_list, features_df = \
          super().__build_dataset_with_imaging_data_aux__(chosen_approach, headers_file, input_dataset_path, \
                                    masks_dataset_path, input_table_file, associations_file, output_path, \
                                        device, layer, text_reports_embeds_method_str)

        return headers_list, features_df

# class Build_Dataset_Text_Embeddings_Only_Hospitalized(Build_Dataset_Only_Hospitalized):
#
#     def __init__(self, **kwargs):
#         pass
#
#     def check_dataset_statistics(self):
#         pass
#
#     def build_dataset_with_imaging_data(self, headers_file, input_dataset_path, \
#                                 masks_dataset_path, input_table_file, associations_file, \
#                                     output_path, text_reports_embeds_method_str, device, layer):
#         chosen_approach = 'No_Deep_Features_Model'
#         headers_list, features_df = \
#           super().__build_dataset_with_imaging_data_aux__(chosen_approach, headers_file, input_dataset_path, \
#                                     masks_dataset_path, input_table_file, associations_file, output_path, \
#                                         device, layer, text_reports_embeds_method_str)
#
#         return headers_list, features_df
#
# class Build_Dataset_Text_Embeddings_Hospitalized_And_Urgencies(Build_Dataset_Hospitalized_And_Urgencies):
#
#     def __init__(self, **kwargs):
#         pass
#
#     def check_dataset_statistics(self):
#         pass
#
#     def build_dataset_with_imaging_data(self, headers_file, input_dataset_path, \
#                                 masks_dataset_path, input_table_file, associations_file, \
#                                     output_path, text_reports_embeds_method_str, device, layer):
#         chosen_approach = 'No_Deep_Features_Model'
#         headers_list, features_df = \
#           super().__build_dataset_with_imaging_data_aux__(chosen_approach, headers_file, input_dataset_path, \
#                                     masks_dataset_path, input_table_file, associations_file, output_path, \
#                                         device, layer, text_reports_embeds_method_str)
#
#         return headers_list, features_df
