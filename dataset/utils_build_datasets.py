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
from feature_extraction.utils_deep_features import LSTM_ProgPred_Model
from feature_extraction.utils_deep_features import VGG_16_Deep_Features_Model
from feature_extraction.utils_images import read_image

class UniversalFactory():

    def __init__(self):
        pass

    def create_object(self, namespace, classname, kwargs):
        ClassName = namespace[classname]
        universal_obj = ClassName(**kwargs)
        return universal_obj

# This class must be inherited by those classes that create datasets
# using only one image timepoint.
class Single_Timepoint_Build_Dataset():

    def __init__(self, **kwargs):
        pass

    def __get_corresponding_timepoints__(self, features_df):
        print('++++ Retrieving only the first timepoints of each patient...')

        patients_ids_list = np.unique(features_df['patient_id'])
        output_values = []
        headers_list = features_df.columns
        it = 1
        total_length = len(patients_ids_list)
        for current_patient_id_aux in patients_ids_list:
            print('++++ Processing patient %s [%d/%d]...'%(current_patient_id_aux, it, total_length), end='\r')
            current_patient_timepoints_df = features_df.query("patient_id==%d and distancia_temporal==0"%current_patient_id_aux)
            if (len(current_patient_timepoints_df)>0):
                output_values.append(current_patient_timepoints_df.values[0].tolist())
            it+=1

        output_values = pd.DataFrame(output_values)
        output_values.columns = headers_list
        output_values = output_values.drop(['primera_fecha_seguimiento', 'distancia_temporal'], axis=1)
        
        print('++++ OK: All the processing has been done successfully!')

        return output_values

    def __associate_clinical_and_imaging_datasets__(self, clinical_data_df, imaging_data_df):
        print('++++ Associating clinical data and imaging features...')

        headers_list = imaging_data_df.columns[1:].values.tolist() + clinical_data_df.columns.values.tolist()
        global_merged_features_df = pd.DataFrame([])

        it = 1
        rxcodes_list = clinical_data_df['codigo']
        total_length = len(rxcodes_list)
        for current_image_rxcode in rxcodes_list:
            print('++++ Processing data with RX code %d [%d/%d]'%(current_image_rxcode, it, total_length), end='\r')
            current_image_name = 'rxcode_%d.jpg'%current_image_rxcode
            current_rxcode_info_df = clinical_data_df.query("codigo==%d"%current_image_rxcode)
            current_image_features_df_rows = imaging_data_df.query("image_name=='%s'"%current_image_name)
            
            # If the image is not present in the filtered dataset, then the values will be filled with -1.
            if (len(current_image_features_df_rows)>0):
                current_image_features = current_image_features_df_rows[current_image_features_df_rows.columns[1:]].iloc[0]
                current_patient_clinical_data = current_rxcode_info_df.iloc[0]
                current_merged_features_df = pd.concat([current_image_features, current_patient_clinical_data])
                global_merged_features_df = pd.concat([global_merged_features_df, current_merged_features_df], axis=1)

            it+=1

        global_merged_features_df = global_merged_features_df.T
        global_merged_features_df.columns = headers_list

        print('++++ OK: the clinical data was associated with the imaging features!')

        return headers_list, global_merged_features_df

    def __apply_progression_model__(self, headers_list, input_df):
        print('++++ Working with a single-timepoint dataset. The method __apply_progression_model__ will do nothing')

        return headers_list, input_df

# This class must be inherited by those classes that create datasets
# using only multiple image timepoints.
class Multiple_Timepoints_Build_Dataset():

    def __init__(self, **kwargs):
        pass

    def __get_corresponding_timepoints__(self, input_df):
        return input_df

    def __associate_clinical_and_imaging_datasets__(self, clinical_data_df, imaging_data_df):
        unique_patients_ids_list = np.unique(clinical_data_df['patient_id'])
        features_rows = []
        clinical_features_headers = clinical_data_df.columns.tolist()
        image_features_headers = imaging_data_df.columns.tolist()
        headers_list = clinical_features_headers + image_features_headers

        for patient_id_aux in unique_patients_ids_list:
            current_patient_images_codes = clinical_data_df.query("patient_id==%d"%patient_id_aux)['codigo'].values.tolist()
            for current_image_code_aux in current_patient_images_codes:
                current_image_name_aux = 'rxcode_%d.jpg'%current_image_code_aux
                current_image_features_aux = imaging_data_df.query("image_name=='%s'"%current_image_name_aux).values.tolist()
                if (len(current_image_features_aux)>0):
                    current_clinical_features_aux = clinical_data_df.query("codigo==%d"%current_image_code_aux).values.tolist()
                    features_rows.append(current_clinical_features_aux[0] + current_image_features_aux[0])

        global_merged_features_df = pd.DataFrame(features_rows)
        global_merged_features_df.columns = headers_list

        return headers_list, global_merged_features_df

    def __apply_progression_model__(self, headers_list, input_df):
        unique_patients_ids_list = np.unique(input_df['patient_id'])
        non_imaging_features_headers = list(filter(lambda input_value: input_value.find('feature_')==-1, headers_list))
        imaging_features_headers = list(filter(lambda input_value: input_value.find('feature_')!=-1, headers_list))
        nof_imaging_features = len(imaging_features_headers)
        lstm_progpred_model = LSTM_ProgPred_Model(device=self.device, nof_imaging_features=nof_imaging_features)
        features_rows = []

        for current_patient_id_aux in unique_patients_ids_list:
            current_patient_clinical_features = input_df.query("patient_id==%d"%current_patient_id_aux)
            current_patient_clinical_features = current_patient_clinical_features[non_imaging_features_headers].values.tolist()
            current_patient_features_df = input_df.query("patient_id==%d"%current_patient_id_aux)
            current_patient_processed_imaging_features = lstm_progpred_model.apply_progression_model(current_patient_features_df)
            features_rows.append(current_patient_clinical_features[0] + current_patient_processed_imaging_features)

        dropped_cols = ['primera_fecha_seguimiento', 'distancia_temporal', 'image_name']
        features_df = pd.DataFrame(features_rows)
        features_df.columns = headers_list
        features_df = features_df.drop(dropped_cols, axis=1)

        headers_list = list(filter(lambda input_value: input_value.find('output')==-1, features_df.columns))
        headers_list += ['output']
        features_df = features_df[headers_list]

        return headers_list, features_df

class Super_Class_Build_Dataset():

    def __init__(self, **kwargs):
        self.dropped_cols_in_simp_version = ['fecha_seguimiento', 'hallazgos', 'fecha_positivo', \
                                             'fecha_exitus', 'biologicos_cuales', 'primera_fecha_seguimiento', \
                                             'distancia_temporal']

    def __filter_cases_by_images_names_list__(self, clinical_data_df, imaging_data_df):
        rxcodes_list = clinical_data_df['codigo']
        columns_list = clinical_data_df.columns
        filtered_clinical_df_rows_list = []
        for current_rxcode_aux in rxcodes_list:
            image_name_found = len(imaging_data_df.query("image_name=='rxcode_%d.jpg'"%current_rxcode_aux))>0
            if (image_name_found):
                current_code_data_row_aux = clinical_data_df.query('codigo==%d and distancia_temporal==0'%current_rxcode_aux).values.tolist()
                filtered_clinical_df_rows_list += current_code_data_row_aux

        clinical_data_df = pd.DataFrame(filtered_clinical_df_rows_list)
        clinical_data_df.columns = columns_list

        return clinical_data_df

    def __get_images_features__(self, selected_approach_obj):
        
        # This function obtains the outcome of a patient if the clinical data is
        # available or an empty list otherwise. In case something is obtained,
        # it will be a list of a single element.
        def get_patient_outcome(rxcode):
            outcome_value = [np.nan]

            if (self.clinical_data_df is not None):
                current_rxcode_output = self.clinical_data_df.query("codigo==%d"%rxcode)['output'].values.tolist()
                if (len(current_rxcode_output)>0):
                    outcome_value = [current_rxcode_output[0]]

            return outcome_value

        print('++++ Retrieving images features...')
        
        features_list = []
        # We choose the images by the rx code, because it's unique.
        images_names_list = os.listdir(self.images_dir_root)
        it = 1
        total_length = len(images_names_list)
        for current_image_name_aux in images_names_list:
            print('++++ Processing image %s [%d/%d]'%(current_image_name_aux, it, total_length), end='\r')
            current_rxcode = int(current_image_name_aux.split('_')[1].split('.')[0])
            full_current_image_path = '%s/%s'%(self.images_dir_root, current_image_name_aux)
            current_input_image = read_image(full_current_image_path)
            headers_list, current_image_features_list_aux = selected_approach_obj.extract_features(current_input_image)
            rxcode_list_one_item_or_empty = get_patient_outcome(current_rxcode)
            features_list.append([current_image_name_aux] + current_image_features_list_aux + rxcode_list_one_item_or_empty)
            it+=1

        features_df = pd.DataFrame(features_list)
        headers_list = ['image_name'] + headers_list

        print('++++ OK: Images features were retrieved!')

        return headers_list, features_df

    def __build_dataset_scenario_II_aux__(self):
        universal_factory = UniversalFactory()
        kwargs = {'device': self.device, 'layer': self.layer}
        selected_approach_obj = universal_factory.create_object(globals(), self.chosen_approach, kwargs)

        imaging_data_columns, imaging_features_df = self.__get_images_features__(selected_approach_obj)
        imaging_features_df.columns = imaging_data_columns + ['output']
        # Removing the features whose values are all 0.
        idxs_to_keep = np.where((imaging_features_df==0).sum()<len(imaging_features_df))[0]
        imaging_features_df = imaging_features_df[imaging_features_df.columns[idxs_to_keep]]

        return imaging_features_df.columns, imaging_features_df

    # The scenario III merges the features of the scenario I with the features of the scenario II.
    def __build_dataset_scenario_III_aux__(self):
        _, clinical_data_df = self.build_full_dataset_scenario_I()
        self.clinical_data_df = clinical_data_df

        if (self.imaging_features_csv_path==None):
            print('++++ WARNING: NO IMAGING FEATURES FILE WAS SPECIFIED. COMPUTING THE DEEP FEATURES ON THE FLY.')
            print('THIS CAN SLOW THE PROCESS. SPECIFY AN IMAGING FEATURES FILE IF AVAILABLE TO PERFORM THE PROCESS QUICKER!')
            _, imaging_features_df = self.__build_dataset_scenario_II_aux__()
        else:
            imaging_features_df = pd.read_csv(self.imaging_features_csv_path)

        merged_features_columns, global_merged_features_df = self.__associate_clinical_and_imaging_datasets__(self.clinical_data_df, imaging_features_df)
        
        merged_features_columns, global_merged_features_df = self.__apply_progression_model__(merged_features_columns, global_merged_features_df)

        global_merged_features_df = self.__get_corresponding_timepoints__(global_merged_features_df)

        return merged_features_columns, global_merged_features_df

    def build_full_dataset_scenario_I(self):
        raise NotImplementedError('++++ The method build_full_dataset_scenario_I has not been implemented for this approach!')

    def build_simplified_dataset_scenario_I(self):
        _, clinical_data_df = self.build_full_dataset_scenario_I()

        imaging_data_df = pd.read_csv(self.imaging_features_csv_path)
        clinical_data_df = self.__filter_cases_by_images_names_list__(clinical_data_df, imaging_data_df)

        self.df_unique_values = clinical_data_df.drop(['codigo'] + self.dropped_cols_in_simp_version, axis=1).drop_duplicates()
        self.df_unique_values = self.df_unique_values.drop(['patient_id'], axis=1)

        return self.df_unique_values.columns, self.df_unique_values

    def build_full_dataset_scenario_II(self):
        raise NotImplementedError('++++ The method build_full_dataset_scenario_II has not been implemented for this approach!')

    def build_simplified_dataset_scenario_II(self):
        _, features_df = self.build_full_dataset_scenario_II()
        # In some scenarios, it is necessary to drop the column image_name.
        if ('image_name' in features_df.columns):
            features_df = features_df.drop(['image_name'], axis=1)
        
        valid_idxs = (1-np.isnan(features_df)['output']).values.astype(bool)
        features_df = features_df.iloc[valid_idxs]
        headers_list = features_df.columns

        return headers_list, features_df

    def build_full_dataset_scenario_III(self):
        raise NotImplementedError('++++ The method build_full_dataset_scenario_III has not been implemented for this approach!')

    def build_simplified_dataset_scenario_III(self):
        _, features_df = self.build_full_dataset_scenario_III()
        features_df = features_df.drop(['codigo', 'hallazgos', \
                                        'fecha_positivo', 'fecha_exitus', \
                                        'biologicos_cuales', 'fecha_seguimiento'], axis=1)
        features_df = features_df.drop_duplicates()
        features_df = features_df.drop(['patient_id'], axis=1)
        headers_list = features_df.columns

        return headers_list, features_df
    
    def store_dataset_in_csv_file(self, built_dataset_to_store, output_csv_file_path):
        built_dataset_to_store.to_csv(output_csv_file_path, index=False)
        print('**** The version of the built dataset has been stored at %s'%output_csv_file_path)

class Build_Dataset_Hospitalized_And_Urgencies(Super_Class_Build_Dataset, Single_Timepoint_Build_Dataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_csv_file_path = kwargs['input_csv_file_path']
        self.imaging_features_csv_path = None
        try:
            self.imaging_features_csv_path = kwargs['imaging_features_csv_path']
        except KeyError:
            pass

    def build_full_dataset_scenario_I(self):
        self.input_dataset_df = pd.read_csv(self.input_csv_file_path)
        self.input_dataset_df = self.input_dataset_df.query("cohorte==0 or cohorte==1")

        # The cohorte is placed as the last column of the dataset and exitus is
        # removed
        self.input_dataset_df = self.input_dataset_df.drop(['exitus'], axis=1)
        cohort_column = self.input_dataset_df['cohorte']
        self.input_dataset_df = self.input_dataset_df.drop(['cohorte'], axis=1)
        self.input_dataset_df.insert(len(self.input_dataset_df.columns), 'output', cohort_column)

        return self.input_dataset_df.columns, self.input_dataset_df

    # build_simplified_dataset is already implemented in the super class.

class Build_Dataset_Only_Hospitalized(Super_Class_Build_Dataset, Single_Timepoint_Build_Dataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_csv_file_path = kwargs['input_csv_file_path']
        self.imaging_features_csv_path = None
        try:
            self.imaging_features_csv_path = kwargs['imaging_features_csv_path']
        except KeyError:
            pass

    def build_full_dataset_scenario_I(self):
        self.input_dataset_df = pd.read_csv(self.input_csv_file_path)
        self.input_dataset_df = self.input_dataset_df.query("cohorte==1")

        # The cohorte is placed as the last column of the dataset and exitus is
        # removed
        exitus_column = self.input_dataset_df['exitus']
        self.input_dataset_df = self.input_dataset_df.drop(['exitus'], axis=1)
        self.input_dataset_df = self.input_dataset_df.drop(['cohorte'], axis=1)
        # The column with the rx codes must be added after dropping repeated rows,
        # given that each row has a different rx code.
        self.input_dataset_df.insert(len(self.input_dataset_df.columns), 'output', exitus_column)

        return self.input_dataset_df.columns, self.input_dataset_df

    # build_simplified_dataset is already implemented in the super class.

class Build_Dataset_VGG_16_Model_All_Cases(Super_Class_Build_Dataset, Single_Timepoint_Build_Dataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_csv_file_path = kwargs['input_csv_file_path']
        self.images_dir_root = kwargs['images_dir_root']
        self.device = kwargs['device']
        self.layer = kwargs['layer']
        self.chosen_approach = 'VGG_16_Deep_Features_Model'

    def build_full_dataset_scenario_II(self):
        self.clinical_data_df = None
        headers_list, features_df = self.__build_dataset_scenario_II_aux__()
        features_df = features_df.drop(['output'], axis=1)

        return headers_list, features_df

class Build_Dataset_VGG_16_ProgPred_Model_All_Cases(Build_Dataset_VGG_16_Model_All_Cases, Multiple_Timepoints_Build_Dataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Build_Dataset_VGG_16_Model_All_Cases.__init__(self, **kwargs)
        self.chosen_approach = 'VGG_16_Progression_Predictor_Model'

class Build_Dataset_VGG_16_Model_Only_Hospitalized(Build_Dataset_Only_Hospitalized, Single_Timepoint_Build_Dataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_csv_file_path = kwargs['input_csv_file_path']
        self.imaging_features_csv_path = kwargs['imaging_features_csv_path']
        self.images_dir_root = kwargs['images_dir_root']
        self.device = kwargs['device']
        self.layer = kwargs['layer']
        self.chosen_approach = 'VGG_16_Deep_Features_Model'

    def build_full_dataset_scenario_II(self):
        _, features_df = self.build_full_dataset_scenario_III()
        get_columns_to_drop_func = \
            lambda input_value: (input_value.find('feature_')==-1 and input_value.find('output')==-1)
        columns_to_drop = list(filter(get_columns_to_drop_func, features_df.columns))
        features_df = features_df.drop(columns_to_drop, axis=1)

        return features_df.columns, features_df

    def build_full_dataset_scenario_III(self):
        headers_list, features_df = super().__build_dataset_scenario_III_aux__()

        return headers_list, features_df

class Build_Dataset_VGG_16_ProgPred_Model_Only_Hospitalized(Multiple_Timepoints_Build_Dataset, Build_Dataset_VGG_16_Model_Only_Hospitalized):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Build_Dataset_VGG_16_Model_Only_Hospitalized.__init__(self, **kwargs)
        self.chosen_approach = 'VGG_16_Progression_Predictor_Model'

class Build_Dataset_VGG_16_Model_Hospitalized_And_Urgencies(Build_Dataset_Hospitalized_And_Urgencies, Single_Timepoint_Build_Dataset):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_csv_file_path = kwargs['input_csv_file_path']
        self.imaging_features_csv_path = kwargs['imaging_features_csv_path']
        self.images_dir_root = kwargs['images_dir_root']
        self.device = kwargs['device']
        self.layer = kwargs['layer']
        self.chosen_approach = 'VGG_16_Deep_Features_Model'

    def build_full_dataset_scenario_II(self):
        _, features_df = self.build_full_dataset_scenario_III()
        get_columns_to_drop_func = \
            lambda input_value: (input_value.find('feature_')==-1 and input_value.find('output')==-1)
        columns_to_drop = list(filter(get_columns_to_drop_func, features_df.columns))
        features_df = features_df.drop(columns_to_drop, axis=1)

        return features_df.columns, features_df

    def build_full_dataset_scenario_III(self):
        headers_list, features_df = super().__build_dataset_scenario_III_aux__()

        return headers_list, features_df

class Build_Dataset_VGG_16_ProgPred_Model_Hospitalized_And_Urgencies(Multiple_Timepoints_Build_Dataset, Build_Dataset_VGG_16_Model_Hospitalized_And_Urgencies):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Build_Dataset_VGG_16_Model_Hospitalized_And_Urgencies.__init__(self, **kwargs)
        self.chosen_approach = 'VGG_16_Progression_Predictor_Model'