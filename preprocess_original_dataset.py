import argparse
import copy
from datetime import datetime
import numpy as np
import pandas as pd

class Preprocess_Dataset():

    def __init__(self):
        pass

    def __preprocess_discrete_attrs__(self):
        for column_aux in self.df_discrete_attrs_headers:
            self.df_preprocessed_data[column_aux] = self.df_preprocessed_data[column_aux].fillna(0)
            self.df_preprocessed_data[column_aux] = self.df_preprocessed_data[column_aux].astype(float).astype(int)

    def __preprocess_numerical_attrs__(self):
        for column_aux in self.df_numerical_attrs_headers:
            self.df_preprocessed_data[column_aux] = self.df_preprocessed_data[column_aux].fillna(-1)

            self.df_preprocessed_data[column_aux] = self.df_preprocessed_data[column_aux].astype(str).str. \
                replace(r'([0-9]+),([0-9]+)', r'\1.\2', regex=True). \
                    replace(r'[a-zA-ZÀ-ÿ\u00f1\u00d1]+', r'', regex=True). \
                        replace(r'(,)*', r'', regex=True).str. \
                            replace(r'<([0-9]+)', r'\1', regex=True). \
                                replace(r'>([0-9])', r'\1', regex=True). \
                                    replace(r'[#¡!+]+', r'', regex=True).str.replace(' ', ''). \
                                        replace(r'^$', r'-1', regex=True)

            if (column_aux in self.original_dataset_df.columns[self.int_attrs_idxs]):
                self.df_preprocessed_data[column_aux] = self.df_preprocessed_data[column_aux].astype(float).astype(int)
            else:
                self.df_preprocessed_data[column_aux] = self.df_preprocessed_data[column_aux].astype(float)

    def __create_and_add_patients_ids__(self, input_df):
        patients_ids_list = []
        patient_id_number_dict = {}
        idx = 0
        for row_aux in input_df.values:
            current_patient_id = ''
            for column_aux in row_aux:
                current_patient_id += str(column_aux)
            patient_id_number_dict[current_patient_id] = idx
            patients_ids_list.append(current_patient_id)
            idx+=1

        input_df.insert(0, 'patient_id', patients_ids_list)
        processed_df = copy.copy(input_df)
        for idx in range(0, len(input_df)):
            current_patient_id = input_df.iloc[idx]['patient_id']
            processed_df = processed_df.replace(current_patient_id, patient_id_number_dict[current_patient_id])

        return processed_df

    def __add_first_following_date__(self, input_df):
        patients_ids_list = input_df['patient_id'].values
        first_following_dates_list = []
        for patient_id_aux in patients_ids_list:
            format_dates_func = lambda input_value: datetime.strptime(input_value, '%d/%m/%Y')
            following_dates_list = list(map(format_dates_func, input_df.query("patient_id==%d"%patient_id_aux)['fecha_seguimiento'].values))
            first_following_date = np.min(following_dates_list)
            first_following_dates_list.append(first_following_date)
        input_df.insert(len(input_df.columns), 'primera_fecha_seguimiento', first_following_dates_list)

        return input_df
    
    def __add_last_following_date__(self, input_df):
        patients_ids_list = input_df['patient_id'].values
        last_following_dates_list = []
        for patient_id_aux in patients_ids_list:
            format_dates_func = lambda input_value: datetime.strptime(input_value, '%d/%m/%Y')
            following_dates_list = list(map(format_dates_func, input_df.query("patient_id==%d"%patient_id_aux)['fecha_seguimiento'].values))
            last_following_date = np.max(following_dates_list)
            last_following_dates_list.append(last_following_date)
        input_df.insert(len(input_df.columns), 'ultima_fecha_seguimiento', last_following_dates_list)

        return input_df

    def __add_whole_followup_time__(self, input_df):
        patients_ids_list = input_df['patient_id'].values
        whole_followup_times_list = []
        for patient_id_aux in patients_ids_list:
            format_dates_func = lambda input_value: datetime.strptime(input_value, '%d/%m/%Y')
            following_dates_list = list(map(format_dates_func, input_df.query("patient_id==%d"%patient_id_aux)['fecha_seguimiento'].values))
            whole_followup_time = (np.max(following_dates_list) - np.min(following_dates_list)).days
            whole_followup_times_list.append(whole_followup_time)
        input_df.insert(len(input_df.columns), 'tiempo_total_seguimiento', whole_followup_times_list)

        return input_df

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

        self.df_preprocessed_data = self.original_dataset_df[self.df_columns_in_order]

        self.df_preprocessed_data = self.__create_and_add_patients_ids__(self.df_preprocessed_data)

        self.df_preprocessed_data = self.df_preprocessed_data.replace('Urgencias', 0)
        self.df_preprocessed_data = self.df_preprocessed_data.replace('Hospitalizados', 1)
        self.df_preprocessed_data = self.df_preprocessed_data.replace('Residencias', 2)
        self.df_preprocessed_data = self.df_preprocessed_data.replace('Controles', 3)

        self.df_preprocessed_data = self.df_preprocessed_data.replace('Hombre', 0)
        self.df_preprocessed_data = self.df_preprocessed_data.replace('Mujer', 1)

        self.df_preprocessed_data = self.df_preprocessed_data.replace('<65', 0)
        self.df_preprocessed_data = self.df_preprocessed_data.replace('[65,80]', 1)
        self.df_preprocessed_data = self.df_preprocessed_data.replace('>80', 2)

        self.df_preprocessed_data = self.df_preprocessed_data.replace('No', 0)
        self.df_preprocessed_data = self.df_preprocessed_data.replace('Desconocido', 0)
        self.df_preprocessed_data = self.df_preprocessed_data.replace('Si', 1)

        self.__preprocess_numerical_attrs__()
        self.__preprocess_discrete_attrs__()

        current_column = 'codigo'
        self.df_preprocessed_data.insert(1, current_column, self.original_dataset_df[current_column])
        current_column = 'fecha_seguimiento'
        self.df_preprocessed_data.insert(2, current_column, self.original_dataset_df[current_column])
        current_column = 'hallazgos'
        self.df_preprocessed_data.insert(3, current_column, self.original_dataset_df[current_column])
        current_column = 'fecha_positivo'
        self.df_preprocessed_data.insert(4, current_column, self.original_dataset_df[current_column])
        current_column = 'fecha_exitus'
        self.df_preprocessed_data.insert(5, current_column, self.original_dataset_df[current_column])
        current_column = 'biologicos_cuales'
        self.df_preprocessed_data.insert(6, current_column, self.original_dataset_df[current_column])
        self.df_preprocessed_data = self.__add_first_following_date__(self.df_preprocessed_data)
        self.df_preprocessed_data = self.__add_last_following_date__(self.df_preprocessed_data)
        self.df_preprocessed_data = self.__add_whole_followup_time__(self.df_preprocessed_data)
        
        format_dates_func = lambda input_value: datetime.strptime(input_value, '%d/%m/%Y')
        current_following_date_df = self.df_preprocessed_data['fecha_seguimiento'].apply(format_dates_func)
        first_following_date_df = self.df_preprocessed_data['primera_fecha_seguimiento']
        self.df_preprocessed_data['distancia_temporal'] = current_following_date_df - first_following_date_df
        self.df_preprocessed_data['distancia_temporal'] = self.df_preprocessed_data['distancia_temporal'].apply(lambda input_value: input_value.days)
        self.df_preprocessed_data = self.df_preprocessed_data.replace(np.nan, '[empty]')
        self.df_preprocessed_data = self.df_preprocessed_data.sort_values(by=['patient_id', 'fecha_seguimiento'])

        return self.df_preprocessed_data

    def build_dataset(self, input_filename, headers_file, \
                            padding_for_missing_values=-1, discretize=False):
        return self.__preprocess_dataset__(input_filename, headers_file)

    def store_dataset_in_csv_file(self, built_dataset_to_store, output_csv_file_path):
        built_dataset_to_store.to_csv(output_csv_file_path, index=False)
        print('**** The version of the built dataset has been stored at %s'%output_csv_file_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_file_path', type=str, required=True)
    parser.add_argument('--headers_associations_file_path', type=str, required=True)
    parser.add_argument('--output_csv_file_path', type=str, required=True)
    
    args = parser.parse_args()
    
    approach_obj = Preprocess_Dataset()
    dataset_rows = approach_obj.build_dataset(args.input_csv_file_path, args.headers_associations_file_path)
    approach_obj.store_dataset_in_csv_file(dataset_rows, args.output_csv_file_path)

main()
