import csv
from datetime import datetime
import sys
sys.path.append('../')
import numpy as np
import os
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

class No_Discretizer():

    def __init__(self):
        pass

    def discretize_linfocitos(self, input_value, padding_for_missing_values=-1):
        return input_value

    def discretize_dimeros_d(self, input_value, padding_for_missing_values=-1):
        return input_value

    def discretize_ldh(self, input_value, padding_for_missing_values=-1):
        return input_value

    def discretize_creatinina(self, input_value, padding_for_missing_values=-1):
        return input_value

    def discretize_filtrado_glomerular_estimado(self, input_value, padding_for_missing_values=-1):
        return input_value

    def discretize_proteina_c_reactiva(self, input_value, padding_for_missing_values=-1):
        return input_value

    def discretize_ferritina(self, input_value, padding_for_missing_values=-1):
        return input_value

    def discretize_il6(self, input_value, padding_for_missing_values=-1):
        return input_value

class Discretizer(No_Discretizer):

    def __init__(self):
        pass

    def discretize_linfocitos(self, input_value_converted, padding_for_missing_values=-1):
        output_value = padding_for_missing_values

        try:
            if ((input_value_converted<0.77) and (input_value_converted>0)):
                output_value = 0
            elif ((input_value_converted>=0.77) and (input_value_converted<=4.5)):
                output_value = 1
            elif (input_value_converted>4.5):
                output_value = 2
        except:
            pass

        return output_value

    def discretize_dimeros_d(self, input_value_converted, padding_for_missing_values=-1):
        output_value = padding_for_missing_values

        try:
            if ((input_value_converted>500)):
                output_value = 0
            elif ((input_value_converted<=500) and (input_value_converted>0)):
                output_value = 1
        except:
            pass

        return output_value

    def discretize_ldh(self, input_value_converted, padding_for_missing_values=-1):
        output_value = padding_for_missing_values

        try:
            if ((input_value_converted<60) and (input_value_converted>0)):
                output_value = 0
            elif ((input_value_converted>=60) and (input_value_converted<=160)):
                output_value = 1
            elif (input_value_converted>160):
                output_value = 2
        except:
            pass

        return output_value

    def discretize_creatinina(self, input_value_converted, padding_for_missing_values=-1):
        output_value = padding_for_missing_values

        try:
            if ((input_value_converted<0.7) and (input_value_converted>0)):
                output_value = 0
            elif ((input_value_converted>=0.7) and (input_value_converted<=1.3)):
                output_value = 1
            elif (input_value_converted>1.3):
                output_value = 2
        except:
            pass

        return output_value

    def discretize_filtrado_glomerular_estimado(self, input_value_converted, padding_for_missing_values=-1):
        output_value = padding_for_missing_values

        try:
            if ((input_value_converted<15) and (input_value_converted>0)):
                output_value = 0
            elif (input_value_converted>60):
                output_value = 1
            elif ((input_value_converted>=15) and (input_value_converted<=60)):
                output_value = 2
        except:
            pass

        return output_value

    def discretize_proteina_c_reactiva(self, input_value_converted, padding_for_missing_values=-1):
        output_value = padding_for_missing_values

        try:
            if (input_value_converted>0.5):
                output_value = 0
            elif ((input_value_converted<=0.5) and (input_value_converted>0)):
                output_value = 1
        except:
            pass

        return output_value

    def discretize_ferritina(self, input_value_converted, padding_for_missing_values=-1):
        output_value = padding_for_missing_values

        try:
            if ((input_value_converted<30) and (input_value_converted>0)):
                output_value = 0
            elif ((input_value_converted>=30) and (input_value_converted<=300)):
                output_value = 1
            elif (input_value_converted>300):
                output_value = 2
        except:
            pass

        return output_value

    def discretize_il6(self, input_value_converted, padding_for_missing_values=-1):
        output_value = padding_for_missing_values

        try:
            if ((input_value_converted<40) and (input_value_converted>0)):
                output_value = 0
            elif (input_value_converted>=40):
                output_value = 1
        except:
            pass

        return output_value

'''
    NOTE: this code is ad-hoc to the problem herein proposed. Therefore, if
    you want to use it for another different problem, it could be probably more
    useful to start another different script from scratch.
'''

def convert_cohorte(input_value):
    output_value = -1

    if (input_value=='Hospitalizados'):
        output_value = 1
    elif (input_value=='Residencias'):
        output_value = 2
    elif (input_value=='Urgencias'):
        output_value = 0

    return output_value

def convert_rango_edad(input_value):
    output_value = -1

    if (input_value=='<65'):
        output_value = 0
    elif (input_value=='[65,80]'):
        output_value = 1
    elif (input_value=='>80'):
        output_value = 2

    return output_value

def convert_sexo(input_value):
    output_value = -1

    if (input_value=='Hombre'):
        output_value = 0
    elif (input_value=='Mujer'):
        output_value = 1

    return output_value

# This function converts the strings to integers, but if the input is an empty
# string, it returns -1.
def convert_integer_attributes(input_value):
    output_value = -1

    try:
        output_value = int(input_value)
    except:
        pass

    return output_value

# This function converts the strings to decimal numbers, but if the input is
# an empty string, it returns -1.0.
def convert_double_attributes(input_value):
    output_value = -1.0

    try:
        output_value = float(input_value)
    except:
        pass

    return output_value

# This function is valid for an important amount of attributes.
# "padding_for missing_values" will determine what value will be in those cells
# with missing values in the original dataset.
def convert_si_no_desconocido_missing_to_integer(input_value, \
                                                padding_for_missing_values=-1):
    output_value = padding_for_missing_values

    if (input_value=='No'):
        output_value = 0
    elif (input_value=='Si'):
        output_value = 1

    return output_value

# This function allows to convert the fields to the desired format.
# This is an ad-hoc function. "padding_for missing_values" will determine what
# value will be in those cells with missing values in the original dataset.
def convert_fields(file_data, padding_for_missing_values=-1, discretize=False):
    output_list = []
    if (discretize):
        discretizer_obj = Discretizer()
    else:
        discretizer_obj = No_Discretizer()

    for item_aux in file_data:
        row_list_aux = []
        column_number_aux = 0
        for field in item_aux:
            try:
                # code
                if (column_number_aux==0):
                    row_list_aux.append(convert_integer_attributes(field))
                # fecha_seguimiento
                if (column_number_aux==1):
                    # row_list_aux.append(datetime.strptime(field, '%d/%m/%Y'))
                    row_list_aux.append(field)
                # hallazgos
                if (column_number_aux==2):
                    row_list_aux.append(field)
                # fecha_rx
                if (column_number_aux==3):
                    row_list_aux.append(int(field))
                # fecha_positivo
                if (column_number_aux==4):
                    # row_list_aux.append(datetime.strptime(field, '%d/%m/%Y'))
                    row_list_aux.append(field)
                # exitus
                if (column_number_aux==5):
                    row_list_aux.append(convert_si_no_to_int(field))
                # fecha_exitus
                if (column_number_aux==6):
                    # row_list_aux.append(datetime.strptime(field, '%d/%m/%Y'))
                    row_list_aux.append(field)
                # cohorte
                if (column_number_aux==7):
                    row_list_aux.append(convert_cohorte(field))
                # rango_edad
                if (column_number_aux==8):
                    row_list_aux.append(convert_rango_edad(field))
                # sexo
                if (column_number_aux==9):
                    row_list_aux.append(convert_sexo(field))
                # edad
                if (column_number_aux==10):
                    row_list_aux.append(int(field))
                # talla_cm
                if (column_number_aux==11):
                    row_list_aux.append(convert_integer_attributes(field))
                # peso_kg
                if (column_number_aux==12):
                    row_list_aux.append(convert_double_attributes(field.replace(',', '.')))
                # imc
                if (column_number_aux==13):
                    row_list_aux.append(convert_double_attributes(field.replace(',', '.')))
                # hta
                if (column_number_aux==14):
                    converted_value = convert_si_no_desconocido_missing_to_integer(field, padding_for_missing_values)
                    row_list_aux.append(converted_value)
                # diabetes_mellitus
                if (column_number_aux==15):
                    converted_value = convert_si_no_desconocido_missing_to_integer(field, padding_for_missing_values)
                    row_list_aux.append(converted_value)
                # epoc
                if (column_number_aux==16):
                    converted_value = convert_si_no_desconocido_missing_to_integer(field, padding_for_missing_values)
                    row_list_aux.append(converted_value)
                # asma
                if (column_number_aux==17):
                    converted_value = convert_si_no_desconocido_missing_to_integer(field, padding_for_missing_values)
                    row_list_aux.append(converted_value)
                # hepatopatia
                if (column_number_aux==18):
                    converted_value = convert_si_no_desconocido_missing_to_integer(field, padding_for_missing_values)
                    row_list_aux.append(converted_value)
                # leucemia
                if (column_number_aux==19):
                    converted_value = convert_si_no_desconocido_missing_to_integer(field, padding_for_missing_values)
                    row_list_aux.append(converted_value)
                # linfoma
                if (column_number_aux==20):
                    converted_value = convert_si_no_desconocido_missing_to_integer(field, padding_for_missing_values)
                    row_list_aux.append(converted_value)
                # neoplasia
                if (column_number_aux==21):
                    converted_value = convert_si_no_desconocido_missing_to_integer(field, padding_for_missing_values)
                    row_list_aux.append(converted_value)
                # hiv
                if (column_number_aux==22):
                    converted_value = convert_si_no_desconocido_missing_to_integer(field, padding_for_missing_values)
                    row_list_aux.append(converted_value)
                # transplante_organo_solido
                if (column_number_aux==23):
                    converted_value = convert_si_no_desconocido_missing_to_integer(field, padding_for_missing_values)
                    row_list_aux.append(converted_value)
                # quimioterapia_ultimos_3_meses
                if (column_number_aux==24):
                    converted_value = convert_si_no_desconocido_missing_to_integer(field, padding_for_missing_values)
                    row_list_aux.append(converted_value)
                # biologicos_ultimos_3_meses
                if (column_number_aux==25):
                    converted_value = convert_si_no_desconocido_missing_to_integer(field, padding_for_missing_values)
                    row_list_aux.append(converted_value)
                # biologicos_cuales
                if (column_number_aux==26):
                    converted_value = convert_si_no_desconocido_missing_to_integer(field, padding_for_missing_values)
                    row_list_aux.append(converted_value)
                # corticoides_cronicos_mas_3_meses
                if (column_number_aux==27):
                    converted_value = convert_si_no_desconocido_missing_to_integer(field, padding_for_missing_values)
                    row_list_aux.append(converted_value)
                # linfocitos
                if (column_number_aux==28):
                    converted_value = convert_double_attributes(field.replace(',', '.'))
                    converted_value = discretizer_obj.discretize_linfocitos(converted_value)
                    row_list_aux.append(converted_value)
                # linfocitos_porcentaje
                if (column_number_aux==29):
                    row_list_aux.append(convert_double_attributes(field.replace(',', '.')))
                # dimeros_d
                if (column_number_aux==30):
                    converted_value = convert_integer_attributes(field)
                    converted_value = discretizer_obj.discretize_dimeros_d(converted_value)
                    row_list_aux.append(converted_value)
                # ldh
                if (column_number_aux==31):
                    converted_value = convert_integer_attributes(field)
                    converted_value = discretizer_obj.discretize_ldh(converted_value)
                    row_list_aux.append(converted_value)
                # creatinina
                if (column_number_aux==32):
                    converted_value = convert_double_attributes(field.replace(',', '.'))
                    converted_value = discretizer_obj.discretize_creatinina(converted_value)
                    row_list_aux.append(converted_value)
                # filtrado_glomerular_estimado
                if (column_number_aux==33):
                    converted_value = convert_double_attributes(field.replace(',', '.'))
                    converted_value = discretizer_obj.discretize_filtrado_glomerular_estimado(converted_value)
                    row_list_aux.append(converted_value)
                # prc
                if (column_number_aux==34):
                    converted_value = convert_double_attributes(field.replace(',', '.'))
                    converted_value = discretizer_obj.discretize_proteina_c_reactiva(converted_value)
                    row_list_aux.append(converted_value)
                # ferritina
                if (column_number_aux==35):
                    converted_value = convert_integer_attributes(field)
                    converted_value = discretizer_obj.discretize_ferritina(converted_value)
                    row_list_aux.append(converted_value)
                # il6
                if (column_number_aux==36):
                    converted_value = convert_double_attributes(field.replace(',', '.'))
                    converted_value = discretizer_obj.discretize_il6(converted_value)
                    row_list_aux.append(converted_value)
            except:
                row_list_aux.append(field)

            column_number_aux+=1

        output_list.append(row_list_aux)

    return output_list

def plot_missing_values_histogram(nof_missing_values_per_row):
    fig, ax = plt.subplots()
    plt.hist(np.array(nof_missing_values_per_row), \
                    bins=len(np.unique(np.array(nof_missing_values_per_row))), \
                        rwidth=0.7)

    frequencies = np.unique(np.array(nof_missing_values_per_row))
    rects = ax.patches
    x_axis = []
    for rect_aux in rects:
        height = rect_aux.get_height()
        ax.text(rect_aux.get_x() + rect_aux.get_width() / 2, height+0.01, int(height),
                ha='center', va='bottom')
        x_axis.append(rect_aux.get_x() + rect_aux.get_width() / 2)

    plt.title('Cantidad de filas que tienen un número determinado de missing values')
    plt.xlabel('Número de missing values')
    plt.ylabel('Número de filas')
    ax.set_xticks(x_axis, frequencies)
    plt.savefig('missing_values_dist.pdf')

# This function removes those registers that have all the fields missing.
def remove_useless_data(dataset_rows, attr_columns_to_check, padding_missing_values=-1):
    indexes = attr_columns_to_check
    # This list will store the indexes of those rows that do not have useless
    # data.
    useful_rows = []
    nof_useless_rows = 0
    nof_missing_values_per_row = []

    tmp_dataset_rows = np.copy(dataset_rows)
    tmp_dataset_rows = tmp_dataset_rows.astype(np.float64).astype(str)[:, indexes]
    nofrow_aux = 0
    for row_aux in tmp_dataset_rows:
        unique_values = np.unique(row_aux)
        # "padding_missing_values" is first converted to float to ensure that,
        # for example, "-1" is converted to "-1.0".
        nof_missing_values_aux = row_aux.tolist().count(str(float(padding_missing_values)))
        if ((len(unique_values)==1) and (unique_values[0]==str(float(padding_missing_values)))):
            nof_useless_rows+=1
        else:
            if (nof_missing_values_aux>0):
                nof_missing_values_per_row.append(nof_missing_values_aux)
            useful_rows.append(nofrow_aux)
        nofrow_aux+=1

    dataset_rows = dataset_rows[useful_rows, :]

    print('**** There were %d useless rows that have been deleted.'%nof_useless_rows)

    return dataset_rows

def filter_populated_rows(threshold, dataset_rows):
    filtered_dataset_rows = []

    for row_aux in dataset_rows:
        non_missing_values = 0
        for column_aux in row_aux:
            if (column_aux!='-1'):
                non_missing_values+=1
        if (non_missing_values/len(row_aux)>threshold):
            filtered_dataset_rows.append(row_aux)

    return filtered_dataset_rows

class Super_Class_Build_Dataset():

    def __init__(self):
        pass

    def build_dataset(self, input_filename, headers_file, \
                            padding_for_missing_values=-1, discretize=False):
        print('++++ WARNING: the method build_dataset has not been implemented for this approach!')

    def build_dataset_with_deep_features(self):
        print('++++ WARNING: the method build_dataset_with_deep_features has not been implemented for this approach!')

class Build_Dataset_Hospitalized_And_Urgencies(Super_Class_Build_Dataset):

    def __init__(self, **kwargs):
        pass

    def build_dataset(self, input_filename, headers_file, \
                            padding_for_missing_values=-1, discretize=False):
        headers, file_data = read_csv_file(input_filename)
        list_with_fields_converted = \
            np.array(convert_fields(file_data, padding_for_missing_values))

        indexes = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, \
                      23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 7]
        hospitalized_cases_indexes = np.where(np.array(file_data)[:, 7]=='Hospitalizados')
        urgencies_cases_indexes = np.where(np.array(file_data)[:, 7]=='Urgencias')
        # This refers to the cases of the hospitalized and the urgencies
        # patients.
        selected_cases_indexes = \
            np.concatenate((urgencies_cases_indexes, hospitalized_cases_indexes), axis=1)

        dataset_rows = list_with_fields_converted[:, indexes]
        dataset_rows = np.array(dataset_rows).astype(str)

        # For this problem, the class 2 (i.e. "Urgencias") must be converted to
        # class 1.
        dataset_rows_shape = np.shape(dataset_rows)
        last_item_index = dataset_rows_shape[1]-1
        convert_urgencies_to_class_1 = \
            lambda input: input.replace('2', '1')
        # Obtaining the index to select the output column.
        dataset_rows[:, last_item_index] = \
            np.array(list(map(convert_urgencies_to_class_1, dataset_rows[:, last_item_index])))

        # Using only the selected patients to build the dataset.
        dataset_rows = dataset_rows[selected_cases_indexes, :]

        headers = np.array(read_headers_file(headers_file))
        # As cohort is selected as the output, the word 'cohort' will be replaced
        # with output to store it like this in the CSV file.
        change_cohort_to_output = lambda input: input.replace('cohorte', 'output')
        headers_to_store = list(map(change_cohort_to_output, headers[indexes]))

        # Removing duplicated data.
        dataset_rows = np.unique(dataset_rows[0], axis=0)

        # As some of the attributes are previously removed, the indexes change and
        # therefore they must be specified again.
        new_indexes_to_check = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, \
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        dataset_rows = remove_useless_data(dataset_rows, \
                              new_indexes_to_check, padding_for_missing_values)

        return headers_to_store, dataset_rows

class Build_Dataset_Only_Hospitalized(Super_Class_Build_Dataset):

    def __init__(self, **kwargs):
        pass

    def build_dataset(self, input_filename, headers_file, \
                            padding_for_missing_values=-1, discretize=False):
        headers, file_data = read_csv_file(input_filename)
        list_with_fields_converted = \
            np.array(convert_fields(file_data, padding_for_missing_values, discretize))

        indexes = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, \
                      23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 5]
        hospitalized_cases_indexes = np.where(np.array(file_data)[:, indexes[0]]=='Hospitalizados')
        dataset_rows = list_with_fields_converted[:, indexes]
        dataset_rows = np.array(dataset_rows).astype(str)

        # Using only the "Hospitalized" patients to build the dataset.
        dataset_rows = dataset_rows[hospitalized_cases_indexes, :]

        headers = np.array(read_headers_file(headers_file))
        # As exitus is selected as the output, the word 'exitus' will be replaced
        # with output to store it like this in the CSV file.
        change_exitus_to_output = lambda input: input.replace('exitus', 'output')
        headers_to_store = list(map(change_exitus_to_output, headers[indexes]))

        # Removing duplicated data.
        dataset_rows = np.unique(dataset_rows[0], axis=0)

        # Remove the cohort, because it is not necessary in this particular case.
        headers_to_store = headers_to_store[1:]
        dataset_rows = dataset_rows[:, 1:]

        # As some of the attributes are previously removed, the indexes change and
        # therefore they must be specified again.
        new_indexes_to_check = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, \
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        dataset_rows = remove_useless_data(dataset_rows, \
                              new_indexes_to_check, padding_for_missing_values)

        return headers_to_store, dataset_rows

    # Just to clear out, in this function, "deepf" denotes "deep features".
    '''
    ###########################################################################
       NOTE: This function stores row by row directly for efficiency reasons.
    ###########################################################################
    '''
    def build_dataset_with_deep_features(self, deepf_model_name, headers_file, \
                                 input_dataset_dir, input_table_file, \
                                     associations_file, output_path, **kwargs):
        _, clinical_data_rows = read_csv_file(input_table_file, has_headers=True)
        clinical_data_rows = convert_fields(clinical_data_rows)
        clinical_data_rows = np.array(clinical_data_rows)
        headers_to_store = read_headers_file(headers_file)

        _, associations_rows = read_csv_file(associations_file, has_headers=False)
        associations_rows = np.array(associations_rows)

        universal_factory = UniversalFactory()

        kwargs_deepf_model = {'device': kwargs['device']}
        deep_features_model = universal_factory.create_object(globals(), \
                        deepf_model_name + '_Deep_Features_Model', kwargs_deepf_model)

        # Finally, we remove some of the fields that we do not need.
        attr_indexes = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                           23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
        # This function is used to replace the string "exitus" by the string
        # "output".
        replace_field_exitus_by_output = lambda input: input.replace('exitus', 'output')
        headers_to_store = np.array(headers_to_store)[attr_indexes].tolist()
        headers_to_store = list(map(replace_field_exitus_by_output, headers_to_store))

        images_names_list = os.listdir(input_dataset_dir)

        with open(output_path, 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            # First of all, we need to write the headers of the fields.
            csv_writer.writerow(headers_to_store[1:])

            total_number_of_images = len(images_names_list)
            spinner_states = ['|', '/', '-', '\\']
            it = 0
            for image_name_aux in images_names_list:
                image_path = '%s/%s'%(input_dataset_dir, image_name_aux)
                # First of all, we need to check if the current image name has a
                # code associated.
                current_image_code_idx = np.where(associations_rows[:, 0]==image_name_aux)
                if (np.any(current_image_code_idx)):
                    # This variable stores the code that corresponds to the current
                    # image file name.
                    current_image_code = associations_rows[current_image_code_idx[0][0], 1]
                    # This variable specifies the index of the current code at the
                    # clinical data table.
                    current_code_idx = np.where(clinical_data_rows[:, 0]==current_image_code)
                    # We also need to check if the image code has a filename
                    # associated.
                    if (np.any(current_code_idx)):
                        # This is the row of the clinical data table that belongs
                        # to the current image code.
                        current_clinical_row = clinical_data_rows[current_code_idx[0][0], :]
                        if (current_clinical_row[7]=='0'):
                            input_image = deep_features_model.read_image(image_path)
                            features_array = deep_features_model.extract_deep_features(input_image)
                            row_to_write = current_clinical_row[attr_indexes[1:]].tolist() + \
                                               features_array + [current_clinical_row[5]]
                            csv_writer.writerow(row_to_write)
                progress = int((it/total_number_of_images)*100)
                print('%s Progress = %d%s '%(spinner_states[it%4], progress, '%'), end='\r')
                it+=1

            print('++++ The new built dataset has been stored at %s'%output_path)

class Build_Dataset_Only_Hospitalized_Only_Deep_Features(Build_Dataset_Only_Hospitalized, \
                                                  Super_Class_Build_Dataset):

    def __init__(self, **kwargs):
        pass

    # Just to clear out, in this function, "deepf" denotes "deep features".
    '''
    ###########################################################################
       NOTE: This function stores row by row directly for efficiency reasons.
    ###########################################################################
    '''
    def build_dataset_with_deep_features(self, deepf_model_name, headers_file, \
                                 input_dataset_dir, input_table_file, \
                                     associations_file, output_path, **kwargs):
        _, clinical_data_rows = read_csv_file(input_table_file, has_headers=True)
        clinical_data_rows = convert_fields(clinical_data_rows)
        clinical_data_rows = np.array(clinical_data_rows)
        headers_to_store = read_headers_file(headers_file)

        _, associations_rows = read_csv_file(associations_file, has_headers=False)
        associations_rows = np.array(associations_rows)

        universal_factory = UniversalFactory()

        kwargs_deepf_model = {'device': kwargs['device']}
        deep_features_model = universal_factory.create_object(globals(), \
                        deepf_model_name + '_Deep_Features_Model', kwargs_deepf_model)

        # Finally, we remove some of the fields that we do not need.
        attr_indexes = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                           23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
        # This function is used to replace the string "exitus" by the string
        # "output".
        replace_field_exitus_by_output = lambda input: input.replace('exitus', 'output')
        headers_to_store = np.array(headers_to_store)[attr_indexes].tolist()
        headers_to_store = list(map(replace_field_exitus_by_output, headers_to_store))

        images_names_list = os.listdir(input_dataset_dir)

        with open(output_path, 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            # First of all, we need to write the headers of the fields.
            csv_writer.writerow(headers_to_store[1:])

            total_number_of_images = len(images_names_list)
            spinner_states = ['|', '/', '-', '\\']
            it = 0
            for image_name_aux in images_names_list:
                image_path = '%s/%s'%(input_dataset_dir, image_name_aux)
                # First of all, we need to check if the current image name has a
                # code associated.
                current_image_code_idx = np.where(associations_rows[:, 0]==image_name_aux)
                if (np.any(current_image_code_idx)):
                    # This variable stores the code that corresponds to the current
                    # image file name.
                    current_image_code = associations_rows[current_image_code_idx[0][0], 1]
                    # This variable specifies the index of the current code at the
                    # clinical data table.
                    current_code_idx = np.where(clinical_data_rows[:, 0]==current_image_code)
                    # We also need to check if the image code has a filename
                    # associated.
                    if (np.any(current_code_idx)):
                        # This is the row of the clinical data table that belongs
                        # to the current image code.
                        current_clinical_row = clinical_data_rows[current_code_idx[0][0], :]
                        if (current_clinical_row[7]=='0'):
                            input_image = deep_features_model.read_image(image_path)
                            features_array = deep_features_model.extract_deep_features(input_image)
                            row_to_write = features_array + [current_clinical_row[5]]
                            csv_writer.writerow(row_to_write)
                progress = int((it/total_number_of_images)*100)
                print('%s Progress = %d%s '%(spinner_states[it%4], progress, '%'), end='\r')
                it+=1

            print('++++ The new built dataset has been stored at %s'%output_path)

class Build_Dataset_Only_Hospitalized_Only_Less_65(Build_Dataset_Only_Hospitalized, \
                                                  Super_Class_Build_Dataset):

    def __init__(self, **kwargs):
        pass

    def build_dataset(self, input_filename, headers_file, \
                            padding_for_missing_values=-1, discretize=False):
        headers_to_store, dataset_rows = \
            super().build_dataset(input_filename, headers_file, \
                                                     padding_for_missing_values)
        headers_to_store = np.array(headers_to_store)[1:]
        filtered_dataset_rows = np.array(dataset_rows)[np.where(dataset_rows[:, 0]=='0'), 1:]

        return headers_to_store.tolist(), filtered_dataset_rows[0].tolist()

class Build_Dataset_Only_Hospitalized_Joint_Inmunosupression(Build_Dataset_Only_Hospitalized, \
                                                  Super_Class_Build_Dataset):

    def __init__(self, **kwargs):
        pass

    def __check_inmunosupression__(self, dataset_rows, inmunosupression_attrs_indexes):
        tmp_array = np.array(dataset_rows)[:, inmunosupression_attrs_indexes].astype(int)
        inmunosupressed_list = []

        for row_aux in tmp_array:
            inmunosupressed_aux = 0
            for column_aux in row_aux:
                if (column_aux==1):
                    inmunosupressed_aux = 1
                    break
            inmunosupressed_list.append(inmunosupressed_aux)

        return inmunosupressed_list

    def build_dataset(self, input_filename, headers_file, \
                            padding_for_missing_values=-1, discretize=False):
        headers_to_store, dataset_rows = \
            super().build_dataset(input_filename, headers_file, padding_for_missing_values)

        inmunosupression_attrs_indexes = list(range(11, 19))
        inmunosupressed_list = \
            np.array(self.__check_inmunosupression__(dataset_rows, \
                                              inmunosupression_attrs_indexes))
        output_attr_index = np.shape(dataset_rows)[1]-1
        # This variable will store the column with the inmunosupression
        # conditions merged and will remove the splitted inmunosupression
        # independent conditions columns.
        pieces_to_join = (dataset_rows[:, :11], dataset_rows[:, 19:output_attr_index], \
            np.expand_dims(inmunosupressed_list, axis=1), \
                    np.expand_dims(dataset_rows[:, output_attr_index], axis=1))
        rows_joint_inmunosupression = \
            np.concatenate(pieces_to_join, axis=1)

        # The same as in the previous case but, in this occasion, for the
        # headers.
        headers_pieces_to_join = (headers_to_store[:11], headers_to_store[19:output_attr_index], \
            ['inmunosupression'], [headers_to_store[output_attr_index]])

        headers_joint_inmunosupression = np.concatenate(headers_pieces_to_join)

        return headers_joint_inmunosupression.tolist(), rows_joint_inmunosupression

class Build_Dataset_Only_Hospitalized_Only_Clinical_Data(Build_Dataset_Only_Hospitalized, \
                                                                   Super_Class_Build_Dataset):

    def __init__(self, **kwargs):
        pass

    def build_dataset(self, input_filename, headers_file, \
                            padding_for_missing_values=-1, discretize=False):
        headers_to_store, dataset_rows = \
            super().build_dataset(input_filename, headers_file, padding_for_missing_values)
        headers_to_store = np.array(headers_to_store)[[0, 1, 2, 19, 20, 22, 25, 28]].tolist()
        dataset_rows = np.array(dataset_rows)[:, [0, 1, 2, 19, 20, 22, 25, 28]].tolist()

        filtered_dataset_rows = []
        for row_aux in dataset_rows:
            missing_values = 0
            for column_aux in row_aux:
                if (str(float(column_aux))=='-1.0'):
                    missing_values+=1
            if (missing_values==0):
                filtered_dataset_rows.append(row_aux)

        return headers_to_store, filtered_dataset_rows

class Build_Dataset_Only_Urgencies(Super_Class_Build_Dataset):

    def __init__(self, **kwargs):
        pass

    def build_dataset(self, input_filename, headers_file, \
                            padding_for_missing_values=-1, discretize=False):
        headers, file_data = read_csv_file(input_filename)
        list_with_fields_converted = \
            np.array(convert_fields(file_data, padding_for_missing_values))

        indexes = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, \
                      23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 5]
        hospitalized_cases_indexes = np.where(np.array(file_data)[:, indexes[0]]=='Urgencias')
        dataset_rows = list_with_fields_converted[:, indexes]
        dataset_rows = np.array(dataset_rows).astype(str)

        # Using only the "Hospitalized" patients to build the dataset.
        dataset_rows = dataset_rows[hospitalized_cases_indexes, :]

        headers = np.array(read_headers_file(headers_file))
        # As exitus is selected as the output, the word 'exitus' will be replaced
        # with output to store it like this in the CSV file.
        change_exitus_to_output = lambda input: input.replace('exitus', 'output')
        headers_to_store = list(map(change_exitus_to_output, headers[indexes]))

        # Removing duplicated data.
        dataset_rows = np.unique(dataset_rows[0], axis=0)

        # Remove the cohort, because it is not necessary in this particular case.
        headers_to_store = headers_to_store[1:]
        dataset_rows = dataset_rows[:, 1:]

        # As some of the attributes are previously removed, the indexes change and
        # therefore they must be specified again.
        new_indexes_to_check = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, \
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        dataset_rows = remove_useless_data(dataset_rows, \
                              new_indexes_to_check, padding_for_missing_values)

        return headers_to_store, dataset_rows

class Build_Dataset_Only_Hospitalized_With_Urgency_Time(Super_Class_Build_Dataset):

    def __init__(self, **kwargs):
        pass

    # This function obtains the urgency time for each patient.
    def __compute_urgency_time(self, file_data, attrs_indexes):
        # This list considers all except the first 5 fields to obtain only the
        # unique rows.
        file_data_reduced = file_data[:, attrs_indexes]
        unique_rows = np.unique(file_data_reduced, axis=0)

        # This list will store the rows grouped by the same patient (that is assumed by
        # the fact that all their clinical information cells have the same values).
        grouped_rows_data = []
        it = 0
        for row_aux in file_data_reduced:
            current_index = unique_rows.tolist().index(row_aux.tolist())

            grouped_rows_data.append([str(current_index)] + file_data[it, :].tolist())
            it+=1

        grouped_rows_data = np.array(grouped_rows_data)

        file_data_with_urgency_time = []

        # As we will add an attribute to the grouped_rows_data, we need to
        # shift one place all the indexes.
        shift_attr_indexes = lambda input: input+1
        grouped_rows_shifted_indexes = list(map(shift_attr_indexes, attrs_indexes))
        print('++++ To build this dataset, all the rows without exitus will be removed.')
        for it in range(0, len(unique_rows)):
            current_index = grouped_rows_data[:, 0].tolist().index(str(it))
            primera_fecha_seguimiento_str = grouped_rows_data[current_index, 2]
            fecha_exitus_str = grouped_rows_data[current_index, 7]

            if (len(fecha_exitus_str)!=0):
                primera_fecha_seguimiento = datetime.strptime(primera_fecha_seguimiento_str, '%d/%m/%Y')
                fecha_exitus = datetime.strptime(fecha_exitus_str, '%d/%m/%Y')
                file_data_with_urgency_time.append(grouped_rows_data[current_index, grouped_rows_shifted_indexes].tolist() + [str((fecha_exitus-primera_fecha_seguimiento).days)])
                # file_data_with_urgency_time.append(grouped_rows_data[current_index, grouped_rows_shifted_indexes].tolist() + \
                #                 [fecha_exitus, primera_fecha_seguimiento, (fecha_exitus-primera_fecha_seguimiento).days])

        return file_data_with_urgency_time

    def build_dataset(self, input_filename, headers_file, \
                            padding_for_missing_values=-1, discretize=False):
        headers, file_data = read_csv_file(input_filename)
        list_with_fields_converted = \
            np.array(convert_fields(file_data, padding_for_missing_values))

        attrs_indexes = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, \
                      23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]

        hosp_patients_indexes = np.where(np.array(file_data)[:, 7]=='Hospitalizados')
        filtered_list_with_fields_converted = \
            list_with_fields_converted[hosp_patients_indexes[0].tolist(), :]

        file_data_with_urgency_time = self.__compute_urgency_time \
            (np.array(list_with_fields_converted)[hosp_patients_indexes[0], :], attrs_indexes)

        headers_to_store = np.array(headers)[attrs_indexes].tolist() + ['output']
        dataset_rows = file_data_with_urgency_time

        return headers_to_store, dataset_rows

class Build_Dataset_Only_Hospitalized_With_Urgency_Time_Without_Weird_Rows(Build_Dataset_Only_Hospitalized_With_Urgency_Time):

        def __init__(self, **kwargs):
            pass

        def build_dataset(self, input_filename, headers_file, \
                            padding_for_missing_values=-1, discretize=False):
            headers_to_store, dataset_rows = \
                    super().build_dataset(input_filename, headers_file, \
                                        padding_for_missing_values, discretize)

            # These lines will remove all those rows with an extremely big
            # urgency time.
            filtered_dataset_rows = []
            for row_aux in dataset_rows:
                row_size = len(row_aux)
                current_urgency_time = int(row_aux[row_size-1])
                if (current_urgency_time<=90):
                    filtered_dataset_rows.append(row_aux)

            return headers_to_store, filtered_dataset_rows


class Build_Dataset_Only_Hospitalized_With_Discretized_Urgency_Time(Build_Dataset_Only_Hospitalized_With_Urgency_Time, \
                                                    Super_Class_Build_Dataset):

        def __init__(self, **kwargs):
            pass

        def __discretize_urgency_time__(self, dataset_rows):
            nofattributes = np.shape(dataset_rows)[1]
            dataset_rows = np.array(dataset_rows)

            it = 0
            for row_aux in dataset_rows:
                urgency_time = int(row_aux[nofattributes-1])
                if (urgency_time<=100):
                    output_value = 0
                else:
                    output_value = 1
                dataset_rows[it, nofattributes-1] = output_value
                it+=1

            return dataset_rows.tolist()

        def build_dataset(self, input_filename, headers_file, \
                            padding_for_missing_values=-1, discretize=False):
            headers_to_store, dataset_rows = \
                super().build_dataset(input_filename, headers_file, \
                    padding_for_missing_values)
            nofattributes = np.shape(dataset_rows)[1]
            dataset_rows = self.__discretize_urgency_time__(dataset_rows)

            return headers_to_store, dataset_rows

class Build_Dataset_Only_Hospitalized_Without_Weird_Rows(Build_Dataset_Only_Hospitalized, \
                                                    Super_Class_Build_Dataset):

    def __init__(self, **kwargs):
        pass

    def __get_indexes_without_weird_rows__(self, dataset_rows):
        indexes_list = []

        with open('./datasets/list_without_weird_rows.cfg', 'rb') as output_file:
            codes_without_weird_rows = pickle.load(output_file)

        for code_aux in codes_without_weird_rows:
            indexes_list.append(np.array(dataset_rows)[:, 0].tolist().index(code_aux))

        return indexes_list

    def build_dataset(self, input_filename, headers_file, \
                            padding_for_missing_values=-1, discretize=False):
        headers, file_data = read_csv_file(input_filename)
        list_with_fields_converted = \
            np.array(convert_fields(file_data, padding_for_missing_values, discretize))

        # Remove all the 'weird' rows (i.e., the rows with a very big urgency
        # time).
        indexes_without_weird_rows = self.__get_indexes_without_weird_rows__(file_data)
        file_data = np.array(file_data)[indexes_without_weird_rows, :].tolist()

        indexes = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, \
                      23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 5]
        hospitalized_cases_indexes = np.where(np.array(file_data)[:, indexes[0]]=='Hospitalizados')
        dataset_rows = list_with_fields_converted[:, indexes]
        dataset_rows = np.array(dataset_rows).astype(str)

        # Using only the "Hospitalized" patients to build the dataset.
        dataset_rows = dataset_rows[hospitalized_cases_indexes, :]

        headers = np.array(read_headers_file(headers_file))
        # As exitus is selected as the output, the word 'exitus' will be replaced
        # with output to store it like this in the CSV file.
        change_exitus_to_output = lambda input: input.replace('exitus', 'output')
        headers_to_store = list(map(change_exitus_to_output, headers[indexes]))

        # Removing duplicated data.
        dataset_rows = np.unique(dataset_rows[0], axis=0)

        # Remove the cohort, because it is not necessary in this particular case.
        headers_to_store = headers_to_store[1:]
        dataset_rows = dataset_rows[:, 1:]

        # As some of the attributes are previously removed, the indexes change and
        # therefore they must be specified again.
        new_indexes_to_check = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, \
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        dataset_rows = remove_useless_data(dataset_rows, \
                              new_indexes_to_check, padding_for_missing_values)

        return headers_to_store, dataset_rows
