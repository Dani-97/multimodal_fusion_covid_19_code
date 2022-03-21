import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from utils import read_csv_file, read_headers_file
from utils import write_csv_file, convert_si_no_to_int

class UniversalFactory():

    def __init__(self):
        pass

    def create_object(self, namespace, classname, kwargs):
        ClassName = namespace[classname]
        universal_obj = ClassName(**kwargs)
        return universal_obj

'''
    NOTE: this code is ad-hoc to the problem herein proposed. Therefore, if
    you want to use it for another different problem, it could be probably more
    useful to start another different script from scratch.
'''

def convert_cohorte(input_value):
    output_value = -1

    if (input_value=='Hospitalizados'):
        output_value = 0
    elif (input_value=='Residencias'):
        output_value = 1
    elif (input_value=='Urgencias'):
        output_value = 2

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
def convert_fields(file_data, padding_for_missing_values=-1):
    output_list = []
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
                    row_list_aux.append(convert_double_attributes(field.replace(',', '.')))
                # linfocitos_porcentaje
                if (column_number_aux==29):
                    row_list_aux.append(convert_double_attributes(field.replace(',', '.')))
                # dimeros_d
                if (column_number_aux==30):
                    row_list_aux.append(convert_integer_attributes(field))
                # ldh
                if (column_number_aux==31):
                    row_list_aux.append(convert_integer_attributes(field))
                # creatinina
                if (column_number_aux==32):
                    row_list_aux.append(convert_double_attributes(field.replace(',', '.')))
                # filtrado_glomerular_estimado
                if (column_number_aux==33):
                    row_list_aux.append(convert_double_attributes(field.replace(',', '.')))
                # prc
                if (column_number_aux==34):
                    row_list_aux.append(convert_double_attributes(field.replace(',', '.')))
                # ferritina
                if (column_number_aux==35):
                    row_list_aux.append(convert_integer_attributes(field))
                # il6
                if (column_number_aux==36):
                    row_list_aux.append(convert_double_attributes(field.replace(',', '.')))
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

class Build_Dataset_Hospitalized_And_Urgencies():

    def __init__(self, **kwargs):
        pass

    def build_dataset(self, input_filename, headers_file,
                                             padding_for_missing_values=-1):
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

class Build_Dataset_Only_Hospitalized():

    def __init__(self, **kwargs):
        pass

    def build_dataset(self, input_filename, headers_file,
                                             padding_for_missing_values=-1):
        headers, file_data = read_csv_file(input_filename)
        list_with_fields_converted = \
            np.array(convert_fields(file_data, padding_for_missing_values))

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

class Build_Dataset_Only_Hospitalized_Joint_Inmunosupression(Build_Dataset_Only_Hospitalized):

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
                                             padding_for_missing_values=-1):
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

class Build_Dataset_Only_Hospitalized_Only_Clinical_Data(Build_Dataset_Only_Hospitalized):

    def __init__(self, **kwargs):
        pass

    def build_dataset(self, input_filename, headers_file, \
                                             padding_for_missing_values=-1):
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

class Build_Dataset_Only_Urgencies():

    def __init__(self, **kwargs):
        pass

    def build_dataset(self, input_filename, headers_file,
                                             padding_for_missing_values=-1):
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

class Build_Dataset_Only_Hospitalized_With_Urgency_Time():

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

    def build_dataset(self, input_filename, headers_file,
                                             padding_for_missing_values=-1):
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

def main():
    description = 'Program to build a dataset suitable for classifiers from the \
                       CHUAC COVID-19 machine learning dataset.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input_filename', type=str, required=True, \
                            help='Name of the input CSV file')
    parser.add_argument('--headers_file', type=str, required=True, \
                            help='Path of the file where the headers are specified')
    parser.add_argument('--approach', type=str, required=True, \
        choices=['Only_Hospitalized', 'Only_Urgencies', 'Only_Hospitalized_With_Urgency_Time', \
                 'Hospitalized_And_Urgencies', 'Only_Hospitalized_Only_Clinical_Data', \
                 'Only_Hospitalized_Joint_Inmunosupression'], \
                 help='This specifies the selected approach')
    parser.add_argument('--padding_missing_values', type=int, \
                 help='It specifies the value that will be used to fill the cells with missing values.' +
                 'If not specified, then this value will be -1.')
    parser.add_argument('--populated_threshold', type=float, required=True, \
                 help='This specifies the minimum proportion of non missing values that each row must have' + \
                      'Set to 0.0 if you do not want to filter the rows.')
    parser.add_argument('--output_path', type=str, required=True,
                            help='Path where the CSV files of the dataset will be stored')
    args = parser.parse_args()

    universal_factory = UniversalFactory()

    kwargs = {}
    selected_approach = universal_factory.create_object(globals(), \
                                      'Build_Dataset_' + args.approach, kwargs)

    # If args.padding_missing_values is None, then the used padding value will
    # be -1.
    if (args.padding_missing_values==None):
        padding_for_missing_values = -1
    else:
        padding_for_missing_values = args.padding_missing_values

    print('++++ The cells with missing values will be filled with %d'%padding_for_missing_values)

    input_filename = args.input_filename
    headers_to_store, dataset_rows = \
        selected_approach.build_dataset(input_filename, args.headers_file, \
                                                   padding_for_missing_values)
    dataset_rows = filter_populated_rows(args.populated_threshold, dataset_rows)
    write_csv_file(args.output_path, headers_to_store, dataset_rows)

main()
