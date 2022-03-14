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
def convert_si_no_desconocido_missing_to_integer(input_value):
    output_value = -1

    if (input_value=='No'):
        output_value = 0
    elif (input_value=='Si'):
        output_value = 1

    return output_value

# This function allows to convert the fields to the desired format.
# This is an ad-hoc function.
def convert_fields(file_data):
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
                    row_list_aux.append(datetime.strptime(field, '%d/%m/%Y'))
                # hallazgos
                if (column_number_aux==2):
                    row_list_aux.append(field)
                # fecha_rx
                if (column_number_aux==3):
                    row_list_aux.append(int(field))
                # fecha_positivo
                if (column_number_aux==4):
                    row_list_aux.append(datetime.strptime(field, '%d/%m/%Y'))
                # exitus
                if (column_number_aux==5):
                    row_list_aux.append(convert_si_no_to_int(field))
                # fecha_exitus
                if (column_number_aux==6):
                    row_list_aux.append(datetime.strptime(field, '%d/%m/%Y'))
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
                    row_list_aux.append(convert_si_no_desconocido_missing_to_integer(field))
                # diabetes_mellitus
                if (column_number_aux==15):
                    row_list_aux.append(convert_si_no_desconocido_missing_to_integer(field))
                # epoc
                if (column_number_aux==16):
                    row_list_aux.append(convert_si_no_desconocido_missing_to_integer(field))
                # asma
                if (column_number_aux==17):
                    row_list_aux.append(convert_si_no_desconocido_missing_to_integer(field))
                # hepatopatia
                if (column_number_aux==18):
                    row_list_aux.append(convert_si_no_desconocido_missing_to_integer(field))
                # leucemia
                if (column_number_aux==19):
                    row_list_aux.append(convert_si_no_desconocido_missing_to_integer(field))
                # linfoma
                if (column_number_aux==20):
                    row_list_aux.append(convert_si_no_desconocido_missing_to_integer(field))
                # neoplasia
                if (column_number_aux==21):
                    row_list_aux.append(convert_si_no_desconocido_missing_to_integer(field))
                # hiv
                if (column_number_aux==22):
                    row_list_aux.append(convert_si_no_desconocido_missing_to_integer(field))
                # transplante_organo_solido
                if (column_number_aux==23):
                    row_list_aux.append(convert_si_no_desconocido_missing_to_integer(field))
                # quimioterapia_ultimos_3_meses
                if (column_number_aux==24):
                    row_list_aux.append(convert_si_no_desconocido_missing_to_integer(field))
                # biologicos_ultimos_3_meses
                if (column_number_aux==25):
                    row_list_aux.append(convert_si_no_desconocido_missing_to_integer(field))
                # biologicos_cuales
                if (column_number_aux==26):
                    row_list_aux.append(convert_si_no_desconocido_missing_to_integer(field))
                # corticoides_cronicos_mas_3_meses
                if (column_number_aux==27):
                    row_list_aux.append(convert_si_no_desconocido_missing_to_integer(field))
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
def remove_useless_data(dataset_rows, attr_columns_to_check):
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
        nof_missing_values_aux = row_aux.tolist().count('-1.0')
        if ((len(unique_values)==1) and (unique_values[0]=='-1.0')):
            nof_useless_rows+=1
        else:
            if (nof_missing_values_aux>0):
                nof_missing_values_per_row.append(nof_missing_values_aux)
            useful_rows.append(nofrow_aux)
        nofrow_aux+=1

    dataset_rows = dataset_rows[useful_rows, :]

    print('**** There were %d useless rows that have been deleted.'%nof_useless_rows)

    return dataset_rows

class Build_Dataset_Hospitalized_And_Urgencies():

    def __init__(self):
        pass

    def build_dataset(self, input_filename, headers_file):
        headers, file_data = read_csv_file(input_filename)
        list_with_fields_converted = np.array(convert_fields(file_data))

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
        dataset_rows = remove_useless_data(dataset_rows, new_indexes_to_check)

        return headers_to_store, dataset_rows

class Build_Dataset_Only_Hospitalized():

    def __init__(self):
        pass

    def build_dataset(self, input_filename, headers_file):
        headers, file_data = read_csv_file(input_filename)
        list_with_fields_converted = np.array(convert_fields(file_data))

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
        dataset_rows = remove_useless_data(dataset_rows, new_indexes_to_check)

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
                            choices=['Only_Hospitalized', 'Hospitalized_And_Urgencies'], \
                            help='This specifies the selected approach')
    parser.add_argument('--output_path', type=str, required=True,
                            help='Path where the CSV files of the dataset will be stored')
    args = parser.parse_args()

    universal_factory = UniversalFactory()

    kwargs = {}
    selected_approach = universal_factory.create_object(globals(), \
                                      'Build_Dataset_' + args.approach, kwargs)

    input_filename = args.input_filename
    headers_to_store, dataset_rows = \
                selected_approach.build_dataset(input_filename, args.headers_file)
    write_csv_file(args.output_path, headers_to_store, dataset_rows)

main()
