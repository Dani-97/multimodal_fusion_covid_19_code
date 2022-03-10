import argparse
from datetime import datetime
import numpy as np
from utils import read_csv_file, read_headers_file
from utils import write_csv_file, convert_si_no_to_int

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
    elif (input_value=='[65, 80]'):
        output_value = 1
    elif (input_value=='[>80]'):
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
                    row_list_aux.append(field)
                # creatinina
                if (column_number_aux==32):
                    row_list_aux.append(field)
                # filtrado_glomerular_estimado
                if (column_number_aux==33):
                    row_list_aux.append(field)
                # prc
                if (column_number_aux==34):
                    row_list_aux.append(convert_double_attributes(field.replace(',', '.')))
                # ferritina
                if (column_number_aux==35):
                    row_list_aux.append(convert_integer_attributes(field))
                # il6
                if (column_number_aux==36):
                    row_list_aux.append(field)
            except:
                row_list_aux.append(field)

            column_number_aux+=1

        output_list.append(row_list_aux)

    return output_list

def build_dataset(input_filename, headers_file):
    headers, file_data = read_csv_file(input_filename)
    list_with_fields_converted = np.array(convert_fields(file_data))

    indexes = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, \
                  23, 24, 25, 27, 28, 29, 30, 5]
    dataset_rows = list_with_fields_converted[:, indexes]

    headers = np.array(read_headers_file(headers_file))
    # As exitus is selected as the output, the word 'exitus' will be replaced
    # with output to store it like this in the CSV file.
    change_exitus_to_output = lambda input: input.replace('exitus', 'output')
    headers_to_store = list(map(change_exitus_to_output, headers[indexes]))

    return headers_to_store, dataset_rows

def main():
    description = 'Program to build a dataset suitable for classifiers from the \
                       CHUAC COVID-19 machine learning dataset.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input_filename', type=str, required=True, \
                            help='Name of the input CSV file')
    parser.add_argument('--headers_file', type=str, required=True, \
                            help='Path of the file where the headers are specified')
    parser.add_argument('--output_directory', type=str, required=True,
                            help='Path where the CSV files of the dataset will be stored')
    args = parser.parse_args()

    input_filename = args.input_filename
    headers_to_store, dataset_rows = \
                         build_dataset(input_filename, args.headers_file)
    write_csv_file(args.output_directory + '/dataset_rows.csv', headers_to_store, dataset_rows)

main()
