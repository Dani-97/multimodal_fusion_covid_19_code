import argparse
from datetime import datetime
import numpy as np
from utils import read_csv_file, convert_si_no_to_int

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
                    row_list_aux.append(field)
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
                    row_list_aux.append(field)
                # rango_edad
                if (column_number_aux==8):
                    row_list_aux.append(field)
                # sexo
                if (column_number_aux==9):
                    row_list_aux.append(field)
                # edad
                if (column_number_aux==10):
                    row_list_aux.append(int(field))
                # talla_cm
                if (column_number_aux==11):
                    row_list_aux.append(int(field))
                # peso_kg
                if (column_number_aux==12):
                    row_list_aux.append(double(field.replace(',', '.')))
                # imc
                if (column_number_aux==13):
                    row_list_aux.append(double(field.replace(',', '.')))
                # hta
                if (column_number_aux==14):
                    row_list_aux.append(field)
                # diabetes_mellitus
                if (column_number_aux==15):
                    row_list_aux.append(field)
                # epoc
                if (column_number_aux==16):
                    row_list_aux.append(field)
                # asma
                if (column_number_aux==17):
                    row_list_aux.append(field)
                # hepatopatia
                if (column_number_aux==18):
                    row_list_aux.append(field)
                # leucemia
                if (column_number_aux==19):
                    row_list_aux.append(field)
                # linfoma
                if (column_number_aux==20):
                    row_list_aux.append(field)
                # neoplasia
                if (column_number_aux==21):
                    row_list_aux.append(field)
                # hiv
                if (column_number_aux==22):
                    row_list_aux.append(field)
                # transplante_organo_solido
                if (column_number_aux==23):
                    row_list_aux.append(field)
                # quimioterapia_ultimos_3_meses
                if (column_number_aux==24):
                    row_list_aux.append(field)
                # biologicos_ultimos_3_meses
                if (column_number_aux==25):
                    row_list_aux.append(field)
                # biologicos_cuales
                if (column_number_aux==26):
                    row_list_aux.append(field)
                # corticoides_cronicos_mas_3_meses
                if (column_number_aux==27):
                    row_list_aux.append(field)
                # linfocitos
                if (column_number_aux==28):
                    row_list_aux.append(double(field.replace(',', '.')))
                # linfocitos_porcentaje
                if (column_number_aux==29):
                    row_list_aux.append(double(field.replace(',', '.')))
                # dimeros_d
                if (column_number_aux==30):
                    row_list_aux.append(int(field))
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
                    row_list_aux.append(double(field.replace(',', '.')))
                # ferritina
                if (column_number_aux==35):
                    row_list_aux.append(int(field))
                # il6
                if (column_number_aux==36):
                    row_list_aux.append(field)
            except:
                row_list_aux.append(field)

            column_number_aux+=1

        output_list.append(row_list_aux)

    return output_list

def build_dataset(input_filename):
    headers, file_data = read_csv_file(input_filename)
    list_with_fields_converted = convert_fields(file_data)
    print(list_with_fields_converted)

def main():
    description = 'Program to build a dataset suitable for classifiers from the \
                       CHUAC COVID-19 machine learning dataset.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input_filename', type=str, required=True)
    args = parser.parse_args()

    input_filename = args.input_filename
    build_dataset(input_filename)

main()
