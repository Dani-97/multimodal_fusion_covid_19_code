import csv
import numpy as np

# This function converts 'Si' to 1 and
# 'No' to 0. Otherwise, it returns -1.
def convert_si_no_to_int(input):
    output = -1

    if (input=='Si'):
        output = 1
    elif (input=='No'):
        output = 0

    return output

def read_csv_file(input_filename):
    with open(input_filename, 'r') as csv_file:
        line_number = 0
        csv_reader = csv.reader(csv_file, delimiter=';')

        file_data = []
        for row in csv_reader:
            if (line_number==0):
                headers = row
            else:
                file_data.append(row)

            line_number+=1

    return headers, file_data

def read_headers_file(headers_filename):
    with open(headers_filename) as input_file:
        lines = input_file.readlines()

    formatted_headers = eval(lines[0].strip())

    return formatted_headers
