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

# This computes the missing values of a list understanding that a missing value is
# an empty string.
def compute_number_of_missing_values_empty_str(input_values):
    missing_values = 0

    for value_aux in input_values:
        if (value_aux==''):
            missing_values+=1

    return missing_values

# This function returns a list of elements that appear more than once. The
# input must be a numpy array.
def get_duplicates(input_list):
    input_list1_aux = input_list
    duplicates_list = []

    it = 0
    list_len = len(input_list)
    for item_aux in input_list1_aux:
        # tmp_input_list will remove the current item of the input_list.
        tmp_input_list = np.delete(input_list1_aux, it)
        if (item_aux in tmp_input_list):
            duplicates_list.append(item_aux)
        it+=1

    return np.unique(duplicates_list)

# This function returns the string "missing_values" in case the input string
# is an empty string. Otherwise, it simply returns the same value as the input.
def detect_missing_values(input_string):
    if (input_string==''):
        output_string = 'missing_values'
    else:
        output_string = input_string

    return output_string

# This function converts an empty string to an integer (-1, to be more
# precise). Otherwise, if the string can be actually converted to an integer,
# it converts the string to its correspondent integer.
# PRECONDITION: this function assumes that the input is a string that can be
# converted to an integer or an empty string.
def convert_emtpy_str_to_integer(input_string):
    if (input_string==''):
        output_converted_value = -1
    else:
        output_converted_value = int(input_string)

    return output_converted_value

# This function converts an empty string to a float (-1.0, to be more
# precise). Otherwise, if the string can be actually converted to an float,
# it converts the string to its correspondent float.
# PRECONDITION: this function assumes that the input is a string that can be
# converted to a float or an empty string.
def convert_emtpy_str_to_float(input_string):
    if (input_string==''):
        output_converted_value = -1.0
    else:
        output_converted_value = float(input_string)

    return output_converted_value

def convert_dict_to_list(input_dict):
    values_list = []

    for key_aux, value_aux in input_dict.items():
        values_list.append(value_aux)

    return values_list

# If has_headers is True, the first row will be retrievd as the row of headers.
# Otherwise, the headers will be returned as None and, therefore, the whole
# file will be returned as the body of the CSV.
def read_csv_file(input_filename, has_headers=True):
    with open(input_filename, 'r') as csv_file:
        line_number = 0
        csv_reader = csv.reader(csv_file, delimiter=';')

        headers = None
        file_data = []
        for row in csv_reader:
            if (has_headers):
                if (line_number==0):
                    headers = row
                else:
                    file_data.append(row)
            else:
                file_data.append(row)

            line_number+=1

    return headers, file_data

def read_headers_file(headers_filename):
    with open(headers_filename) as input_file:
        lines = input_file.readlines()

    formatted_headers = eval(lines[0].strip())

    return formatted_headers
