import argparse
import numpy as np
from utils import read_csv_file, read_headers_file

def analyze_data(nofrows, nofcolumns, formatted_headers, file_data):
    fields = []

    # Initialize the lists of fields.
    for column_aux in range(0, nofcolumns):
        fields.append([])

    for row_aux in file_data:
        column_number = 0
        for column_aux in row_aux:
            fields[column_number].append(column_aux)
            column_number+=1

    fields = list(map(np.unique, fields))
    nofuniquevalues = list(map(len, fields))

    # Dictionary of unique values for each field.
    unique_values = {}
    # Dictionary of unique values count for each field.
    uniqueness_values_count = {}
    it = 0
    for item in nofuniquevalues:
        unique_values[formatted_headers[it]] = fields[it]
        uniqueness_values_count[formatted_headers[it]] = item
        it+=1

    return unique_values, uniqueness_values_count

def main():
    description = 'Program to analyze the CHUAC COVID-19 machine learning dataset.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input_filename', type=str, required=True)
    parser.add_argument('--headers_filename', type=str, required=True)
    args = parser.parse_args()

    input_filename = args.input_filename
    headers_filename = args.headers_filename
    headers, file_data = read_csv_file(input_filename)
    formatted_headers = read_headers_file(headers_filename)
    # Obtain the number of rows and the number of columns of the table.
    nofrows, nofcolumns = np.shape(file_data)
    print('+++++++ Number of rows in the table: %d'%nofrows)
    unique_values, uniqueness_values_count = analyze_data(nofrows, nofcolumns, formatted_headers, file_data)
    print('+++++++ Uniqueness values count -> ', uniqueness_values_count)
    print('+++++++ Unique values -> ', unique_values)

main()
