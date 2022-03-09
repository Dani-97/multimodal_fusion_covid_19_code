import csv
import numpy as np
from utils import read_csv_file, get_duplicates

# This function retrieves the codes list of all the patients that are included
# on the input dataset.
def get_codes_list_from_input_dataset(input_dataset):
    codes_list = []

    for row_aux in input_dataset:
        codes_list.append(row_aux[0])

    return codes_list

# This function returns the images_filenames field from the associations file
# and the list of codes that refer to each patient.
def get_important_fields_from_associations_data(associations_data):
    image_filenames_list = []
    codes_list = []

    for row_aux in associations_data:
        image_filenames_list.append(row_aux[0])
        codes_list.append(row_aux[1])

    return image_filenames_list, codes_list

def main():
    headers_input_dataset, input_dataset = read_csv_file('../input.csv')
    _, associations_data = read_csv_file('../associations.txt', has_headers=False)
    # List of codes that are retrieved from the associations file.
    codes_list_from_associations = []
    # List of codes that are retrieved from the input dataset.
    codes_list_from_input_dataset = []

    image_filenames_list, codes_list_from_associations =  \
             get_important_fields_from_associations_data(associations_data)
    codes_list_from_input_dataset = \
             get_codes_list_from_input_dataset(input_dataset)

    repeated_images_filenames = get_duplicates(image_filenames_list)
    repeated_codes = get_duplicates(codes_list_from_associations)

    # This variable will contain the list of codes that appear in both sides.
    input_dataset_conv_to_set = set(codes_list_from_input_dataset)
    associations_data_conv_to_set = set(codes_list_from_associations)
    valid_codes = list(input_dataset_conv_to_set & associations_data_conv_to_set)

    print('++++ Number of rows in associations file = ', len(codes_list_from_associations))
    print('++++ Repeated images filenames -> ', repeated_images_filenames)
    print('++++ Repeated codes (%d in total) -> '%len(repeated_codes), repeated_codes)
    print('++++ Number of codes that appear in both sides = %d'%(len(valid_codes)))
    print('++++ List of codes that appear in both sides = ', valid_codes)

main()
