import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from datasets.utils_build_datasets import *

class UniversalFactory():

    def __init__(self):
        pass

    def create_object(self, namespace, classname, kwargs):
        ClassName = namespace[classname]
        universal_obj = ClassName(**kwargs)
        return universal_obj

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
                 'Only_Hospitalized_Joint_Inmunosupression', 'Only_Hospitalized_Only_Less_65'], \
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
