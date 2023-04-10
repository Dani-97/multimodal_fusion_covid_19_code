import argparse
import matplotlib.pyplot as plt
import numpy as np
from dataset.utils_build_datasets import *

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
        choices=['Hospitalized_And_Urgencies', 'Only_Hospitalized'], \
                 help='This specifies the selected approach')
    parser.add_argument('--output_path', type=str, required=True,
                            help='Path where the CSV files of the dataset will be stored')
    args = parser.parse_args()

    universal_factory = UniversalFactory()

    kwargs = {}
    selected_approach = universal_factory.create_object(globals(), \
                                      'Build_Dataset_' + args.approach, kwargs)

    input_filename = args.input_filename
    dataset_headers, dataset_rows = \
        selected_approach.build_dataset(input_filename, args.headers_file)
    selected_approach.check_dataset_statistics()
    selected_approach.store_dataset_in_csv_file(dataset_rows, args.output_path)

main()
