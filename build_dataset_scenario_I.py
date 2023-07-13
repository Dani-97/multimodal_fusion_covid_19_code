import argparse
import matplotlib.pyplot as plt
import numpy as np
from dataset.utils_build_datasets import Build_Dataset_Hospitalized_And_Urgencies
from dataset.utils_build_datasets import Build_Dataset_Only_Hospitalized

class UniversalFactory():

    def __init__(self):
        pass

    def create_object(self, namespace, classname, kwargs):
        ClassName = namespace[classname]
        universal_obj = ClassName(**kwargs)
        return universal_obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv_file_path', type=str, required=True, \
                            help='Name of the input CSV file')
    parser.add_argument('--approach', type=str, required=True, \
        choices=['Hospitalized_And_Urgencies', 'Only_Hospitalized'], \
                 help='This specifies the selected approach')
    parser.add_argument('--output_path', type=str, required=True,
                            help='Path where the CSV files of the dataset will be stored')
    parser.add_argument('--dataset_version', type=str, choices=['full', 'simplified'], required=True, \
                            help='Select if you want all the variables or you want to remove ' + \
                                'the patients ids and drop the duplicates.')
    parser.add_argument('--imaging_features_csv_path', type=str, \
                            help='If specified, the clinical data will be filtered. Only those ' + \
                                'patients with an imaging study will be stored to the build dataset.')
    args = parser.parse_args()

    universal_factory = UniversalFactory()

    kwargs = {'input_csv_file_path': args.input_csv_file_path, \
              'imaging_features_csv_path': args.imaging_features_csv_path}
    selected_approach = universal_factory.create_object(globals(), \
                                      'Build_Dataset_' + args.approach, kwargs)

    func_to_execute = 'build_' + args.dataset_version + '_dataset_scenario_I'
    print('++++ Calling to function %s...'%func_to_execute)
    _, dataset_rows = getattr(selected_approach, func_to_execute)()
    selected_approach.store_dataset_in_csv_file(dataset_rows, args.output_path)

main()
