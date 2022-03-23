import argparse
from deep_features.utils_architectures import *
from datasets.utils_build_datasets import *
from datasets.utils_datasets import *
import numpy as np
import torch
import torchvision
from utils import *

class UniversalFactory():

    def __init__(self):
        pass

    def create_object(self, namespace, classname, kwargs):
        ClassName = namespace[classname]
        universal_obj = ClassName(**kwargs)
        return universal_obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_path", help="Path of the dataset", required=True)
    parser.add_argument("--headers_file", help="Path to the file with the headers of the fields", required=True)
    parser.add_argument("--input_table_file", help="Path of the input table CSV with clinical data", \
                                            required=True)
    parser.add_argument("--associations_file", required=True, \
        help="Path to the file with the associations between an image and its code in the input table")
    parser.add_argument('--approach', type=str, required=True, \
      choices=['Only_Hospitalized', 'Only_Hospitalized_Only_Deep_Features'], \
                                          help='This specifies the selected approach')
    parser.add_argument('--output_path', type=str, help='Path where the built dataset will be stored', \
                                              required=True)
    parser.add_argument("--model", help="Name of the autoencoder architecture to use", \
                        choices=['VGG_16', 'AlexNet'], required=True)
    parser.add_argument("--device", help="Select CPU or GPU", required=True, \
                        choices=['CPU', 'CUDA'])
    args = parser.parse_args()

    universal_factory = UniversalFactory()

    kwargs = {}
    selected_approach = universal_factory.create_object(globals(), \
                                      'Build_Dataset_' + args.approach, kwargs)
    selected_approach.build_dataset_with_deep_features(args.model, \
        args.headers_file, args.input_dataset_path, args.input_table_file, \
            args.associations_file, args.output_path, device = args.device)

main()
