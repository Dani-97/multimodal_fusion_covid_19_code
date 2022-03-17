import argparse
from deep_features.utils_architectures import *
from datasets.utils_datasets import *
import torch
import torchvision

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
    parser.add_argument("--model", help="Name of the autoencoder architecture to use", \
                        choices=['VGG_16', 'AlexNet'], required=True)
    parser.add_argument("--device", help="Select CPU or GPU", required=True, \
                        choices=['CPU', 'CUDA'])
    args = parser.parse_args()

    universal_factory = UniversalFactory()

    # The optimizer is hardcoded at the moment.
    model_chosen = args.model + '_Deep_Features_Model'
    kwargs = {'device': args.device}
    model = universal_factory.create_object(globals(), model_chosen, kwargs)

    input_dataset_loader = model.load_dataset(args.input_dataset_path)
    # Move data to CUDA if the user selected this option.
    model.extract_deep_features(input_dataset_loader)

main()
