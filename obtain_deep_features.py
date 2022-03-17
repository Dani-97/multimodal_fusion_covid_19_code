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
    parser.add_argument("--splitting", choices=['Holdout'], \
        help="Name of the criterion to split in training and test", required=True)
    parser.add_argument("--test_size", type=float, \
        help="Size of the test set when using Holdout")
    args = parser.parse_args()

    if ((args.splitting=='Holdout') and (args.test_size==None)):
        print('++++ If you use holdout, you must to specify --test_size')
        exit(-1)

    universal_factory = UniversalFactory()

    kwargs = {'test_size': args.test_size}
    splitting_chosen = args.splitting + '_Split_Pytorch'
    splitting = universal_factory.create_object(globals(), splitting_chosen, kwargs)

    input_dataset = splitting.load_dataset(args.input_dataset_path)
    train_loader, test_loader = splitting.split(input_dataset)

    # The optimizer is hardcoded at the moment.
    model_chosen = args.model + '_Deep_Features_Model'
    kwargs = {'device': args.device}
    model = universal_factory.create_object(globals(), model_chosen, kwargs)

    # Move data to CUDA if the user selected this option.
    model.extract_deep_features(train_loader)

main()
