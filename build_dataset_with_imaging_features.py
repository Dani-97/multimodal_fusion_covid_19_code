import argparse
from datasets.utils_build_datasets import *
from datasets.utils_datasets import *
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
    parser.add_argument("--masks_dataset_path", help="Path to the mask images, if it is necessary to use them")
    parser.add_argument("--headers_file", help="Path to the file with the headers of the fields", required=True)
    parser.add_argument("--input_table_file", help="Path of the input table CSV with clinical data", \
                                            required=True)
    parser.add_argument("--associations_file", required=True, \
        help="Path to the file with the associations between an image and its code in the input table")
    parser.add_argument('--approach', type=str, required=True, \
      choices=['Debugging_Radiomics_Features'], \
                                          help='This specifies the selected approach')
    parser.add_argument('--output_path', type=str, help='Path where the built dataset will be stored', \
                                              required=True)
    parser.add_argument("--image_feature_extraction_approach", help="Name of the autoencoder architecture to use if deep features were chosen", \
                        choices=['Radiomics_Features', 'VGG_16_Deep_Features_Model', 'AlexNet_Deep_Features_Model'])
    parser.add_argument("--device", help="Select CPU or GPU", required=True, \
                        choices=['CPU', 'CUDA'])
    args = parser.parse_args()

    universal_factory = UniversalFactory()

    kwargs = {}
    selected_approach = universal_factory.create_object(globals(), \
                                      'Build_Dataset_' + args.approach, kwargs)
    dataset_headers, dataset_rows = \
        selected_approach.build_dataset_with_imaging_data(args.headers_file, \
            args.input_dataset_path, args.masks_dataset_path, args.input_table_file, \
                args.associations_file, args.output_path, device = args.device)
    selected_approach.check_dataset_statistics()
    selected_approach.store_dataset_in_csv_file(dataset_rows, args.output_path)

main()
