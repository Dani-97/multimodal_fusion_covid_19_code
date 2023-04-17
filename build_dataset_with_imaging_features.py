import argparse
from dataset.utils_build_datasets import *
from dataset.utils_datasets import *
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
      choices=['VGG_16_Model_Only_Hospitalized', 'VGG_16_Model_Hospitalized_And_Urgencies', \
               'DeiT_Model_Only_Hospitalized', 'DeiT_Model_Hospitalized_And_Urgencies', \
               'DPN_Model_Only_Hospitalized', 'DPN_Model_Hospitalized_And_Urgencies'], \
                                          help='This specifies the selected approach')
    ''' The commented version includes the options to obtain imaging features from the mixed vision
    transformer and the dpn. Useful to uncomment if necessary. '''
    # parser.add_argument('--approach', type=str, required=True, \
    #   choices=['Mixed_Vision_Transformer_Only_Hospitalized', 'Mixed_Vision_Transformer_Hospitalized_And_Urgencies', \
    #            'DPN_Model_Only_Hospitalized', 'DPN_Model_Hospitalized_And_Urgencies', \
    #            'VGG_16_Model_Only_Hospitalized', 'VGG_16_Model_Hospitalized_And_Urgencies'], \
    #                                       help='This specifies the selected approach')
    parser.add_argument('--output_path', type=str, help='Path where the built dataset will be stored', \
                                              required=True)
    parser.add_argument("--layer", type=str, help="Name of the layer to select from the model of deep features", \
                         choices=['layer_fc6', 'layer_fc7', 'layer_fc8', 'all'], required=True)
    parser.add_argument("--device", help="Select CPU or GPU", required=True, \
                        choices=['CPU', 'CUDA'])
    parser.add_argument("--text_reports_embeds_method", type=str, choices=['No', 'LaBSE'], required=True, \
                        help='Method to obtain the embeddings of the text reports. Select No in case only imaging features must be ' + \
                        'obtained')
    args = parser.parse_args()

    universal_factory = UniversalFactory()

    kwargs = {}
    selected_approach = universal_factory.create_object(globals(), \
                                      'Build_Dataset_' + args.approach, kwargs)
    dataset_headers, dataset_rows = \
        selected_approach.build_dataset_with_imaging_data(args.headers_file, \
            args.input_dataset_path, args.masks_dataset_path, args.input_table_file, \
                args.associations_file, args.output_path, args.text_reports_embeds_method, \
                    device = args.device, layer=args.layer)

    selected_approach.check_dataset_statistics()
    selected_approach.store_dataset_in_csv_file(dataset_rows, args.output_path)

main()
