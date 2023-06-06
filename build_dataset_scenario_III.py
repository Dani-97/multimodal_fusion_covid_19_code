import argparse
from dataset.utils_build_datasets import Build_Dataset_VGG_16_Model_Only_Hospitalized
from dataset.utils_build_datasets import Build_Dataset_VGG_16_Model_Hospitalized_And_Urgencies

class UniversalFactory():

    def __init__(self):
        pass

    def create_object(self, namespace, classname, kwargs):
        ClassName = namespace[classname]
        universal_obj = ClassName(**kwargs)
        return universal_obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir_root', help="Path of the directory with the input images", required=True)
    parser.add_argument('--input_csv_file_path', help="Path of the input table CSV with clinical data", required=True)
    parser.add_argument('--approach', type=str, help='This specifies the selected approach', required=True, \
                choices=['VGG_16_Model_Only_Hospitalized', 'VGG_16_Model_Hospitalized_And_Urgencies'])
    parser.add_argument('--imaging_features_csv_path', type=str, \
                help='If specified, the features of the images will be retrieved from here instead of being calculated.')
    parser.add_argument('--output_path', type=str, help='Path where the built dataset will be stored', required=True)
    parser.add_argument('--layer', type=str, help="Name of the layer to select from the model of deep features", \
                choices=['layer_fc6', 'layer_fc7', 'layer_fc8', 'all'], required=True)
    parser.add_argument('--dataset_version', type=str, choices=['full', 'simplified'], required=True, \
                            help='Select if you want all the variables or you want to remove ' + \
                                'the patients ids and drop the duplicates.')
    parser.add_argument('--device', help="Select CPU or GPU", choices=['CPU', 'CUDA'], required=True)
    args = parser.parse_args()

    universal_factory = UniversalFactory()

    kwargs = {'images_dir_root': args.images_dir_root, 
              'input_csv_file_path': args.input_csv_file_path,
              'imaging_features_csv_path': args.imaging_features_csv_path,
              'device': args.device,
              'layer': args.layer}
    
    selected_approach = universal_factory.create_object(globals(), 'Build_Dataset_' + args.approach, kwargs)
    func_to_execute = 'build_' + args.dataset_version + '_dataset_scenario_III'
    print('++++ Calling to function %s...'%func_to_execute)
    _, dataset_rows = getattr(selected_approach, func_to_execute)()
    selected_approach.store_dataset_in_csv_file(dataset_rows, args.output_path)

main()
