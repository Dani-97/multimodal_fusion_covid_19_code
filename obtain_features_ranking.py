import argparse
from dataset.utils_datasets import *
from dataset.utils_features import *

class UniversalFactory():

    def __init__(self):
        pass

    def create_object(self, namespace, classname, kwargs):
        ClassName = namespace[classname]
        universal_obj = ClassName(**kwargs)
        return universal_obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_retrieval", \
        choices=['VarianceThreshold', 'Fisher', 'MutualInformation'], \
        required=True, help="Name of the algorithm used to make feature selection")
    parser.add_argument("--dataset_path", type=str, \
                              help="Path to the input dataset", required=True)
    parser.add_argument("--dir_to_store_results", type=str, \
                              help="Directory where the result will be stored")
    parser.add_argument("--csv_file_name", type=str, required=True, \
            help="CSV filename where the report will be stored. " + \
                           "This filename is appended to dir_to_store_results")
    parser.add_argument("--noftopfeatures", type=str, required=True, \
            help="Number of top features that we want the algorithm to select")
    parser.add_argument('--translations_csv_path', type=str, \
        help='Path to the CSV file with the translations of the names of the headers.')

    args = parser.parse_args()

    universal_factory = UniversalFactory()

    # Creating the splitting object with the universal factory.
    # In this case, the splitting approach does not matter, as we only need to
    # load the dataset. Then, we just create an object of the class
    # Super_Split.
    splitting = Super_Splitting_Class()
    # Retrieving the feature selection method with the universal factory.
    kwargs = {'noftopfeatures': args.noftopfeatures}
    feature_retrieval = universal_factory.create_object(globals(), args.feature_retrieval + '_Feature_Retrieval', kwargs)

    attrs_headers, input_data, output_data = splitting.load_dataset(args.dataset_path)

    feature_retrieval.execute_feature_retrieval(input_data, output_data)
    feature_retrieval.set_dir_to_store_results(args.dir_to_store_results)
    csv_file_path = args.dir_to_store_results + '/' + args.csv_file_name
    attrs_headers = feature_retrieval.translate_headers(attrs_headers, args.translations_csv_path)
    feature_retrieval.store_report(csv_file_path, attrs_headers, False)

main()
