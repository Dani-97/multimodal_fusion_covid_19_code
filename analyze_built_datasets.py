import argparse
from analysis.utils_analysis_built_datasets import *

class UniversalFactory():

    def __init__(self):
        pass

    def create_object(self, namespace, classname, kwargs):
        ClassName = namespace[classname]
        universal_obj = ClassName(**kwargs)
        return universal_obj

def main():
    parser = argparse.ArgumentParser(description='Program to make queries on built datasets.')
    parser.add_argument('--input_file', type=str, required=True, \
                            help='Path to the input file where we want to make the query')
    parser.add_argument('--built_dataset', type=str, required=True, \
                            choices=['Only_Hospitalized'], \
                            help='Descriptive name of the dataset as is used in this project')

    args = parser.parse_args()

    universal_factory = UniversalFactory()

    # Creating the splitting object with the universal factory.
    kwargs = {}
    analysis_object = universal_factory.create_object(globals(), 'Analysis_' + args.built_dataset, kwargs)

    input_dataframe = pd.read_csv(args.input_file)
    analysis_object.execute_analysis(input_dataframe)

main()
