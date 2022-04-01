import argparse
from analysis.utils_analysis_particular_attrs import *
import numpy as np
import pandas as pd

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
    parser.add_argument('--list_of_attrs', type=str, nargs='+', required=True, \
                            help='List of attributes to analyze')
    parser.add_argument('--dir_to_store_analysis', type=str, required=True, \
                            help='Directory where the analysis files will be stored')

    args = parser.parse_args()

    universal_factory = UniversalFactory()

    # Creating the splitting object with the universal factory.
    kwargs = {}
    analysis_object = universal_factory.create_object(globals(), 'Analysis_Covariances', kwargs)

    input_dataframe = pd.read_csv(args.input_file)
    # print('++++ The correlations study ' + \
    #                     'will be stored at %s'%args.dir_to_store_analysis)
    analysis_object.execute_analysis(input_dataframe, args.dir_to_store_analysis, args.list_of_attrs)

main()
