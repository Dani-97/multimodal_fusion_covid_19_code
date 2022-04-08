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
    parser = argparse.ArgumentParser(description='Program to plot the dispersion ' + \
                     ' of the values of a certain attribute or attributes.')
    parser.add_argument('--input_file', type=str, required=True, \
                            help='Path to the input file that we want to be analyzed')
    parser.add_argument('--list_of_attrs', type=str, nargs='+', required=True, \
                            help='List of attributes to analyze')
    parser.add_argument('--dir_to_store_analysis', type=str, required=True, \
                            help='Directory where the analysis files will be stored')

    args = parser.parse_args()

    universal_factory = UniversalFactory()

    # Check the number of attributes to plot the values dispersion.
    nofattributes = len(args.list_of_attrs)
    if (not ((nofattributes>0) and (nofattributes<3))):
        print('++++ ERROR: you cannot plot the dispersion with 3 attributes or more.')
        exit(0)

    # Creating the splitting object with the universal factory.
    kwargs = {}
    analysis_object = universal_factory.create_object(globals(), \
                         'Analysis_Dispersion_%dD'%nofattributes, kwargs)

    input_dataframe = pd.read_csv(args.input_file)

    analysis_object.execute_analysis(input_dataframe, args.dir_to_store_analysis, args.list_of_attrs)

main()
