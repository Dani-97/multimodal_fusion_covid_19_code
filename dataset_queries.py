import argparse
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Program to make queries on built datasets.')
    parser.add_argument('--input_file', type=str, required=True, \
                            help='Path to the input file where we want to make the query')
    parser.add_argument('--fields', type=str, nargs='+', \
                            help='Specifies the fields that are going to be shown. If None, ' + \
                            'all fields are returned. This parameter is ignored when --get_count' + \
                            'is specified')
    parser.add_argument('--query', type=str, required=True, \
                            help='Defines the query that we want to make to the dataset.' + \
                            'If the query is an empty string, then the entire dataset is shown')
    parser.add_argument('--get_count', action='store_true', \
                            help='If specified, then it only returns the number of rows retrieved')

    args = parser.parse_args()

    input_dataframe = pd.read_csv(args.input_file)

    if (args.query.strip()!=''):
        query_result = input_dataframe.query(args.query)
    else:
        query_result = input_dataframe

    if (args.get_count):
        print('++++ This query has retrieved %d rows'%len(query_result))
    else:
        if (args.fields!=None):
            result_to_show = query_result[args.fields]
        else:
            result_to_show = query_result

        print(result_to_show)

main()
