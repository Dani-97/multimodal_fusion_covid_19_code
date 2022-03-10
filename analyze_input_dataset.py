import argparse
from analysis.utils_analysis import *
import configparser
import numpy as np
from utils import read_csv_file, read_headers_file, get_duplicates

'''
    IMPORTANT NOTE: the file ./analysis/ad_hoc_items.cfg has some important
    parameters. Modify it to adapt it to your desired behavior.
    Another warning is that this code is ad-hoc to the specific problem we are
    working with. Then, if you change the problem, it could be useful to start
    a new script from zero.
'''

class UniversalFactory():

    def __init__(self):
        pass

    def create_object(self, namespace, classname, kwargs):
        ClassName = namespace[classname]
        universal_obj = ClassName(**kwargs)
        return universal_obj

def uniqueness_analysis(nofrows, nofcolumns, formatted_headers, file_data):
    fields = []

    # Initialize the lists of fields.
    for column_aux in range(0, nofcolumns):
        fields.append([])

    for row_aux in file_data:
        column_number = 0
        for column_aux in row_aux:
            fields[column_number].append(column_aux)
            column_number+=1

    fields = list(map(np.unique, fields))
    nofuniquevalues = list(map(len, fields))

    # Dictionary of unique values for each field.
    unique_values = {}
    # Dictionary of unique values count for each field.
    uniqueness_values_count = {}
    it = 0
    for item in nofuniquevalues:
        unique_values[formatted_headers[it]] = fields[it]
        uniqueness_values_count[formatted_headers[it]] = item
        it+=1

    return unique_values, uniqueness_values_count

# This function works with the original_dataset.
def obtain_histograms_from_original_dataset(file_data):
    # This variable is a list where each item is another list that contains all
    # the values of a certain attribute. This is the appropriate format to plot
    # a histogram.
    file_data_for_histogram = []

    file_data = np.array(file_data)
    nofrows, nofcolumns = np.shape(file_data)

    for column_aux in range(0, nofcolumns):
        attr_data_tmp = []
        for row_aux in range(0, nofrows):
            attr_data_tmp.append(file_data[row_aux, column_aux])
        file_data_for_histogram.append(attr_data_tmp)

    '''
        IMPORTANT NOTE: the list of the desired attributes to be analyzed,
        the titles of the plots and the bins of the histograms are ad-hoc, having
        an important dependency with the input dataset. Therefore, it was decided
        to write those variables in an individual txt file. This file is read,
        and its content is evaluated.
    '''

    ad_hoc_variables = configparser.ConfigParser()
    ad_hoc_variables.read('./analysis/ad_hoc_items_original_dataset.cfg')

    attrs_list = eval(ad_hoc_variables['VARIABLES']['attrs_list'])
    titles_list = eval(ad_hoc_variables['VARIABLES']['titles_list'])
    nofbins_list = eval(ad_hoc_variables['VARIABLES']['nofbins_list'])

    universal_factory = UniversalFactory()

    it = 0
    for nofattr_aux in attrs_list:
        # Creating the splitting object with the universal factory.
        kwargs = {'nofattribute': nofattr_aux, 'title': titles_list[it], 'nofbins': nofbins_list[it]}
        obtain_histogram_obj = universal_factory.create_object(globals(), 'Obtain_Histogram_Attr_%d'%nofattr_aux, kwargs)

        output_filename = '../histograms/attr_%d_histogram.pdf'%(nofattr_aux)
        obtain_histogram_obj.plot_histogram(file_data_for_histogram, output_filename)
        it+=1

# This function works with the reduced version of the dataset, where we deleted
# some of the attributes.
def obtain_histograms_from_reduced_dataset(file_data):
    # This variable is a list where each item is another list that contains all
    # the values of a certain attribute. This is the appropriate format to plot
    # a histogram.
    file_data_for_histogram = []

    file_data = np.array(file_data)
    nofrows, nofcolumns = np.shape(file_data)

    for column_aux in range(0, nofcolumns):
        attr_data_tmp = []
        for row_aux in range(0, nofrows):
            attr_data_tmp.append(file_data[row_aux, column_aux])
        file_data_for_histogram.append(attr_data_tmp)

    '''
        IMPORTANT NOTE: the list of the desired attributes to be analyzed,
        the titles of the plots and the bins of the histograms are ad-hoc, having
        an important dependency with the input dataset. Therefore, it was decided
        to write those variables in an individual txt file. This file is read,
        and its content is evaluated.
    '''

    ad_hoc_variables = configparser.ConfigParser()
    ad_hoc_variables.read('./analysis/ad_hoc_items_reduced_dataset.cfg')

    attrs_list = eval(ad_hoc_variables['VARIABLES']['attrs_list'])
    titles_list = eval(ad_hoc_variables['VARIABLES']['titles_list'])
    nofbins_list = eval(ad_hoc_variables['VARIABLES']['nofbins_list'])

    universal_factory = UniversalFactory()

    it = 0
    for nofattr_aux in attrs_list:
        # Creating the splitting object with the universal factory.
        kwargs = {'nofattribute': nofattr_aux, 'title': titles_list[it], 'nofbins': nofbins_list[it]}
        obtain_histogram_obj = universal_factory.create_object(globals(), 'Obtain_Histogram_Attr_%d'%nofattr_aux, kwargs)

        output_filename = '../histograms/attr_%d_histogram.pdf'%(nofattr_aux)
        obtain_histogram_obj.plot_histogram(file_data_for_histogram, output_filename)
        it+=1

# This function will detect implicitly duplicated patients in order to obtain
# truly unique patients rows.
def detect_unique_patients(input_data):
    input_data = np.array(input_data)

    reduced_data = input_data[:, [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, \
                        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, \
                            28, 29, 30, 31, 32, 33, 34, 35, 36]]

    return np.unique(reduced_data, axis=0)

def main():
    description = 'Program to analyze the CHUAC COVID-19 machine learning dataset.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input_filename', type=str, required=True)
    parser.add_argument('--headers_filename', type=str, required=True)
    args = parser.parse_args()

    input_filename = args.input_filename
    headers_filename = args.headers_filename
    headers, file_data = read_csv_file(input_filename)
    formatted_headers = read_headers_file(headers_filename)
    # Obtain the number of rows and the number of columns of the table.
    nofrows, nofcolumns = np.shape(file_data)

    print('+++++++ Number of rows in the table: %d'%nofrows)

    unique_patients_data = detect_unique_patients(file_data)
    print('+++++++ Number of unique patients: %d'%len(unique_patients_data))
    print('+++++++ Number of repeated patients: %d'%(len(file_data)-len(unique_patients_data)))

    unique_values, uniqueness_values_count = uniqueness_analysis(nofrows, nofcolumns, formatted_headers, file_data)
    print('+++++++ Uniqueness values count -> ', uniqueness_values_count)

    # obtain_histograms_from_original_dataset(file_data)
    obtain_histograms_from_reduced_dataset(unique_patients_data)

main()
