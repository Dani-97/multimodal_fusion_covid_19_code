import csv
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import torch
import torchvision

# If has_headers is True, the first row will be retrievd as the row of headers.
# Otherwise, the headers will be returned as None and, therefore, the whole
# file will be returned as the body of the CSV.
def read_csv_file(input_filename, has_headers=True):
    with open(input_filename, 'r') as csv_file:
        line_number = 0
        csv_reader = csv.reader(csv_file, delimiter=',')

        headers = None
        file_data = []
        for row in csv_reader:
            if (has_headers):
                if (line_number==0):
                    headers = row
                else:
                    file_data.append(row)
            else:
                file_data.append(row)

            line_number+=1

    return headers, file_data

# This is the parent of every splitting classes.
class Super_Splitting_Class():

    def __init__(self):
        pass

    # This function assumes that the dataset_path refers to a CSV file. In the
    # same way, it assumes that the last column of the file refers to the
    # target output and that the first row of the file has the headers.
    def load_dataset(self, dataset_path):
        headers, file_data = read_csv_file(dataset_path)
        file_data = np.array(file_data)

        file_data_shape = np.shape(file_data)
        input_data = file_data[:, 0:file_data_shape[1]-1]
        output_data = file_data[:, file_data_shape[1]-1]

        return headers, input_data, output_data

    # This function checks the type of a single attribute.
    def __check_attrs_type_aux__(self, attr_values_list):
        attr_type = 'categorical'

        for value_aux in attr_values_list:
            try:
                int(value_aux)
            except:
                attr_type = 'numerical'
                break

        return attr_type

    # This function checks which attributes are categorical and which ones not.
    def check_attrs_type(self, input_data):
        nofattrs = np.shape(input_data)[1]
        attrs_types_list = []

        for current_attr_idx in range(0, nofattrs):
            current_attr_type_aux = self.__check_attrs_type_aux__(input_data[1:, current_attr_idx])
            attrs_types_list.append(current_attr_type_aux)

    def set_partition(self, nofpartition):
        pass

    def split(self, input_data, output_data, seed=-1):
        pass

class Holdout_Split(Super_Splitting_Class):

    def __init__(self, **kwargs):
        self.test_size = float(kwargs['test_size'])
        print('++++ The dataset will be splitted in a Holdout fashion')
        print('---- Test size is %.2f. Therefore, train size is %.2f'%(self.test_size, 1.0-self.test_size))

    # The method set_partition will not do anything for this splitting method.

    # input_data variable refers to the data that will be used as input of
    # the classifier and output refers to the labels.
    def split(self, input_data, output_data, seed=-1):
        if (seed==-1):
            self.seed = None
            print('++++ No seed was chosen for the random splitting of the dataset.')
        else:
            self.seed = seed
            print('++++ The dataset will be splitted considering the manual seed %d.'%self.seed)

        splitting_subsets = train_test_split(input_data, output_data, \
                                   test_size=self.test_size, shuffle=True, random_state=self.seed)

        return splitting_subsets

    # The rest of the methods are inherited from the parent class.

class Balanced_Holdout_Split(Super_Splitting_Class):

    def __init__(self, **kwargs):
        self.test_size = float(kwargs['test_size'])
        print('++++ The dataset will be splitted in a Balanced Holdout fashion')
        print('---- Test size is %.2f. Therefore, train size is %.2f'%(self.test_size, 1.0-self.test_size))

    # The method set_partition will not do anything for this splitting method.

    # input_data variable refers to the data that will be used as input of
    # the classifier and output refers to the labels.
    def split(self, input_data, output_data, seed=-1):
        if (seed==-1):
            self.seed = None
            print('++++ No seed was chosen for the random splitting of the dataset.')
        else:
            self.seed = seed
            print('++++ The dataset will be splitted considering the manual seed %d.'%self.seed)

        negative_samples = np.where(output_data.astype(float).astype(int)==0)[0]
        nof_negative_samples = len(negative_samples)
        positive_samples = np.where(output_data.astype(float).astype(int)==1)[0]
        nof_positive_samples = len(positive_samples)

        nof_samples_per_class_dist = np.array([nof_negative_samples, nof_positive_samples])
        minority_class = np.argmin(nof_samples_per_class_dist)
        min_nof_class_samples = nof_samples_per_class_dist[minority_class]

        balanced_positive_samples_input_data = input_data[positive_samples[:min_nof_class_samples], :]
        excessive_positive_samples_input_data = input_data[positive_samples[min_nof_class_samples:], :]
        balanced_positive_samples_output_data = output_data[positive_samples[:min_nof_class_samples]]
        excessive_positive_samples_output_data = output_data[positive_samples[min_nof_class_samples:]]

        balanced_negative_samples_input_data = input_data[negative_samples[:min_nof_class_samples], :]
        excessive_negative_samples_input_data = input_data[negative_samples[min_nof_class_samples:], :]
        balanced_negative_samples_output_data = output_data[negative_samples[:min_nof_class_samples]]
        excessive_negative_samples_output_data = output_data[negative_samples[min_nof_class_samples:]]

        balanced_input_data = np.concatenate([balanced_positive_samples_input_data, balanced_negative_samples_input_data])
        balanced_output_data = np.concatenate([balanced_positive_samples_output_data, balanced_negative_samples_output_data])

        splitting_subsets = train_test_split(balanced_input_data, balanced_output_data, \
                                test_size=self.test_size, shuffle=True, random_state=self.seed, stratify=balanced_output_data)
        input_train_subset, input_test_subset, output_train_subset, output_test_subset = splitting_subsets

        input_train_subset = np.concatenate([input_train_subset, excessive_positive_samples_input_data, excessive_negative_samples_input_data])
        output_train_subset = np.concatenate([output_train_subset, excessive_positive_samples_output_data, excessive_negative_samples_output_data])

        splitting_subsets = input_train_subset, input_test_subset, output_train_subset, output_test_subset

        return splitting_subsets

    # The rest of the methods are inherited from the parent class.
