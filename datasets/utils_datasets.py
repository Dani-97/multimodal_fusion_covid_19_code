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

    def split(self, input_data, output_data):
        pass

class Holdout_Split(Super_Splitting_Class):

    def __init__(self, **kwargs):
        self.test_size = float(kwargs['test_size'])
        self.seed = kwargs['seed']
        print('++++ The dataset will be splitted in a Holdout fashion')
        print('---- Test size is %.2f. Therefore, train size is %.2f'%(self.test_size, 1.0-self.test_size))
        
        if (self.seed!=None):
            print('++++ The datset will be splitted considering the manual seed %d.'%self.seed)
        else:
            print('++++ No seed was chosen for the random splitting of the dataset.')

    # The method set_partition will not do anything for this splitting method.

    # input_data variable refers to the data that will be used as input of
    # the classifier and output refers to the labels.
    def split(self, input_data, output_data):
        splitting_subsets = train_test_split(input_data, output_data, \
                                   test_size=self.test_size, shuffle=True, random_state=self.seed)

        return splitting_subsets

    # The rest of the methods are inherited from the parent class.

class Cross_Validation_Split(Super_Splitting_Class):

    def __init__(self, **kwargs):
        self.noffolds = int(kwargs['noffolds'])
        self.seed = kwargs['seed']
        print('++++ The dataset will be splitted in a Cross Validation fashion with %d folds'%\
                self.noffolds)
        if (self.seed!=None):
            print('++++ The datset will be splitted considering the manual seed %d.'%self.seed)
        else:
            print('++++ No seed was chosen for the random splitting of the dataset.')

        self.splitting_module = KFold(n_splits=self.noffolds, shuffle=True, random_state=self.seed)
        self.partitions = None

    def set_partition(self, nofpartition):
        self.nofpartition = nofpartition

    def split(self, input_data, output_data):
        # The first time this function is called, the partitions of the input
        # dataset are made. The next times the function is called, the
        # already made set of partitions is used.
        if (self.partitions==None):
            self.partitions = []
            joint_subsets_generator = self.splitting_module.split(input_data)
            for train_subset, test_subset in joint_subsets_generator:
                self.partitions.append((train_subset, test_subset))

        current_train_folds = self.partitions[self.nofpartition][0]
        current_test_fold = self.partitions[self.nofpartition][1]

        input_train_subset = input_data[current_train_folds, :]
        output_train_subset = output_data[current_train_folds]
        input_test_subset = input_data[current_test_fold, :]
        output_test_subset = output_data[current_test_fold]

        splitting_subsets = input_train_subset, input_test_subset, \
                                 output_train_subset, output_test_subset

        return splitting_subsets
