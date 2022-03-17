import csv
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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

class Holdout_Split(Super_Splitting_Class):

    def __init__(self, **kwargs):
        self.test_size = float(kwargs['test_size'])
        print('++++ The dataset will be splitted in a Holdout fashion')
        print('---- Test size is %.2f. Therefore, train size is %.2f'%(self.test_size, 1.0-self.test_size))

    def load_dataset(self, dataset_path):
        return super().load_dataset(dataset_path)

    # input_data variable refers to the data that will be used as input of
    # the classifier and output refers to the labels.
    def split(self, input_data, output_data):
        splitting_subsets = train_test_split(input_data, output_data, test_size=self.test_size)

        return splitting_subsets

class Holdout_Split_Pytorch(Super_Splitting_Class):

    def __init__(self, **kwargs):
        self.test_size = float(kwargs['test_size'])
        print('++++ The dataset will be splitted in a Holdout fashion')
        print('---- Test size is %.2f. Therefore, train size is %.2f'%(self.test_size, 1.0-self.test_size))

    def load_dataset(self, dataset_path):
        print('++++ Loading the images of the path %s'%dataset_path)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), \
                                                    torchvision.transforms.Resize([768, 768])])

        input_dataset = \
            torchvision.datasets.ImageFolder(dataset_path, transform=transform)

        return input_dataset

    def split(self, input_dataset):
        # This variable specifies the number of samples of the input dataset.
        len_of_dataset = len(input_dataset)

        nof_test_samples = int(self.test_size*len_of_dataset)
        nof_training_samples = len_of_dataset - nof_test_samples
        train_dataset, test_dataset = \
                       torch.utils.data.random_split(input_dataset, \
                                   [nof_training_samples, nof_test_samples])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

        return train_loader, test_loader
