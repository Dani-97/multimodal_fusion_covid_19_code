from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# This is the parent of every splitting classes.
class Super_Splitting_Class():

    def __init__(self):
        pass

    def load_dataset(self, dataset_path):
        print('DEBUGGING: WARNING! "load_dataset" from SuperSplittingClass is still in testing phase.')
        print('This implementation is currently being tested with IRIS dataset.')

        irisData = load_iris()

        # Create feature and target arrays
        input_data = irisData.data
        output_data = irisData.target

        return input_data, output_data

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
