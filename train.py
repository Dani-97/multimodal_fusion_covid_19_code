import argparse
from classifiers.utils_classifiers import SVM_Classifier, kNN_Classifier
from datasets.utils_datasets import Holdout_Split
from datasets.utils_features import No_Feature_Retrieval, SelectKBest_Feature_Retrieval, PCA_Feature_Retrieval
from utils import convert_metrics_dict_to_list

class UniversalFactory():

    def __init__(self):
        pass

    def create_object(self, namespace, classname, kwargs):
        ClassName = namespace[classname]
        universal_obj = ClassName(**kwargs)
        return universal_obj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_file_path", help="Path to the CSV file where the logs will be stored", required=True)
    parser.add_argument("--classifier", help="Choose the classifier (SVM, kNN or whatever)", \
                                              choices=['SVM', 'kNN'], required=True)
    parser.add_argument("--dataset_path", help="Path where the dataset is stored", required=True)
    parser.add_argument("--feature_retrieval", help="Selected algorithm for feature selection or extraction. Choose 'No' to avoid feature retrieval", required=True, \
                                              choices=['No', 'SelectKBest', 'PCA'])
    parser.add_argument("--splitting", help="Choose the kind of dataset splitting method to use", \
                                              choices=['Holdout'], required=True)
    parser.add_argument("--noftopfeatures", help="Number of top features to select in the case of using SelectKBest feature selection algorithm", type=int)
    parser.add_argument('--nofrepetitions', help="Number of times that the trainig process must be performed", type=int, required=True)
    parser.add_argument("--n_neighbors", help="Number of neighbors in case of training a kNN classifier", type=int)
    parser.add_argument("--test_size", help="Size of the the test subset (in percentage) in case of using Holdout", type=float)
    args = parser.parse_args()

    if ((args.classifier=='kNN') and (args.n_neighbors is None)):
        print('++++ ERROR: if you choose the kNN algorithm, you need to specify the --n_neighbors argument')
        exit(-1)

    universal_factory = UniversalFactory()

    # Creating the splitting object with the universal factory.
    kwargs = {'test_size': args.test_size}
    splitting = universal_factory.create_object(globals(), args.splitting + '_Split', kwargs)
    # Retrieving the feature selection method with the universal factory.
    kwargs = {'noftopfeatures': args.noftopfeatures}
    feature_retrieval = universal_factory.create_object(globals(), args.feature_retrieval + '_Feature_Retrieval', kwargs)

    input_data, output_data = splitting.load_dataset(args.dataset_path)
    # The input_data variable is overwritten with the data obtained after the
    # feature selection (or no feature selection process).
    input_data = feature_retrieval.execute_feature_retrieval(input_data, output_data)

    log_csv_file = args.logs_file_path
    for it in range(0, args.nofrepetitions):
        print('**** Starting repetition number %d...'%it)
        # Split into training and test set
        subsets = splitting.split(input_data, output_data)
        input_train_subset, input_test_subset, \
                             output_train_subset, output_test_subset = subsets

        # Creating the classifier with the universal factory.
        kwargs = {'n_neighbors': args.n_neighbors}
        classifier = universal_factory.create_object(globals(), args.classifier + '_Classifier', kwargs)

        classifier.train(input_train_subset, output_train_subset)
        # The function returns the predicted values (i.e., labels such as
        # 1 for positive class and 0 for negative class) and the probabilities
        # output (for example, 0.97).
        classifier_output_pred, classifier_output_prob = classifier.test(input_test_subset)
        metrics_values = classifier.classification_metrics(classifier_output_pred, classifier_output_prob, output_test_subset)
        headers_list, metrics_values_list = convert_metrics_dict_to_list(metrics_values)
        if (it==0):
            classifier.add_headers_to_csv_file(log_csv_file, headers_list)
        # This function will store the number of the current repetition (the value of it)
        # and the metrics of the current repetition.
        classifier.store_classification_metrics(it, metrics_values_list, log_csv_file)
        print('---- NOTE: the logs of this repetition are now stored at %s'%log_csv_file)

main()
