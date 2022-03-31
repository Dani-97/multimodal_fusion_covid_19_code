import argparse
from classifiers.utils_classifiers import *
from regressors.utils_regressors import *
from datasets.utils_balancing import *
from datasets.utils_datasets import *
from datasets.utils_features import *
from datasets.utils_normalization import *
import numpy as np
import os
from utils import convert_metrics_dict_to_list, clear_csv_file

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
    parser.add_argument("--model", help="Choose the model (SVM, kNN or whatever, that can be a classifier or a regressor)", \
                            choices=['SVM_Classifier', 'kNN_Classifier', 'DT_Classifier', 'MLP_Classifier', 'XGBoost_Classifier', \
                                     'SVM_Regressor', 'Linear_Regressor', 'DT_Regressor'], required=True)
    parser.add_argument("--dataset_path", help="Path where the dataset is stored", required=True)
    parser.add_argument("--preprocessing", help="The preprocessing method that is desired to be selected", \
                            choices=['No', 'Standardization', 'Normalization'], required=True)
    parser.add_argument("--balancing", help="This decides the kind of dataset balancing to use", required=True, \
                                              choices=['No', 'Oversampling', 'Undersampling'])
    parser.add_argument("--feature_retrieval", help="Selected algorithm for feature selection or extraction. Choose 'No' to avoid feature retrieval", required=True, \
                                              choices=['No', 'PCA', 'ReliefF'])
    parser.add_argument("--store_features_selection_report", help="If this option is selected, then the features selection report will be stored to the logging results file", \
                                              action='store_true')
    parser.add_argument("--splitting", help="Choose the kind of dataset splitting method to use", \
                                              choices=['Holdout', 'Cross_Validation'], required=True)
    parser.add_argument("--noftopfeatures", help="Number of top features to select in the case of using SelectKBest feature selection algorithm", type=int)
    parser.add_argument("--nofcomponents", help="Number of components to be extracted with the PCA algorithm", type=int)
    parser.add_argument('--nofsplits', help="Number of different holdouts to be performed or number of folds of the cross validation", type=int, required=True)
    parser.add_argument("--n_neighbors", help="Number of neighbors in case of training a kNN classifier", type=int)
    parser.add_argument("--test_size", help="Size of the the test subset (in percentage) in case of using Holdout", type=float)
    parser.add_argument("--plot_data", action='store_true', \
                           help="This argument indicates that we want to plot the data if possible (in case it is 2D or 3D)")
    args = parser.parse_args()

    if ((args.model=='kNN') and (args.n_neighbors is None)):
        print('++++ ERROR: if you choose the kNN algorithm, you need to specify the --n_neighbors argument')
        exit(-1)
    if ((args.feature_retrieval=='PCA') and (args.nofcomponents is None)):
        print('++++ ERROR: if you choose the PCA algorithm, you need to specify the --nofcomponents argument')
        exit(-1)
    if ((args.feature_retrieval=='RBFSampler') and (args.nofcomponents is None)):
        print('++++ ERROR: if you choose RBFSampler algorithm, you need to specify the --nofcomponents argument')
        exit(-1)
    if ((args.model.find('Regressor')!=-1) and (args.balancing!='No')):
        print('++++ ERROR: if you choose a regression model, you cannot use any kind of balancing')
        exit(-1)

    universal_factory = UniversalFactory()

    # Creating the splitting object with the universal factory.
    kwargs = {'test_size': args.test_size, 'noffolds': args.nofsplits}
    splitting = universal_factory.create_object(globals(), args.splitting + '_Split', kwargs)
    # Retrieving the feature selection method with the universal factory.
    kwargs = {'noftopfeatures': args.noftopfeatures, 'nofcomponents': args.nofcomponents}
    feature_retrieval = universal_factory.create_object(globals(), args.feature_retrieval + '_Feature_Retrieval', kwargs)

    attrs_headers, input_data, output_data = splitting.load_dataset(args.dataset_path)

    # These lines execute the preprocessing step in case one was selected.
    kwargs = {}
    preprocessing = universal_factory.create_object(globals(), args.preprocessing + '_Preprocessing', kwargs)
    input_data = preprocessing.execute_preprocessing(input_data.astype(np.float64))

    # The input_data variable is overwritten with the data obtained after the
    # feature selection (or no feature selection process).
    input_data = feature_retrieval.execute_feature_retrieval(input_data, output_data, plot_data=args.plot_data)

    # If the CSV logs file had previous content, then this function will remove
    # it.
    clear_csv_file(args.logs_file_path)
    # This function will store the report of the feature selection process if it
    # is available and if the user has decided to store it.
    if (args.store_features_selection_report):
        dir_to_store_results = os.path.dirname(args.logs_file_path)
        feature_retrieval.set_dir_to_store_results(dir_to_store_results)
        feature_retrieval.store_report(args.logs_file_path, attrs_headers)
    else:
        print('++++ As the user decided, the report of the features' + \
            ' selection algorithm will not be stored to the logging results file')

    for it in range(0, args.nofsplits):
        print('**** Starting repetition number %d...'%it)
        # For certain splitting methods, this line does not do anything
        # (for example, in the case of the holdout).
        splitting.set_partition(it)
        # Split into training and test set
        subsets = splitting.split(input_data, output_data)
        input_train_subset, input_test_subset, \
                             output_train_subset, output_test_subset = subsets

        # Performing the balancing on the dataset.
        kwargs = {}
        balancing_module = \
                universal_factory.create_object(globals(), args.balancing + '_Balancing', kwargs)
        input_train_subset, output_train_subset = \
                balancing_module.execute_balancing(input_train_subset, output_train_subset)

        # Creating the model with the universal factory.
        kwargs = {'n_neighbors': args.n_neighbors}
        model = universal_factory.create_object(globals(), args.model, kwargs)

        # Convert all the cells of the subsets to double.
        input_train_subset = input_train_subset.astype(np.float64)
        input_test_subset = input_test_subset.astype(np.float64)
        output_train_subset = output_train_subset.astype(np.float64)
        output_test_subset = output_test_subset.astype(np.float64)

        model.train(input_train_subset, output_train_subset)
        # The function returns the predicted values.
        model_output_pred = model.test(input_test_subset)
        metrics_values = model.model_metrics(model_output_pred, output_test_subset)
        headers_list, metrics_values_list = convert_metrics_dict_to_list(metrics_values)
        if (it==0):
            model.add_headers_to_csv_file(args.logs_file_path, headers_list)
        # This function will store the number of the current repetition (the value of "it")
        # and the metrics of the current repetition.
        model.store_model_metrics(it, metrics_values_list, args.logs_file_path)
        # For those models that are not decision trees, this is strictly
        # necessary to execute the SHAP algorithm.
        model.load_input_data_for_explainability(input_test_subset)
        # Here we obtain the explainability of the model if it is available.
        model.explainability()
        print('---- NOTE: the logs of this repetition are now stored at %s\n'%args.logs_file_path)

    # Lastly, the mean and the standard deviation of the metrics obtained for
    # all the repetitions are stored at the end of the CSV file.
    model.compute_mean_and_std_performance(args.logs_file_path)

main()
