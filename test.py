import argparse
from main_train_test_code import Test_Class

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
    parser.add_argument("--experiment_name", help="Representative name of the experiment (for example, only_hospitalized or hospitalized_and_urgencies)", required=True)
    parser.add_argument("--model", help="Choose the model (SVM, kNN or whatever, that can be a classifier or a regressor)", \
                            choices=['SVM_Classifier', 'kNN_Classifier', 'DT_Classifier', 'MLP_Classifier', 'XGBoost_Classifier', \
                                     'SVM_Regressor', 'Linear_Regressor', 'DT_Regressor'], required=True)
    parser.add_argument("--model_path", help="The path to the model that with 'rep_0', 'rep_1', 'rep_2'...", type=str, required=True)
    parser.add_argument("--dataset_path", help="Path where the dataset is stored", required=True)
    parser.add_argument("--preprocessing", help="The preprocessing method that is desired to be selected", \
                            choices=['No', 'Standardization', 'Normalization'], required=True)
    parser.add_argument("--manual_seeds", type=int, nargs='+', \
                            help="If specified, the dataset splitting will be done considering these seeds.")
    parser.add_argument("--balancing", help="This decides the kind of dataset balancing to use", required=True, \
                                              choices=['No', 'Oversampling', 'SMOTE', 'Gaussian_Copula'])
    parser.add_argument("--imputation", help="This decides the kind of approach to use for the imputation of missing values", required=True,
                                              choices=['No_Imputation_Model', 'kNN_Values_Imputation_Model'])
    parser.add_argument("--csv_path_with_attrs_types", help="This determines the path to a CSV file that specifies if a variable is categorical or continuous.", required=True, \
                                              type=str)
    parser.add_argument("--feature_retrieval", help="Selected algorithm for feature selection or extraction. Choose 'No' to avoid feature retrieval", required=True, \
                                              choices=['No', 'PCA', 'VarianceThreshold', 'Fisher', 'MutualInformation'])
    parser.add_argument("--store_features_selection_report", help="If this option is selected, then the features selection report will be stored to the logging results file", \
                                              action='store_true')
    parser.add_argument("--splitting", help="Choose the kind of dataset splitting method to use", \
                                              choices=['Holdout', 'Balanced_Holdout'], required=True)
    parser.add_argument("--noftopfeatures", help="Number of top features to select from the ranking that was obtained with the feature selection algorithm", type=str)
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
    if ((args.manual_seeds is not None) and (len(args.manual_seeds)!=args.nofsplits)):
        print('++++ ERROR: if specified, the number of manual seeds must be the same as --nofsplits')
        exit(-1)

    # We check if the variable can be converted to an integer.
    # If not, then we check that the value is 'all'. Otherwise, the code will end in these lines.
    try:
        int(args.noftopfeatures)
    except ValueError:
        if (args.noftopfeatures!='all'):
            print("++++ ERROR: the number of top features must be a number or the string 'all', but '%s' received!"%args.noftopfeatures)
            exit(-1)

    test_class_obj = Test_Class()
    test_class_obj.execute_approach(args)

main()
