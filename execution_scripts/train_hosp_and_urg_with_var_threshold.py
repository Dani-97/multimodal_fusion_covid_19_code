import os
from utils_scripts_train import execute_train

os.chdir('../')

noffeatures_list = list(range(1, 29))
nofexperiments = len(noffeatures_list)
logs_dir_path_list = ['../results/hospitalized_and_urgencies/']*nofexperiments
logs_file_name_list = ['dt_%s_feature_var_thresh_oversampling_results.csv']*nofexperiments
model_list = ['DT_Classifier']*nofexperiments
dataset_path_list = ['../built_dataset/hospitalized_and_urgencies.csv']*nofexperiments
balancing_list = ['Oversampling']*nofexperiments
feature_retrieval_list = ['VarianceThreshold']*nofexperiments
splitting_list = ['Holdout']*nofexperiments
test_size_list = [0.2]*nofexperiments
nofsplits_list = [5]*nofexperiments
preprocessing_list = ['No']*nofexperiments
store_features_selection_report_list = ['--store_features_selection_report']*nofexperiments
n_neighbors_list = [5]*nofexperiments

execute_train(noffeatures_list, nofexperiments, logs_dir_path_list, \
        logs_file_name_list, model_list, dataset_path_list, balancing_list, \
            feature_retrieval_list, splitting_list, test_size_list, \
                nofsplits_list, preprocessing_list, \
                    store_features_selection_report_list)

logs_file_name_list = ['xgboost_%s_feature_var_thresh_oversampling_results.csv']*nofexperiments
model_list = ['XGBoost_Classifier']*nofexperiments

execute_train(noffeatures_list, nofexperiments, logs_dir_path_list, \
        logs_file_name_list, model_list, dataset_path_list, balancing_list, \
            feature_retrieval_list, splitting_list, test_size_list, \
                nofsplits_list, preprocessing_list, \
                    store_features_selection_report_list)

logs_file_name_list = ['kNN_%s_feature_var_thresh_oversampling_results.csv']*nofexperiments
model_list = ['kNN_Classifier']*nofexperiments

execute_train(noffeatures_list, nofexperiments, logs_dir_path_list, \
        logs_file_name_list, model_list, dataset_path_list, balancing_list, \
            feature_retrieval_list, splitting_list, test_size_list, \
                nofsplits_list, preprocessing_list, \
                    store_features_selection_report_list, n_neighbors_list)

logs_file_name_list = ['mlp_%s_feature_var_thresh_oversampling_results.csv']*nofexperiments
model_list = ['MLP_Classifier']*nofexperiments

execute_train(noffeatures_list, nofexperiments, logs_dir_path_list, \
        logs_file_name_list, model_list, dataset_path_list, balancing_list, \
            feature_retrieval_list, splitting_list, test_size_list, \
                nofsplits_list, preprocessing_list, \
                    store_features_selection_report_list)

logs_file_name_list = ['svm_%s_feature_var_thresh_oversampling_results.csv']*nofexperiments
model_list = ['SVM_Classifier']*nofexperiments

execute_train(noffeatures_list, nofexperiments, logs_dir_path_list, \
        logs_file_name_list, model_list, dataset_path_list, balancing_list, \
            feature_retrieval_list, splitting_list, test_size_list, \
                nofsplits_list, preprocessing_list, \
                    store_features_selection_report_list)
