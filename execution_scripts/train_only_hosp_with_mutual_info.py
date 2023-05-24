import os
from utils_scripts_train import execute_train

def obtain_params_for_experiment(noffeatures_list, cohorts, deep_features_scenario, classifier_model):
    experiments_to_execute_list = []
    for noffeatures_aux in noffeatures_list:
        logs_file_path = '../results/%s_%s_%d_features_%s.csv'%(cohorts,
                     classifier_model, noffeatures_aux, deep_features_scenario)
        current_exp_params = \
            {'logs_file_path': logs_file_path,
             'experiment_name': cohorts + '_' + deep_features_scenario,
             'model': classifier_model,
             'dataset_path': '../built_dataset/%s_%s.csv'%(cohorts, deep_features_scenario),
             'preprocessing': 'Normalization',
             'manual_seeds': '0 1 2 3 4',
             'balancing': 'SMOTE',
             'csv_path_with_attrs_types': '../original_dataset/attrs_headers_types.csv',
             'feature_retrieval': 'MutualInformation',
             'store_features_selection_report': 'store_true',
             'splitting': 'Balanced_Holdout',
             'noftopfeatures': '%d'%noffeatures_aux,
             'nofsplits': '5',
             'test_size': '0.2'}
        experiments_to_execute_list.append(current_exp_params)

    return experiments_to_execute_list

def main():
    os.chdir('../')

    experiments_to_execute_list = []

    noffeatures_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    cohorts = 'only_hospitalized'
    deep_features_scenario = 'vgg_16_model_layer_fc7'
    classifier_model = 'SVM_Classifier'

    experiments_to_execute_list+=obtain_params_for_experiment(noffeatures_list, cohorts, deep_features_scenario, classifier_model)

    noffeatures_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
    cohorts = 'only_hospitalized'
    deep_features_scenario = 'vgg_16_model_layer_fc7'
    classifier_model = 'XGBoost_Classifier'

    experiments_to_execute_list+=obtain_params_for_experiment(noffeatures_list, cohorts, deep_features_scenario, classifier_model)

    execute_train(experiments_to_execute_list)

main()
