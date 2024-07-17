import os
from utils_scripts_train import execute_train

def obtain_params_for_experiment(noffeatures_list, cohorts, deep_features_scenario, classifier_model, balancing):
    experiments_to_execute_list = []
    for noffeatures_aux in noffeatures_list:
        logs_file_path = '../heliyon_results/results_scenario_III/%s_%s_%s_features_%s_%s.csv'%(cohorts,
                     classifier_model, noffeatures_aux, deep_features_scenario, balancing)
        current_exp_params = \
            {'logs_file_path': logs_file_path,
             'experiment_name': cohorts + '_' + deep_features_scenario + '_' + balancing,
             'model': classifier_model,
             'dataset_path': '../built_dataset/%s_%s.csv'%(cohorts, deep_features_scenario),
             'preprocessing': 'Standardization',
             'manual_seeds': '0 1 2 3 4',
             'imputation': 'No_Imputation_Model',
             'balancing': balancing,
             'csv_path_with_attrs_types': '../original_dataset/attrs_headers_types.csv',
             'feature_retrieval': 'MutualInformation',
             'store_features_selection_report': 'store_true',
             'splitting': 'Cross_Val_And_Holdout',
             'noftopfeatures': '%s'%noffeatures_aux,
             'nof_folds': '5',
             'nofsplits': '5',
             'test_size': '0.2',
             'n_neighbors': 5}
        experiments_to_execute_list.append(current_exp_params)

    return experiments_to_execute_list

def main():
    os.chdir('../')

    experiments_to_execute_list = []

    noffeatures_list = list(map(lambda input_value: str(input_value), list(range(20, 4000, 20)))) + ['all']
    cohorts_list = ['hospitalized_and_urgencies', 'only_hospitalized']
    deep_features_scenario = ['vgg_16_fc6', 'vgg_16_fc7', 'vgg_16_fc8']
    balancing = 'Oversampling'

    classifier_model = 'XGBoost_Classifier'

    for current_cohort_aux in cohorts_list:
        for current_deep_features_scenario in deep_features_scenario:
            experiments_to_execute_list+=obtain_params_for_experiment(noffeatures_list, current_cohort_aux, current_deep_features_scenario, classifier_model, balancing)

    execute_train(experiments_to_execute_list)

main()
