import os
from pathlib import Path
from utils_scripts_train import execute_train, execute_test

def obtain_params_for_experiment(noffeatures_list, cohorts, deep_features_scenario, classifier_model, balancing, operation_point):
    feature_retrieval = 'MutualInformation'
    experiments_to_execute_list = []
    for noffeatures_aux in noffeatures_list:
        operation_point_str = str(operation_point)
        operation_point_str = operation_point_str.replace('.', '_')
        logs_file_path = '../results_scenario_III/%s_%s_%s_features_%s_%s_op_%s.csv'%(cohorts,
                     classifier_model, noffeatures_aux, deep_features_scenario, balancing, operation_point_str)
        model_path = str(Path(logs_file_path).parent) + '/' + \
                     cohorts + '_' + \
                     deep_features_scenario + '_' + \
                     balancing + '_' + \
                     classifier_model + '_' + \
                     str(noffeatures_aux) + '_' + \
                     feature_retrieval + '_' + \
                     balancing + '_' + \
                     'model'
        current_exp_params = \
            {'logs_file_path': logs_file_path,
             'experiment_name': cohorts + '_' + deep_features_scenario + '_' + balancing,
             'model': classifier_model,
             'model_path': model_path,
             'dataset_path': '../built_dataset/%s_%s.csv'%(cohorts, deep_features_scenario),
             'preprocessing': 'Normalization',
             'manual_seeds': '0 1 2 3 4',
             'imputation': 'No_Imputation_Model',
             'balancing': balancing,
             'csv_path_with_attrs_types': '../original_dataset/attrs_headers_types.csv',
             'feature_retrieval': feature_retrieval,
             'store_features_selection_report': 'store_true',
             'splitting': 'Holdout',
             'noftopfeatures': '%s'%noffeatures_aux,
             'nofsplits': '5',
             'test_size': '0.2',
             'operation_point': operation_point,
             'n_neighbors': 5}
        experiments_to_execute_list.append(current_exp_params)

    return experiments_to_execute_list

def main():
    os.chdir('../')

    experiments_to_execute_list = []

    noffeatures_list = ['20', '40', '60', '80', '100', '120', '140', '160', '180', '200', '400', '600', '800', '1000', 'all']
    cohorts = 'only_hospitalized'
    deep_features_scenario = 'vgg_16_fc8'
    balancing = 'Oversampling'
    classifier_model = 'XGBoost_Classifier'
    
    operation_point = 0.4
    experiments_to_execute_list+=obtain_params_for_experiment(noffeatures_list, cohorts, deep_features_scenario, \
                                                              classifier_model, balancing, operation_point)

    operation_point = 0.3
    experiments_to_execute_list+=obtain_params_for_experiment(noffeatures_list, cohorts, deep_features_scenario, \
                                                              classifier_model, balancing, operation_point)

    operation_point = 0.2
    experiments_to_execute_list+=obtain_params_for_experiment(noffeatures_list, cohorts, deep_features_scenario, \
                                                              classifier_model, balancing, operation_point)

    execute_test(experiments_to_execute_list)

main()
