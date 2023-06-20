import os
from utils_scripts_train import execute_train

def obtain_params_for_experiment(noffeatures_list, experiment_name, classifier_model, balancing):
    experiments_to_execute_list = []
    for noffeatures_aux in noffeatures_list:
        logs_file_path = '../results_scenario_I/%s_%s_%s_features_%s.csv'%(experiment_name,
                     classifier_model, noffeatures_aux, balancing)
        current_exp_params = \
            {'logs_file_path': logs_file_path,
             'experiment_name': experiment_name + '_' + balancing,
             'model': classifier_model,
             'dataset_path': '../bspc_built_dataset/%s.csv'%(experiment_name),
             'preprocessing': 'Normalization',
             'manual_seeds': '0 1 2 3 4',
             'imputation': 'No_Imputation_Model',
             'balancing': balancing,
             'csv_path_with_attrs_types': '../original_dataset/attrs_headers_types.csv',
             'feature_retrieval': 'MutualInformation',
             'store_features_selection_report': 'store_true',
             'splitting': 'Holdout',
             'noftopfeatures': '%s'%noffeatures_aux,
             'nofsplits': '5',
             'test_size': '0.2', 
             'n_neighbors': 5}
        experiments_to_execute_list.append(current_exp_params)

    return experiments_to_execute_list

def main():
    os.chdir('../')

    experiments_to_execute_list = []

    noffeatures_list = ['20', '28']
    experiment_name = 'experiment_II'
    balancing = 'Oversampling'

    # classifier_model = 'SVM_Classifier'

    # experiments_to_execute_list+=obtain_params_for_experiment(noffeatures_list, cohorts, deep_features_scenario, classifier_model, balancing)

    classifier_model = 'XGBoost_Classifier'

    experiments_to_execute_list+=obtain_params_for_experiment(noffeatures_list, experiment_name, classifier_model, balancing)

    # classifier_model = 'DT_Classifier'

    # experiments_to_execute_list+=obtain_params_for_experiment(noffeatures_list, cohorts, deep_features_scenario, classifier_model, balancing)

    # classifier_model = 'kNN_Classifier'

    # experiments_to_execute_list+=obtain_params_for_experiment(noffeatures_list, cohorts, deep_features_scenario, classifier_model, balancing)

    # classifier_model = 'MLP_Classifier'

    # experiments_to_execute_list+=obtain_params_for_experiment(noffeatures_list, cohorts, deep_features_scenario, classifier_model, balancing)

    execute_train(experiments_to_execute_list)

main()
