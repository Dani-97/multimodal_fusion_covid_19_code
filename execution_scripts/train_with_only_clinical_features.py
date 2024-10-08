import os
from utils_scripts_train import execute_train

def obtain_params_for_experiment(noffeatures_list, cohorts, classifier_model, balancing):
    experiments_to_execute_list = []
    for noffeatures_aux in noffeatures_list:
        logs_file_path = '../heliyon_results/results_scenario_I/%s_%s_%s_features_%s.csv'%(cohorts,
                     classifier_model, noffeatures_aux, balancing)
        current_exp_params = \
            {'logs_file_path': logs_file_path,
             'experiment_name': cohorts + '_' + balancing,
             'model': classifier_model,
             'dataset_path': '../built_dataset/%s_only_clinical_data.csv'%(cohorts),
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

    noffeatures_list = ['20', 'all']
    cohorts_list = ['hospitalized_and_urgencies', 'only_hospitalized']
    balancing = 'Oversampling'

    classifier_model = 'XGBoost_Classifier'

    for current_cohort_aux in cohorts_list:
        experiments_to_execute_list+=obtain_params_for_experiment(noffeatures_list, current_cohort_aux, classifier_model, balancing)

    execute_train(experiments_to_execute_list)

main()
