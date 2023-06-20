import os
from utils_scripts_train import execute_train

def obtain_params_for_experiment(noffeatures_list, cohorts, deep_features_scenario, classifier_model, balancing):
    experiments_to_execute_list = []
    for noffeatures_aux in noffeatures_list:
        logs_file_path = '../results_scenario_III/%s_%s_%s_features_%s_%s.csv'%(cohorts,
                     classifier_model, noffeatures_aux, deep_features_scenario, balancing)
        current_exp_params = \
            {'logs_file_path': logs_file_path,
             'experiment_name': cohorts + '_' + deep_features_scenario + '_' + balancing,
             'model': classifier_model,
             'dataset_path': '../built_dataset/%s_%s.csv'%(cohorts, deep_features_scenario),
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

    # noffeatures_list = ['20', '40', '60', '80', '100', '120', '140', '160', '180', '200', '400', '600', '800', '1000', 'all']
    noffeatures_list = ['220', '240', '260', '280', '300', '320', '340', '360', '380', '420', '440', '460', '480', '500', '520', '540', '560', '580', '620', '640', '660', '680', '700', '720', '740', '760', '780', '820', '840', '860', '880', '900', '920', '940', '960', '980', '1020', '1040', '1060', '1080', '1100', '1120', '1140', '1160', '1180', '1200', '1220', '1240', '1260', '1280', '1300', '1320', '1340', '1360', '1380', '1400', '1420', '1440', '1460', '1480', '1500', '1520', '1540', '1560', '1580', '1600', '1620', '1640', '1660', '1680', '1700', '1720', '1740', '1760', '1780', '1800', '1820', '1840', '1860', '1880', '1900', '1920', '1940', '1960', '1980', '2000', '2020', '2040', '2060', '2080', '2100', '2120', '2140', '2160', '2180', '2200', '2220', '2240', '2260', '2280', '2300', '2320', '2340', '2360', '2380', '2400', '2420', '2440', '2460', '2480', '2500', '2520', '2540', '2560', '2580', '2600', '2620', '2640', '2660', '2680', '2700', '2720', '2740', '2760', '2780', '2800', '2820', '2840', '2860', '2880', '2900', '2920', '2940', '2960', '2980', '3000', '3020', '3040', '3060', '3080', '3100', '3120', '3140', '3160', '3180', '3200', '3220', '3240', '3260', '3280', '3300', '3320', '3340', '3360', '3380', '3400', '3420', '3440', '3460', '3480', '3500', '3520', '3540', '3560', '3580', '3600', '3620', '3640', '3660', '3680', '3700', '3720', '3740', '3760', '3780', '3800', '3820', '3840', '3860', '3880', '3900', '3920', '3940', '3960', '3980', 'all']
    cohorts = 'hospitalized_and_urgencies'
    deep_features_scenario = 'vgg_16_fc6'
    balancing = 'Oversampling'

    #classifier_model = 'SVM_Classifier'

    #experiments_to_execute_list+=obtain_params_for_experiment(noffeatures_list, cohorts, deep_features_scenario, classifier_model, balancing)

    classifier_model = 'XGBoost_Classifier'

    experiments_to_execute_list+=obtain_params_for_experiment(noffeatures_list, cohorts, deep_features_scenario, classifier_model, balancing)

    #classifier_model = 'DT_Classifier'

    #experiments_to_execute_list+=obtain_params_for_experiment(noffeatures_list, cohorts, deep_features_scenario, classifier_model, balancing)

    #classifier_model = 'kNN_Classifier'

    #experiments_to_execute_list+=obtain_params_for_experiment(noffeatures_list, cohorts, deep_features_scenario, classifier_model, balancing)

    #classifier_model = 'MLP_Classifier'

    #experiments_to_execute_list+=obtain_params_for_experiment(noffeatures_list, cohorts, deep_features_scenario, classifier_model, balancing)

    execute_train(experiments_to_execute_list)

main()
