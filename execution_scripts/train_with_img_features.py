import os

os.chdir('../')
input_built_dataset_root_dir = '../built_dataset_dpn'
datasets_names_list = os.listdir(input_built_dataset_root_dir)
models_list = ['SVM_Classifier', 'XGBoost_Classifier', 'kNN_Classifier', 'MLP_Classifier', 'DT_Classifier']
feature_retrieval_methods_list = ['MutualInformation']
nof_features_list = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

for current_dataset_name in datasets_names_list:
    for current_model_aux in models_list:
        for current_feature_retrieval_method in feature_retrieval_methods_list:
            for current_nof_features in nof_features_list:
                current_dataset_path = '%s/%s'%(input_built_dataset_root_dir, current_dataset_name)
                current_lowcased_model = current_model_aux.replace('_Classifier', '').lower()
                logs_file_path = '../results/' + current_lowcased_model + '_%d_features_'%current_nof_features + current_dataset_name

                str_to_execute = 'python3 train.py' + ' '
                str_to_execute += '--logs_file_path' + ' ' + logs_file_path + ' '
                str_to_execute += '--model' + ' ' + current_model_aux + ' '
                str_to_execute += '--dataset_path' + ' ' + current_dataset_path + ' '
                str_to_execute += '--balancing' + ' ' + 'Oversampling' + ' '
                str_to_execute += '--feature_retrieval' + ' ' + current_feature_retrieval_method + ' '
                str_to_execute += '--splitting' + ' ' + 'Holdout' + ' '
                str_to_execute += '--test_size' + ' ' + '0.2' + ' '
                str_to_execute += '--nofsplits' + ' ' + '5' + ' '
                str_to_execute += '--preprocessing' + ' ' + 'No' + ' '
                str_to_execute += '--manual_seed' + ' ' + '10' + ' '
                str_to_execute += '--n_neighbors' + ' ' + '5' + ' '
                str_to_execute += '--noftopfeatures' + ' ' + str(current_nof_features) + ' '

                print('++++ Executing %s...'%str_to_execute)
                os.system(str_to_execute)
