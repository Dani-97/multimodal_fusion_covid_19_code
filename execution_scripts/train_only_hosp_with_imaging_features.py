import os

os.chdir('../')
input_built_dataset_root_dir = '../built_dataset'
datasets_names_list = os.listdir(input_built_dataset_root_dir)
models_list = ['SVM_Classifier', 'kNN_Classifier', 'DT_Classifier', 'MLP_Classifier', 'XGBoost_Classifier']

for current_dataset_name in datasets_names_list:
    for current_model_aux in models_list:
        current_dataset_path = '%s/%s'%(input_built_dataset_root_dir, current_dataset_name)
        current_lowcased_model = current_model_aux.replace('_Classifier', '').lower()
	
        str_to_execute = 'python3 train.py' + ' '
        str_to_execute += '--logs_file_path' + ' ' + '../results/' + current_lowcased_model + '_' + current_dataset_name + ' '
        str_to_execute += '--model' + ' ' + current_model_aux + ' '
        str_to_execute += '--dataset_path' + ' ' + current_dataset_path + ' '
        str_to_execute += '--balancing' + ' ' + 'Oversampling' + ' '
        str_to_execute += '--feature_retrieval' + ' ' + 'No' + ' '
        str_to_execute += '--splitting' + ' ' + 'Holdout' + ' '
        str_to_execute += '--test_size' + ' ' + '0.2' + ' ' 
        str_to_execute += '--nofsplits' + ' ' + '5' + ' '
        str_to_execute += '--preprocessing' + ' ' + 'No' + ' '
        str_to_execute += '--manual_seed' + ' ' + '10' + ' '
        str_to_execute += '--n_neighbors' + ' ' + '5' + ' '

        print('++++ Executing %s...'%str_to_execute)
        os.system(str_to_execute)
