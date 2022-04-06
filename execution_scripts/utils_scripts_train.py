import os

def execute_train(noffeatures_list, nofexperiments, logs_dir_path_list, logs_file_name_list, model_list, \
            dataset_path_list, balancing_list, feature_retrieval_list, splitting_list, \
                test_size_list, nofsplits_list, preprocessing_list, \
                    store_features_selection_report_list, n_neighbors_list=None):

    for it in range(0, nofexperiments):
        if (len(str(logs_dir_path_list[it]).strip())!=0):
            current_logs_file_path = ' --logs_file_path ' + str(logs_dir_path_list[it]) + logs_file_name_list[it]%str(noffeatures_list[it])
        else:
            current_logs_file_path = ''

        if (len(str(model_list[it]).strip())!=0):
            current_model = ' --model ' + str(model_list[it])
        else:
            current_model = ' '

        if (len(str(dataset_path_list[it]).strip())!=0):
            current_dataset_path = ' --dataset_path ' + str(dataset_path_list[it])
        else:
            current_dataset_path = ' '

        if (len(str(balancing_list[it]).strip())!=0):
            current_balancing = ' --balancing ' + str(balancing_list[it])
        else:
            current_balancing = ' '

        if (len(str(feature_retrieval_list[it]).strip())!=0):
            current_feature_retrieval = ' --feature_retrieval ' + str(feature_retrieval_list[it])
        else:
            current_feature_retrieval = ' '

        if (len(str(splitting_list[it]).strip())!=0):
            current_splitting = ' --splitting ' + str(splitting_list[it])
        else:
            current_splitting = ' '

        if (len(str(test_size_list[it]).strip())!=0):
            current_test_size = ' --test_size ' + str(test_size_list[it])
        else:
            current_test_size = ' '

        if (len(str(nofsplits_list[it]).strip())!=0):
            current_nofsplits = ' --nofsplits ' + str(nofsplits_list[it])
        else:
            current_nofsplits = ' '

        if (noffeatures_list[it]=='all'):
            current_noftopfeatures = ' '
        else:
            current_noftopfeatures = ' --noftopfeatures ' + str(noffeatures_list[it])

        if (len(str(preprocessing_list[it]).strip())!=0):
            current_preprocessing = ' --preprocessing ' + str(preprocessing_list[it])
        else:
            current_preprocessing = ' '

        if (n_neighbors_list!=None):
            current_n_neighbors = ' --n_neighbors ' + str(n_neighbors_list[it])
        else:
            current_n_neighbors = ''

        if (len(str(store_features_selection_report_list[it]).strip())!=0):
            current_store_features_selection_report = ' ' + str(store_features_selection_report_list[it])
        else:
            current_store_features_selection_report = ' '

        command_to_execute = 'python3 train.py' + current_logs_file_path + \
            current_model + current_dataset_path + current_balancing + \
                current_feature_retrieval + current_splitting + current_test_size + \
                    current_nofsplits + current_noftopfeatures + current_preprocessing + \
                        current_n_neighbors + current_store_features_selection_report

        print('\n##################################################################')
        print(' Executing command [%s]'%command_to_execute)
        print('##################################################################\n')
        os.system(command_to_execute)
