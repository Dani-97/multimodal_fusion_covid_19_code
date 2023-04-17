import os

def main():
    layers_list = ['all']
    deep_features_models_list = ['DPN_Model_']
    chosen_cohorts_list = ['Only_Hospitalized', 'Hospitalized_And_Urgencies']

    os.chdir('../')
    for deep_features_model_aux in deep_features_models_list:
        for current_chosen_cohort_aux in chosen_cohorts_list:
            for current_layer_aux in layers_list:
                current_output_path_aux = '../built_dataset_dpn/%s_%s%s.csv'%(current_chosen_cohort_aux.lower(), deep_features_model_aux.lower(), current_layer_aux)
                current_approach_aux = deep_features_model_aux + current_chosen_cohort_aux
	
                str_to_execute = 'python3 build_dataset_with_imaging_features.py '
                str_to_execute += ('--input_dataset_path' + ' ' + '../original_dataset/distributed_images/' + ' ')
                str_to_execute += ('--headers_file' + ' ' + '../original_dataset/headers_with_patient_id_dict.txt' + ' ')
                str_to_execute += ('--input_table_file' + ' ' + '../original_dataset/original_input_with_ids.csv' + ' ')
                str_to_execute += ('--associations_file' + ' ' + '../original_dataset/filtered_associations.csv' + ' ')
                str_to_execute += ('--approach' + ' ' + current_approach_aux + ' ')
                str_to_execute += ('--output_path' + ' ' + current_output_path_aux + ' ')
                str_to_execute += ('--device' + ' ' + 'CUDA' + ' ')
                str_to_execute += ('--layer' + ' ' + current_layer_aux + ' ')
                str_to_execute += ('--text_reports_embeds_method' + ' ' + 'No' + ' ')

                print('Executing %s...'%str_to_execute)
                os.system(str_to_execute)
                
main()
