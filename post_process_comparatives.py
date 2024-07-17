import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# NOTE: in the dataframe, the experiments with operation point = 0.5 actually have
# operation_point = NaN. It is important to take this into account.

# mode is 'w' to write and 'a' to append.
def write_to_csv_file(output_file_path, operation_point, headers_list, result_matrix, deep_features_model_str, layer_str, mode):
    # This metric will be considered to get the approach with the highest value.
    global_metric_ref = 'auc_roc'
    with open(output_file_path, mode) as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['%s (%s)'%(deep_features_model_str, layer_str), 'operation_point = %.1f'%operation_point])
        idx = 0
        for current_row_aux in result_matrix:
            if (headers_list[idx]==global_metric_ref):
                global_metric_ref_values = list(map(lambda input_value: float(input_value.split('$\pm$')[0]), current_row_aux))
                highest_auc_roc_pos = np.argmax(global_metric_ref_values)
                highest_auc_roc = global_metric_ref_values[highest_auc_roc_pos]

            csv_writer.writerow([headers_list[idx]] + current_row_aux.tolist())
            idx+=1

        best_result_row = ['best_result'] + ['']*len(current_row_aux)
        best_result_row[highest_auc_roc_pos + 1] = 'X (AUC-ROC = %.4f)'%highest_auc_roc
        csv_writer.writerow(best_result_row)
        csv_writer.writerow([])

def main():
    input_root_dir = './heliyon_results/'
    # Write the name of the CSV file without the .csv file extension.
    input_csv_file_name = 'scenario_III'
    input_df = pd.read_csv('%s/%s.csv'%(input_root_dir, input_csv_file_name))
    # Possible options: only_clinical_data, multimodal_data
    experiment_type = 'multimodal_data'
    # operation_points_list = [0.5, 0.8, 0.9]
    layers_list = ['fc6', 'fc7', 'fc8']
    cohort = 'only_hospitalized'
    output_csv_file_path = './%s/%s/comparison_%s.csv'%(input_root_dir, cohort, input_csv_file_name)
    deep_features_model_str = 'vgg_16'
    
    idx = 0
    # for current_operation_point_aux in operation_points_list:
    for current_layer_aux in layers_list:
        if (experiment_type=='only_clinical_data'):
            experiment_name = f'{cohort}_Oversampling'
        if (experiment_type=='multimodal_data'):
            experiment_name = f'{cohort}_{deep_features_model_str}_{current_layer_aux}_Oversampling'
        # query_str = "experiment_name=='%s' and operation_point==%.1f"%(experiment_name, current_operation_point_aux)
        query_str = "experiment_name=='%s'"%(experiment_name)
        # This if must be implemented because of the reason stated in the note at
        # the top of this script.
        # if (current_operation_point_aux==0.5):
        #     query_str = "experiment_name=='%s' and operation_point!=operation_point"%(experiment_name)
            
        result_df = input_df.query(query_str)
        selected_headers_list = ['accuracy', 'mcc', 'f1_score', 'precision', 'recall', 'specificity', \
                                    'auc_roc', 'nof_imaging_features', 'nof_clinical_features']

        result_df = result_df[selected_headers_list]
        result_df['noftopfeatures'] = result_df['nof_imaging_features'] + result_df['nof_clinical_features']
        # Placing the last column as the first one.
        selected_headers_list = [result_df.columns[len(result_df.columns)-1]] + result_df.columns[0:-1].tolist()
        result_df = result_df.sort_values(by=['noftopfeatures'])
        result_df = result_df[selected_headers_list]
        result_matrix = np.transpose(result_df.values)

        # The first time, if the CSV file already exists, it must be overwritten.
        if (idx==0):
            mode='w'
        else:
            mode='a'
            
        default_operation_point = 0.5
        write_to_csv_file(output_csv_file_path, default_operation_point, selected_headers_list, \
                            result_matrix, deep_features_model_str, current_layer_aux, mode)
        
        idx+=1
    
main()
