import argparse
import csv
import os
import pandas as pd

def obtain_metrics_from_csv_file(csv_file_path, csv_only_filename):
    # Open the CSV file in read mode
    with open(csv_file_path, 'r') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.reader(csv_file)

        output_dict = {}
        metrics_row = []
        top_features_list = []
        it = 0
        top_features_found = False
        parameters_found = False
        # Read the CSV file line by line
        for row in csv_reader:
            if (top_features_found):
                top_features_found = False
                top_features_list = row
            
            if ((len(row)>0) and (row[0]=='top_features')):
                top_features_found = True
            
            if ((len(row)>0) and (row[0]=='mean')):
                metrics_row = row

            if ((len(row)>0) and (row[0]=='std')):
                metrics_std_row = row

            if (parameters_found):
                output_dict[row[0]] = row[1]

            if ((len(row)>0) and (row[0]=='repetition')):
                metrics_headers = row

            if ((len(row)>0) and (row[0]=='parameters')):
                parameters_found = True

            it+=1

        if (len(metrics_row)>0):
            accuracy_idx = metrics_headers.index('accuracy')
            f1_score_idx = metrics_headers.index('f1_score')
            precision_idx = metrics_headers.index('precision')
            specificity_idx = metrics_headers.index('specificity')
            recall_idx = metrics_headers.index('recall')
            auc_roc_idx = metrics_headers.index('auc_roc')
            # auc_pr_idx = metrics_headers.index('auc_pr')

            accuracy = float(metrics_row[accuracy_idx])*100, float(metrics_std_row[accuracy_idx])*100
            f1_score = float(metrics_row[f1_score_idx])*100, float(metrics_std_row[f1_score_idx])*100
            precision = float(metrics_row[precision_idx])*100, float(metrics_std_row[precision_idx])*100
            specificity = float(metrics_row[specificity_idx])*100, float(metrics_std_row[specificity_idx])*100
            recall = float(metrics_row[recall_idx])*100, float(metrics_std_row[recall_idx])*100
            auc_roc = float(metrics_row[auc_roc_idx]), float(metrics_std_row[auc_roc_idx])
            # auc_pr = float(metrics_row[auc_pr_idx]), float(metrics_std_row[auc_pr_idx])

            output_dict['accuracy'] = '%.2f%s $\pm$ %.2f%s'%(accuracy[0], '\%', accuracy[1], '\%')
            output_dict['f1_score'] = '%.2f%s $\pm$ %.2f%s'%(f1_score[0], '\%', f1_score[1], '\%')
            output_dict['precision'] = '%.2f%s $\pm$ %.2f%s'%(precision[0], '\%', precision[1], '\%')
            output_dict['specificity'] = '%.2f%s $\pm$ %.2f%s'%(specificity[0], '\%', specificity[1], '\%')
            output_dict['recall'] = '%.2f%s $\pm$ %.2f%s'%(recall[0], '\%', recall[1], '\%')
            output_dict['auc_roc'] = '%.4f $\pm$ %.4f'%auc_roc
            # AUC-PR will not be used in the results of information fusion, so it will be 
            # discarded at the moment.
            # output_dict['auc_pr'] = '%.4f+-%.4f'%auc_pr

        if (len(top_features_list)>0):
            if (len(output_dict)>0):
                # Example: if the number of top features is 20, then this list will only
                # retrieve the first 20 features. If the number is "all", then it will be
                # the whole top_features_list.
                if (output_dict['noftopfeatures']=='all'):
                    only_selected_top_features_list = top_features_list
                else:
                    noftopfeatures = int(output_dict['noftopfeatures'])
                    only_selected_top_features_list = top_features_list[:noftopfeatures] 
                
                imaging_features_list = list(filter(lambda input_value: input_value.find('feature_')!=-1, only_selected_top_features_list))
                nof_imaging_features = len(imaging_features_list)
            
                clinical_features_list = list(filter(lambda input_value: input_value.find('feature_')==-1, only_selected_top_features_list))
                nof_clinical_features = len(clinical_features_list)
            
                # Include the statistics related with the number of clinical features and the number of imaging features.
                output_dict['nof_imaging_features'] = nof_imaging_features
                output_dict['nof_clinical_features'] = nof_clinical_features

    return output_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root_dir', type=str, required=True)
    parser.add_argument('--output_csv_file_path', type=str, required=True)
    args = parser.parse_args()

    input_root_dir = args.input_root_dir
    files_names_list = os.listdir(input_root_dir)
    metrics_values_list = []
    for file_name_aux in files_names_list:
        if (file_name_aux.find('.csv')!=-1):
            full_file_path = '%s/%s'%(input_root_dir, file_name_aux)
            metrics_values = obtain_metrics_from_csv_file(full_file_path, file_name_aux)
            if (len(metrics_values)>0):
                metrics_values_list.append(metrics_values)

    metrics_values_df = pd.DataFrame(metrics_values_list)
    metrics_values_df.to_csv(args.output_csv_file_path, index=False)

main()