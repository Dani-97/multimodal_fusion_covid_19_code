import argparse
import csv
import numpy as np
import os
import pandas as pd

def obtain_top_features_info(noftopfeatures, nofrepetitions, csv_file_path):
    nof_clinical_features = 0
    nof_imaging_features = 0
    for it in range(0, nofrepetitions):
        current_top_features_file_path = csv_file_path.replace('.csv', '_feature_ranking_%d.csv'%it)
        current_top_features_df = pd.read_csv(current_top_features_file_path)
        features_names_list = current_top_features_df['feature_name'].values
        if (noftopfeatures=='all'):
            noftopfeatures = len(features_names_list)
        noftopfeatures = int(noftopfeatures)
        nof_clinical_features += len(list(filter(lambda input_value: input_value.find('feature_')==-1, features_names_list[:noftopfeatures])))
        nof_imaging_features += len(list(filter(lambda input_value: input_value.find('feature_')!=-1, features_names_list[:noftopfeatures])))

    mean_nof_clinical_features = nof_clinical_features / nofrepetitions
    mean_nof_imaging_features = nof_imaging_features / nofrepetitions

    return mean_nof_clinical_features, mean_nof_imaging_features

def obtain_metrics_from_csv_file(csv_file_path, csv_only_filename):
    # Open the CSV file in read mode
    with open(csv_file_path, 'r') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.reader(csv_file)

        output_dict = {}
        repetitions_list = []
        metrics_row = []
        top_features_list = []
        it = 0
        parameters_found = False
        metrics_per_repetition_found = False

        # Read the CSV file line by line
        for row in csv_reader:
            if (metrics_per_repetition_found):
                if (len(row)>0):
                    repetitions_list.append(row[0])
                else:
                    metrics_per_repetition_found = False
            
            if ((len(row)>0) and (row[0]=='mean')):
                metrics_row = row

            if ((len(row)>0) and (row[0]=='std')):
                metrics_std_row = row

            if (parameters_found):
                if (row[0]=='noftopfeatures'):
                    nofrepetitions = len(repetitions_list)
                    noftopfeatures = row[1]
                    nof_clinical_features, nof_imaging_features = \
                        obtain_top_features_info(noftopfeatures, nofrepetitions, csv_file_path)
                else:
                    output_dict[row[0]] = row[1]

            if ((len(row)>0) and (row[0]=='repetition')):
                metrics_headers = row
                metrics_per_repetition_found = True

            if ((len(row)>0) and (row[0]=='parameters')):
                parameters_found = True

            it+=1

        if (len(metrics_row)>0):
            accuracy_idx = metrics_headers.index('accuracy')
            mcc_idx = metrics_headers.index('mcc')
            f1_score_idx = metrics_headers.index('f1_score')
            precision_idx = metrics_headers.index('precision')
            specificity_idx = metrics_headers.index('specificity')
            recall_idx = metrics_headers.index('recall')
            auc_roc_idx = metrics_headers.index('auc_roc')
            # auc_pr_idx = metrics_headers.index('auc_pr')

            accuracy = float(metrics_row[accuracy_idx])*100, float(metrics_std_row[accuracy_idx])*100
            mcc = float(metrics_row[mcc_idx]), float(metrics_std_row[mcc_idx])
            f1_score = float(metrics_row[f1_score_idx])*100, float(metrics_std_row[f1_score_idx])*100
            precision = float(metrics_row[precision_idx])*100, float(metrics_std_row[precision_idx])*100
            specificity = float(metrics_row[specificity_idx])*100, float(metrics_std_row[specificity_idx])*100
            recall = float(metrics_row[recall_idx])*100, float(metrics_std_row[recall_idx])*100
            auc_roc = float(metrics_row[auc_roc_idx]), float(metrics_std_row[auc_roc_idx])
            # auc_pr = float(metrics_row[auc_pr_idx]), float(metrics_std_row[auc_pr_idx])

            output_dict['accuracy'] = '%.2f%s $\pm$ %.2f%s'%(accuracy[0], '\%', accuracy[1], '\%')
            output_dict['mcc'] = '%.4f $\pm$ %.4f'%(mcc[0], mcc[1])
            output_dict['f1_score'] = '%.2f%s $\pm$ %.2f%s'%(f1_score[0], '\%', f1_score[1], '\%')
            output_dict['precision'] = '%.2f%s $\pm$ %.2f%s'%(precision[0], '\%', precision[1], '\%')
            output_dict['specificity'] = '%.2f%s $\pm$ %.2f%s'%(specificity[0], '\%', specificity[1], '\%')
            output_dict['recall'] = '%.2f%s $\pm$ %.2f%s'%(recall[0], '\%', recall[1], '\%')
            output_dict['auc_roc'] = '%.4f $\pm$ %.4f'%auc_roc
            # AUC-PR will not be used in the results of information fusion, so it will be
            # discarded at the moment.
            # output_dict['auc_pr'] = '%.4f+-%.4f'%auc_pr

        output_dict['noftopfeatures'] = noftopfeatures
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
    for it, file_name_aux in enumerate(files_names_list):
        print('++++ [%d/%d] Processing %s...'%(it, len(files_names_list), file_name_aux))
        if ((file_name_aux.find('.csv')!=-1) and not (file_name_aux.find('feature_ranking')!=-1)):
            full_file_path = '%s/%s'%(input_root_dir, file_name_aux)
            metrics_values = obtain_metrics_from_csv_file(full_file_path, file_name_aux)
            metrics_values_list.append(metrics_values)

    metrics_values_df = pd.DataFrame(metrics_values_list)
    metrics_values_df.to_csv(args.output_csv_file_path, index=False)

main()