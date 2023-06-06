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
        it = 0
        parameters_found = False
        # Read the CSV file line by line
        for row in csv_reader:
            if ((len(row)>0) and (row[0]=='mean')):
                metrics_row = row

            if (parameters_found):
                output_dict[row[0]] = row[1]

            if ((len(row)>0) and (row[0]=='parameters')):
                parameters_found = True

            it+=1

        if (len(metrics_row)>0):
            accuracy = float(metrics_row[1])
            f1_score = float(metrics_row[2])
            precision = float(metrics_row[3])
            specificity = float(metrics_row[4])
            recall = float(metrics_row[5])
            auc_roc = float(metrics_row[6])
            auc_pr = float(metrics_row[7])

            output_dict['accuracy'] = accuracy
            output_dict['f1_score'] = f1_score
            output_dict['precision'] = precision
            output_dict['specificity'] = specificity
            output_dict['recall'] = recall
            output_dict['auc_roc'] = auc_roc
            output_dict['auc_pr'] = auc_pr

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
            metrics_values_list.append(metrics_values)

    metrics_values_df = pd.DataFrame(metrics_values_list)
    metrics_values_df.to_csv(args.output_csv_file_path, index=False)

main()
