import configparser
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def plot_report(headers_list_with_it, scores_df, pdf_path_to_save):
    translations_path = 'chuac_classic_ml_covid19_prognosis/dataset/translations.cfg'
    translations_list = configparser.ConfigParser()
    translations_list.read(translations_path)

    xticks_list = []
    for current_column_aux in headers_list_with_it:
        current_item = '%d.%s'%(int(current_column_aux.split('.')[0])+1,
                                translations_list['TRANSLATIONS'][current_column_aux.split('.')[1]])
        xticks_list.append(current_item)

    fig = plt.figure(figsize=(15, 5))
    plt.xticks(list(range(0, len(scores_df.columns))), xticks_list, rotation=90)
    plt.bar(list(range(0, len(scores_df.columns))), scores_df.values[0], color='green')
    plt.subplots_adjust(left=0.10, bottom=0.4, right=0.90, top=0.97)

    plt.savefig(pdf_path_to_save)

def main():
    # This should be scenario_I or scenario_III, given that scenario_II do not have clinical features,
    # but only imaging features.
    chosen_scenario = 'scenario_III'
    input_root_dir = './heliyon_results/results_%s'%chosen_scenario
    # This can be either 'hospitalized_and_urgencies' or 'only_hospitalized'.
    risk_scenario = 'only_hospitalized'
    deep_network_layer = 'fc8'
    deep_network_architecture = 'vgg_16'
    experiment_name = '%s_XGBoost_Classifier_all_features_%s_%s_Oversampling'%(risk_scenario,
                                                                               deep_network_architecture,
                                                                               deep_network_layer)
    pdf_path_to_save = '%s_%s_%s.pdf'%(risk_scenario, deep_network_layer, 'mutual_information')

    files_list = os.listdir(input_root_dir)

    indexes_lists = {}
    scores_lists = {}
    it = 0
    for file_name_aux in files_list:
        if ((file_name_aux.find(experiment_name)!=-1) \
                    and (file_name_aux.find('.csv')!=-1) and (file_name_aux.find('feature_ranking')!=-1)):
            full_file_path = '%s/%s'%(input_root_dir, file_name_aux)
            input_df = pd.read_csv(full_file_path)
            
            if (it==0):
                features_names_list = input_df['feature_name'].values
                for current_feature_name_aux in features_names_list:
                    indexes_lists[current_feature_name_aux] = []
                    scores_lists[current_feature_name_aux] = []
            
            for current_feature_name_aux in features_names_list:
                indexes_lists[current_feature_name_aux] = indexes_lists[current_feature_name_aux] + \
                    input_df.query("feature_name=='%s'"%current_feature_name_aux)['index'].values.tolist()
                scores_lists[current_feature_name_aux] = scores_lists[current_feature_name_aux] + \
                    input_df.query("feature_name=='%s'"%current_feature_name_aux)['score'].values.tolist()

            it+=1

    mean_indexes_list = {}
    mean_scores_list = {}
    for current_feature_name_aux in features_names_list:
        mean_index_aux = np.mean(np.array(indexes_lists[current_feature_name_aux]))
        mean_score_aux = np.mean(np.array(scores_lists[current_feature_name_aux]))

        mean_indexes_list[current_feature_name_aux] = [mean_index_aux]
        mean_scores_list[current_feature_name_aux] = [mean_score_aux]
    
    mean_scores_df = pd.DataFrame.from_dict(mean_scores_list)
    sorted_idxs = np.flip(np.argsort(mean_scores_df))
    sorted_headers_list = mean_scores_df.columns[sorted_idxs[0]]
    
    headers_list_with_it = []
    for idx, current_header_aux in enumerate(sorted_headers_list):
        headers_list_with_it.append('%d.%s'%(idx, current_header_aux))

    sorted_headers_list = list(filter(lambda input_value: input_value.find('feature_')==-1, sorted_headers_list))
    headers_list_with_it = list(filter(lambda input_value: input_value.find('feature_')==-1, headers_list_with_it))

    if (len(sorted_headers_list)>0):
        mean_scores_df = mean_scores_df[sorted_headers_list]
        plot_report(headers_list_with_it, mean_scores_df, pdf_path_to_save)
    else:
        print('++++ The report has not been obtained, given that there are not clinical features in this ranking, but only imaging features.')

main()