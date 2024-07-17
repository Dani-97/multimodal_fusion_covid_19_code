import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# This function obtains the mean and the standard deviation
# of the selected metric.
def get_metric_values(query_df, metric_name, sorted_idxs):
    metric_mean_values_list = query_df[metric_name].apply(lambda input_value: float(input_value.split('$\pm$')[0].replace('\%', '')))
    metric_std_values_list = query_df[metric_name].apply(lambda input_value: float(input_value.split('$\pm$')[1].replace('\%', '')))

    metric_mean_values_list = np.array(metric_mean_values_list)[sorted_idxs]
    metric_std_values_list = np.array(metric_std_values_list)[sorted_idxs]

    return metric_mean_values_list, metric_std_values_list

def main():
    input_root_dir = './heliyon_results/'
    csv_baseline_path = '%s/scenario_I.csv'%input_root_dir
    csv_paths_list = ['%s/scenario_II.csv'%input_root_dir, '%s/scenario_III.csv'%input_root_dir]

    # baseline_experiment_name = 'only_hospitalized_Oversampling'
    baseline_experiment_name = 'hospitalized_and_urgencies_Oversampling'
    
    # experiment_name_root = 'only_hospitalized_vgg_16'
    experiment_name_root = 'hospitalized_and_urgencies_vgg_16'
    experiment_names = ['%s_fc6_Oversampling'%experiment_name_root, \
                        '%s_fc7_Oversampling'%experiment_name_root, \
                        '%s_fc8_Oversampling'%experiment_name_root]
    # operation_point = None
    # hr stands for "human readable".
    # hr_subplots_names = ['VGG-16 (fc6)', 'VGG-16 (fc7)', 'VGG-16 (fc8)']
    hr_subplots_names = ['VGG-16 (fc6)', 'VGG-16 (fc7)', 'VGG-16 (fc8)']
    colours_list = ['#1f77b4', '#ff7f0e']
    symbols_list = ['--', '-']
    # This selects if, for example, you want to show the F1-Score or the AUC-ROC.
    chosen_metric = 'f1_score'
    path_to_save_fig = 'output.pdf'

    baseline_dataframe = pd.read_csv(csv_baseline_path)

    dataframes_list = []
    for current_csv_path_aux in csv_paths_list:
        dataframes_list.append(pd.read_csv(current_csv_path_aux))

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    fig.tight_layout(pad=5.0)
    for experiment_it, current_experiment_name_aux in enumerate(experiment_names):
        baseline_query_str_without_op = "experiment_name=='%s' and noftopfeatures=='all'"%(baseline_experiment_name)
        query_str = "experiment_name=='%s'"%(current_experiment_name_aux)

        for dataframe_it, current_dataframe_aux in enumerate(dataframes_list):
            baseline_query_df = baseline_dataframe.query(baseline_query_str_without_op)
            query_df = current_dataframe_aux.query(query_str)

            nof_imaging_features = query_df['nof_imaging_features']
            nof_clinical_features = query_df['nof_clinical_features']

            total_nof_imaging_features = np.max(query_df['nof_imaging_features'])
            total_nof_clinical_features = np.max(query_df['nof_clinical_features'])

            noftopfeatures = query_df['nof_imaging_features'] + query_df['nof_clinical_features']
            # noftopfeatures = query_df['noftopfeatures']
            sorted_idxs = np.argsort(noftopfeatures)
            noftopfeatures = np.array(noftopfeatures)[sorted_idxs]
            
            nof_imaging_features = np.array(nof_imaging_features)[sorted_idxs]
            nof_clinical_features = np.array(nof_clinical_features)[sorted_idxs]

            # Plot the dashed line that represents the baseline performance.
            baseline_value = baseline_query_df[chosen_metric].values[0]
            baseline_value = float(baseline_value.split('$\pm$')[0].replace('\%', ''))
            
            # The performance of the baseline must be plotted only once.
            if (dataframe_it==0):
                ax[experiment_it].plot(noftopfeatures, [baseline_value]*len(noftopfeatures), 'k--')

            # Show the evolution of the number of clinical features.
            ax2 = ax[experiment_it].twinx()
            ax2.grid()
            ax2.set_ylim([0, 105])
            ax2.set_ylabel('#CF (%)')
            if ((len(nof_clinical_features)>0) and (np.max(nof_clinical_features)>0)):
                clinical_features_percentage = (np.array(nof_clinical_features)/np.max(nof_clinical_features))*100
                ax2.bar(noftopfeatures, clinical_features_percentage, width=10.0, alpha=0.35, \
                        color=colours_list[dataframe_it])
                ax2.legend(['#CF'], fancybox=True, shadow=True, loc='center right')

            mean_metric_values_list, std_metric_values_list = get_metric_values(query_df, chosen_metric, sorted_idxs)

            # Plot the evolution of the metric.
            ax[experiment_it].set_title(hr_subplots_names[experiment_it])
            ax[experiment_it].plot(noftopfeatures, mean_metric_values_list, symbols_list[dataframe_it], color='%s'%(colours_list[dataframe_it]))
            
        ylabel_left = chosen_metric.replace('_', '-')
        if (ylabel_left!='auc_roc'):
            ylabel_left = ylabel_left + ' (%)'
        
        ax[experiment_it].set_xlim([-20, 3700])
        # ax[experiment_it].set_ylim([34.5, 55.5])
        ax[experiment_it].set_xlabel('Total number of features')
        ax[experiment_it].set_ylabel(ylabel_left)

    legend = fig.legend(['Approach I: Only clinical data', 'Approach II: Only imaging data', 'Approach III: Multimodal data fusion'], fancybox=True, shadow=True, ncol=3, loc='upper center')
    plt.savefig(path_to_save_fig)

main()
