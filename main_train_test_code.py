import argparse
from classifiers.utils_classifiers import *
from regressors.utils_regressors import *
from dataset.utils_balancing import *
from dataset.utils_datasets import *
from dataset.utils_features import *
from dataset.utils_missing_values import *
from dataset.utils_normalization import *
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import auc
from utils import convert_metrics_dict_to_list, clear_csv_file
import warnings

class UniversalFactory():

    def __init__(self):
        pass

    def create_object(self, namespace, classname, kwargs):
        ClassName = namespace[classname]
        universal_obj = ClassName(**kwargs)
        return universal_obj

class Super_Train_Test_Class():

    def __init__(self):
        self.operation_point = 0.5

    # This functions decides if the model will be trained or loaded from a
    # previous stored checkpoint. The method is just an interface, so it must
    # be implemented by each subclass.
    def __train_or_load_model__(self, it, args, model, input_train_subset, output_train_subset):
        raise NotImplementedError('++++ ERROR: the __train_or_load_model__ method has not been implemented!')

    def __validate_model__(self, it, args, model, input_train_subset, output_train_subset):
        raise NotImplementedError('++++ ERROR: the __validate_model__ method has not been implemented!')

    def __store_roc_curves__(self, experiment_name, roc_curves_list, output_path):
        fig, ax = plt.subplots()
        ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title(experiment_name, fontsize=20)
        plt.xlabel('False Positive Ratio', fontsize=10)
        plt.ylabel('True Positive Ratio', fontsize=10)
        legend = ['Random classifier']
        it = 0
        plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '--')

        color = plt.cm.rainbow(np.linspace(0, 1, len(roc_curves_list)))
        for current_metrics_values_aux, current_roc_curve_aux in roc_curves_list:
            current_fpr, current_tpr, _ = current_roc_curve_aux
            plt.plot(current_fpr, current_tpr, c=np.array([color[it]]))
            recall = current_metrics_values_aux['recall']
            specificity = current_metrics_values_aux['specificity']
            plt.scatter(1.0-specificity, recall, c=np.array([color[it]]))
            legend+=['roc_rep_%d = %.4f'%(it, auc(current_fpr, current_tpr)), 'metrics_rep_%d'%it]
            it+=1

        plt.legend(legend, fontsize="10")
        plt.savefig(output_path)

    # This model makes possible to save the model in training but not in testing reusing the 
    # same code scheme for both cases.
    def __save_trained_model__(self, it, current_fold, args, model, is_best_model=False):
        raise NotImplementedError('++++ ERROR: the method __save_trained_model__ must be implemented by the subclass!')

    def execute_approach(self, args):
        # warnings.filterwarnings('ignore')
        universal_factory = UniversalFactory()

        # Creating the splitting object with the universal factory.
        kwargs = {'test_size': args.test_size, 'noffolds': args.nofsplits}
        splitting = universal_factory.create_object(globals(), args.splitting + '_Split', kwargs)
        # Retrieving the feature selection method with the universal factory.
        # kwargs = {'noftopfeatures': args.noftopfeatures, 'nofcomponents': args.nofcomponents}
        kwargs = {'noftopfeatures': args.noftopfeatures}
        feature_retrieval = universal_factory.create_object(globals(), args.feature_retrieval + '_Feature_Retrieval', kwargs)
        # Creating the object with the currently selected preprocessing approach.
        kwargs = {}
        preprocessing = universal_factory.create_object(globals(), args.preprocessing + '_Preprocessing', kwargs)

        attrs_headers, input_data, output_data = splitting.load_dataset(args.dataset_path)

        # Check if the number of top features that we want to retrieve is lower or equal than the number of 
        # features that the data have. In case the requested number is greater, then the execution will end.
        if ((args.noftopfeatures!='all') and (int(args.noftopfeatures)>input_data.shape[1])):
            print('++++ ERROR: The number of requested features is greater than the number of features that the data have!')
            exit(-1)

        roc_curves_list = []
        for it in range(0, args.nofsplits):
            print('**** Starting repetition number %d...'%it)
            # For certain splitting methods, this line does not do anything
            # (for example, in the case of the holdout).
            splitting.set_partition(it)

            # This list specifies if the attributes of the input dataset are categorical
            # or numerical.
            attrs_types_list = splitting.check_attrs_type(input_data)

            # Split into training and test set
            subsets = splitting.split_train_test(input_data, output_data, args.manual_seeds[it])
            preprocessing_params = splitting.obtain_attributes_statistics(input_data)
            input_train_subset, input_test_subset, \
                                 output_train_subset, output_test_subset = subsets
            output_train_subset = output_train_subset.astype(float).astype(int).astype(str)
            output_test_subset = output_test_subset.astype(float).astype(int).astype(str)

            train_idxs, val_idxs = splitting.split_train_val(args.nof_folds,
                                                             input_train_subset,
                                                             output_train_subset,
                                                             args.manual_seeds[it])
            
            # This one is for validation, in order to select the feature set that obtains the highest performance.
            best_global_metric = -1
            for current_fold_aux in range(0, args.nof_folds):
                current_train_input_data, current_val_input_data, \
                    current_train_output_data, current_val_output_data = \
                        splitting.get_current_fold_data(current_fold_aux,
                                                        input_train_subset,
                                                        output_train_subset,
                                                        train_idxs, val_idxs)

                # The input_data variable is overwritten with the data obtained after the
                # feature selection. If the user decided not to use feature selection, then
                # this function will not actually do anything.
                feature_retrieval.set_attributes_types(attrs_types_list)
                top_features_idxs = feature_retrieval.execute_feature_retrieval(current_train_input_data,
                                                                                current_train_output_data,
                                                                                plot_data=args.plot_data)
                transformed_input_train_subset = feature_retrieval.apply_feature_ranking_transform(current_train_input_data,
                                                                                    top_features_idxs)
                transformed_input_val_subset = feature_retrieval.apply_feature_ranking_transform(current_val_input_data,
                                                                                    top_features_idxs)

                # If the CSV logs file had previous content, then this function will remove
                # it.
                # clear_csv_file(args.logs_file_path)
                top_features_attrs_headers = feature_retrieval.get_ordered_top_features(attrs_headers, args.noftopfeatures)
                top_features_categorical_attrs_headers, top_features_continuous_attrs_headers = \
                        feature_retrieval.get_ordered_categorical_and_continuous_top_features(attrs_headers, \
                                                                            args.noftopfeatures, args.csv_path_with_attrs_types)

                kwargs = {}
                imputation_module = \
                        universal_factory.create_object(globals(), args.imputation, kwargs)
                transformed_input_train_subset, output_train_subset = \
                        imputation_module.execute_imputation(transformed_input_train_subset, output_train_subset)

                # Performing the balancing on the dataset.
                kwargs = {'manual_seed': args.manual_seeds[it]}
                balancing_module = \
                        universal_factory.create_object(globals(), args.balancing + '_Balancing', kwargs)
                top_features_tuple = top_features_categorical_attrs_headers, \
                                     top_features_continuous_attrs_headers, \
                                     top_features_attrs_headers
                transformed_input_train_subset, current_train_output_data = \
                        balancing_module.execute_balancing(transformed_input_train_subset, current_train_output_data, top_features_tuple)

                transformed_input_train_subset, _ = preprocessing.execute_preprocessing(transformed_input_train_subset.astype(np.float64), top_features_attrs_headers, preprocessing_params)
                transformed_input_val_subset, _ = preprocessing.execute_preprocessing(transformed_input_val_subset.astype(np.float64), top_features_attrs_headers, preprocessing_params)

                # Convert all the cells of the subsets to double.
                transformed_input_train_subset = transformed_input_train_subset.astype(np.float64)
                transformed_input_val_subset = transformed_input_val_subset.astype(np.float64)
                
                current_train_output_data = current_train_output_data.astype(np.float64)
                current_val_output_data = current_val_output_data.astype(np.float64)

                # Creating the model with the universal factory. output_train_data may be necessary to
                # calculate the imbalance of the dataset and train the model accordingly.
                kwargs = {'n_neighbors': args.n_neighbors, 'output_train_data': current_train_output_data}
                # This function will train the model or load a previous one that
                # was stored.
                model = universal_factory.create_object(globals(), args.model, kwargs)
                model = self.__train_or_load_model__(it, args, model, transformed_input_train_subset, current_train_output_data)
                val_metrics = self.__validate_model__(it, args, model, transformed_input_val_subset, current_val_output_data)

                current_global_metric_value = val_metrics['mcc']
                if (current_global_metric_value > best_global_metric):
                    self.__save_trained_model__(it, current_fold_aux, args, model, is_best_model=True)
                    best_global_metric = current_global_metric_value
                    best_feature_set_idxs = top_features_idxs
                    best_model = model

                    # This function will store the report of the feature selection process if it
                    # is available and if the user has decided to store it.
                    if (args.store_features_selection_report):
                        dir_to_store_results = os.path.dirname(args.logs_file_path)
                        feature_retrieval.set_dir_to_store_results(dir_to_store_results)
                        feature_ranking_logs_file_path = args.logs_file_path.replace('.csv', '_feature_ranking_%d.csv'%it)
                        feature_retrieval.store_report(feature_ranking_logs_file_path, attrs_headers, append=False)
                    else:
                        print('++++ As the user decided, the report of the features' + \
                            ' selection algorithm will not be stored to the logging results file')

                self.__save_trained_model__(it, current_fold_aux, args, model)

            # Obtain metrics for test.
            transformed_input_test_subset = feature_retrieval.apply_feature_ranking_transform(input_test_subset, best_feature_set_idxs)
            transformed_input_test_subset, _ = preprocessing.execute_preprocessing(transformed_input_test_subset.astype(np.float64), top_features_attrs_headers, preprocessing_params)
            
            transformed_input_test_subset = transformed_input_test_subset.astype(np.float64)
            output_test_subset = output_test_subset.astype(np.float64)

            test_model_output_pred = best_model.test(transformed_input_test_subset, self.operation_point)
            test_metrics_values = best_model.model_metrics(test_model_output_pred, output_test_subset)

            current_roc_curve = best_model.get_roc_curve(test_model_output_pred, output_test_subset)
            roc_curves_list.append((test_metrics_values, current_roc_curve))

            headers_list, test_metrics_values_list = convert_metrics_dict_to_list(test_metrics_values)
            if (it==0):
                best_model.add_headers_to_csv_file(args.logs_file_path, headers_list, append=False)
            # This function will store the number of the current repetition (the value of "it")
            # and the metrics of the current repetition.
            best_model.store_model_metrics(it, test_metrics_values_list, args.logs_file_path)
            # Here we obtain the explainability of the model if it is available.
            best_model.explainability()
            print('---- NOTE: the logs of this repetition are now stored at %s\n'%args.logs_file_path)

        roc_curve_filename = args.experiment_name + '_' + args.model + '_' + str(args.noftopfeatures) + '_' + \
                               args.feature_retrieval + '_' + args.balancing + '_model'
        roc_curve_file_full_path = \
                '%s/%s.pdf'%(str(Path(args.logs_file_path).parent), roc_curve_filename)
        # The mean and the standard deviation of the metrics obtained for
        # all the repetitions are stored at the end of the CSV file.
        best_model.compute_mean_and_std_performance(args.logs_file_path)
        # In addition, we also store the parameters of the current experiment to
        # the CSV file (experiment_name, classifier, number of features...).
        best_model.store_experiment_parameters(args, args.logs_file_path)
        self.__store_roc_curves__(args.experiment_name, roc_curves_list, roc_curve_file_full_path)

class Train_Class(Super_Train_Test_Class):

    def __init__(self):
        super().__init__()

    def __save_trained_model__(self, it, current_fold, args, model, is_best_model=False):
        model_filename = args.experiment_name + '_' + args.model + '_' + str(args.noftopfeatures) + '_' + \
                            args.feature_retrieval + '_' + args.balancing + '_model'
        if (is_best_model):
            model_file_full_path = \
                    '%s/%s_best_model_rep_%d.pkl'%(str(Path(args.logs_file_path).parent), model_filename, it)
        else:
            model_file_full_path = \
                    '%s/%s_rep_%d_cv_%d.pkl'%(str(Path(args.logs_file_path).parent), model_filename, it, current_fold)
        model.save_model(model_file_full_path)

    def __train_or_load_model__(self, it, args, model, input_train_subset, output_train_subset):
        model.train(input_train_subset, output_train_subset)
        # Obtain metrics for training.
        model_training_output_pred = model.test(input_train_subset)
        training_metrics_values = model.model_metrics(model_training_output_pred, output_train_subset)

        training_metrics_file_path = os.path.dirname(args.logs_file_path) + '/' + args.experiment_name + \
                             '_' + args.model + '_' + str(args.noftopfeatures) + '_' + \
                                   args.feature_retrieval + '_' + args.balancing + '_training_metrics.txt'
        model.show_training_metrics(training_metrics_values, training_metrics_file_path, it)

        return model
    
    def __validate_model__(self, it, args, model, input_val_subset, output_val_subset):
        model_output_pred = model.test(input_val_subset)
        val_metrics_values = model.model_metrics(model_output_pred, output_val_subset)
        val_metrics_file_path = os.path.dirname(args.logs_file_path) + '/' + args.experiment_name + \
                             '_' + args.model + '_' + str(args.noftopfeatures) + '_' + \
                                   args.feature_retrieval + '_' + args.balancing + '_val_metrics.txt'
        model.show_training_metrics(val_metrics_values, val_metrics_file_path, it)

        return val_metrics_values

class Test_Class(Super_Train_Test_Class):

    def __init__(self):
        super().__init__()

    # During the test phase, the models will not be stored.
    def __save_trained_model__(self, it, current_fold, args, model, is_best_model=False):
        pass

    def __train_or_load_model__(self, it, args, model, input_train_subset, output_train_subset):
        self.operation_point = args.operation_point
        current_model_path_to_load = args.model_path.replace('.pkl', '')
        current_model_path_to_load = current_model_path_to_load + '_rep_%d.pkl'%it
        print('**** Loading the model %s...'%current_model_path_to_load)
        model.load_model(current_model_path_to_load)

        return model
