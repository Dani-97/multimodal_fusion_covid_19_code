import csv
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, auc
from sklearn.metrics import precision_recall_curve
import sys
from xgboost import XGBClassifier

# This is the parent of every classifier models that are going to be
# implemented for the purposes of this work.
class Super_Classifier_Class():

    def __init__(self, **kwargs):
        pass

    def __custom_roc_curve__(self, y_test, probabilities, partitions=100):
        sorted_probabilities_idxs = np.argsort(probabilities)[::-1]
        sorted_probabilities = probabilities[sorted_probabilities_idxs]
        sorted_y_test = y_test[sorted_probabilities_idxs]
        roc_curve = []
        thresholds = []
        for it in range(partitions + 1):
            thresholds.append(it/partitions)
            threshold_vector = np.greater_equal(sorted_probabilities, it/partitions).astype(int)
            true_positive = np.equal(threshold_vector, 1) & np.equal(sorted_y_test, 1)
            true_negative = np.equal(threshold_vector, 0) & np.equal(sorted_y_test, 0)
            false_positive = np.equal(threshold_vector, 1) & np.equal(sorted_y_test, 0)
            false_negative = np.equal(threshold_vector, 0) & np.equal(sorted_y_test, 1)

            fpr = false_positive.sum()/(false_positive.sum() + true_negative.sum())
            tpr = true_positive.sum()/(true_positive.sum() + false_negative.sum())

            roc_curve.append([fpr, tpr])

        roc_curve = np.array(roc_curve)
        fpr, tpr = roc_curve[:, 0], roc_curve[:, 1]

        return fpr, tpr, thresholds

    def __custom_auc_function__(self, fpr, tpr, thresholds):
        rectangle_auc = 0
        for it in range(0, len(thresholds)-1):
            rectangle_auc = rectangle_auc + (fpr[it] - fpr[it + 1]) * tpr[it]

        return rectangle_auc

    def train(self, input_data, output_data):
        self.classifier.fit(input_data, output_data)

    def test(self, input_data, threshold=0.5):
        classifier_probabilities = self.classifier.predict_proba(input_data)
        classifier_label_predictions = (classifier_probabilities[:, 1]>threshold).astype(int)

        return classifier_label_predictions, classifier_probabilities

    def show_training_metrics(self, metrics_values, metrics_file_path, nofsplit):
        tn, fp, fn, tp = metrics_values['confusion_matrix'].ravel()

        original_stdout = sys.stdout
        if (nofsplit>0):
            metrics_file_mode = 'a'
        else:
            metrics_file_mode = 'w'

        metrics_file = open(metrics_file_path, metrics_file_mode)
        sys.stdout = metrics_file
        print('#### Training performance report (split number %d) ####'%nofsplit)
        print('=================')
        print('Confusion matrix')
        print('[[%d \t %d\n  %d \t %d]]'%(tn, fp, fn, tp))
        print('=================')
        print('AUC-ROC = %.4f'%metrics_values['auc_roc'])
        print('=================')
        print('AUC-PR = %.4f'%metrics_values['auc_pr'])
        print('=================')
        print('Accuracy = %.2f'%(metrics_values['accuracy']*100))
        print('=================')
        print('F1-Score = %.2f'%(metrics_values['f1_score']*100))
        print('=================')
        print('Precision = %.2f'%(metrics_values['precision']*100))
        print('=================')
        print('Recall = %.2f'%(metrics_values['recall']*100))
        print('=================')
        print('Specificity = %.2f'%(metrics_values['specificity']*100))
        print('=================')
        print('')
        sys.stdout = original_stdout
        metrics_file.close()

    # This function receives 3 arguments: "predicted" which is the output
    # returned by the model in form of label (i.e., already thresholded)
    # "probabilities" that refers to the output of the model but, in this case,
    # to the probabilities that it returns and "target" which is the actual
    # output that the classifier needs to learn.
    def model_metrics(self, model_outputs, target):
        predicted, probabilities = model_outputs

        metrics_values = {}

        cm = confusion_matrix(y_true=target, y_pred=predicted)
        # Obtain the coefficients of the confusion_matrix.
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        specificity = tn / (tn + fp)
        f1_score = 2 * ((precision * recall)/(precision + recall))
        recall_specificity_tradeoff = recall - specificity

        # fpr, tpr, thresholds = roc_curve(y_true=target, y_score=probabilities[:, 1])
        fpr, tpr, _ = self.__custom_roc_curve__(target, probabilities[:, 1])
        # auc_roc = self.__custom_auc_function__(fpr, tpr, thresholds)
        auc_roc = auc(fpr, tpr)

        precision_list, recall_list, _ = precision_recall_curve(target, probabilities[:, 1])
        auc_pr = auc(recall_list, precision_list)

        metrics_values['accuracy'] = accuracy
        metrics_values['recall_specificity_tradeoff'] = recall_specificity_tradeoff
        metrics_values['f1_score'] = f1_score
        metrics_values['precision'] = precision
        metrics_values['specificity'] = specificity
        metrics_values['recall'] = recall
        metrics_values['auc_roc'] = auc_roc
        metrics_values['auc_pr'] = auc_pr
        metrics_values['confusion_matrix'] = cm

        return metrics_values

    # This function adds the headers to the logs csv file.
    def add_headers_to_csv_file(self, output_filename, \
                                                headers_list, append=True):
        if (append):
            file_mode = 'a'
        else:
            file_mode = 'w'

        with open(output_filename, file_mode) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')

            csv_writer.writerow(['performance_metrics_values'])
            csv_writer.writerow(['repetition'] + headers_list)

    # This function stores the current repetition followed by 1 row of
    # metrics_values, a variable that must be a list.
    def store_model_metrics(self, repetition, \
                            metrics_values_list, output_filename, append=True):
        if (append):
            file_mode = 'a'
        else:
            file_mode = 'w'

        with open(output_filename, file_mode) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')

            csv_writer.writerow([repetition] + metrics_values_list)

    # This function stores all the parameters related with the current experiment
    # (experiment name, dataset, classifier, number of features...).
    def store_experiment_parameters(self, args, output_filename, append=True):
        args_dict = vars(args)
        values_to_store_list = []
        for key_aux in args_dict.keys():
            if (args_dict[key_aux]==None):
                value_aux = 'None'
            else:
                value_aux = args_dict[key_aux]
            values_to_store_list.append([key_aux, value_aux])

        if (append):
            file_mode = 'a'
        else:
            file_mode = 'w'

        with open(output_filename, file_mode) as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')

            csv_writer.writerow([])
            csv_writer.writerow(['parameters'])
            for current_value_aux in values_to_store_list:
                csv_writer.writerow(current_value_aux)

    # This function reads the CSV file with the logs of the performance metrics
    # obtained in a certain experiment to compute the mean and the standard
    # deviation of the whole amount of repetitions.
    def compute_mean_and_std_performance(self, csv_log_file_path):
        accuracy_values_list = []
        recall_specificity_tradeoff_values_list = []
        f1_score_values_list = []
        precision_values_list = []
        specificity_values_list = []
        recall_values_list = []
        auc_roc_values_list = []
        auc_pr_values_list = []

        # Reading the performance metrics from file.
        with open(csv_log_file_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            # Number of rows since the reference header (in this case,
            # "performance_metrics_values" is found)
            reference_header_found = 0
            for row_aux in csv_reader:
                if (reference_header_found==1):
                    reference_header_found+=1
                elif (reference_header_found==2):
                    accuracy_values_list.append(row_aux[1])
                    recall_specificity_tradeoff_values_list.append(row_aux[2])
                    f1_score_values_list.append(row_aux[3])
                    precision_values_list.append(row_aux[4])
                    specificity_values_list.append(row_aux[5])
                    recall_values_list.append(row_aux[6])
                    auc_roc_values_list.append(row_aux[7])
                    auc_pr_values_list.append(row_aux[8])

                if ('performance_metrics_values' in row_aux):
                    reference_header_found+=1

        accuracy_values_list = np.array(accuracy_values_list).astype(np.float64)
        recall_specificity_tradeoff_values_list = np.array(recall_specificity_tradeoff_values_list).astype(np.float64)
        f1_score_values_list = np.array(f1_score_values_list).astype(np.float64)
        precision_values_list = np.array(precision_values_list).astype(np.float64)
        specificity_values_list = np.array(specificity_values_list).astype(np.float64)
        recall_values_list = np.array(recall_values_list).astype(np.float64)
        auc_roc_values_list = np.array(auc_roc_values_list).astype(np.float64)
        auc_pr_values_list = np.array(auc_pr_values_list).astype(np.float64)

        accuracy_summary = np.mean(accuracy_values_list), np.std(accuracy_values_list)
        recall_specificity_tradeoff_summary = np.mean(recall_specificity_tradeoff_values_list), np.std(recall_specificity_tradeoff_values_list)
        f1_score_summary = np.mean(f1_score_values_list), np.std(f1_score_values_list)
        precision_summary = np.mean(precision_values_list), np.std(precision_values_list)
        specificity_summary = np.mean(specificity_values_list), np.std(specificity_values_list)
        recall_summary = np.mean(recall_values_list), np.std(recall_values_list)
        auc_roc_summary = np.mean(auc_roc_values_list), np.std(auc_roc_values_list)
        auc_pr_summary = np.mean(auc_pr_values_list), np.std(auc_pr_values_list)

        # Writing the performance metrics summary (mean and standard deviation)
        # in csv file.
        with open(csv_log_file_path, 'a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')

            summary_headers = ['mean', 'std']
            csv_writer.writerow([])
            csv_writer.writerow(['metrics_summary'])
            for it in range(0, 2):
                csv_writer.writerow([summary_headers[it], accuracy_summary[it], recall_specificity_tradeoff_summary[it], \
                    f1_score_summary[it], precision_summary[it], specificity_summary[it], recall_summary[it], \
                         auc_roc_summary[it], auc_pr_summary[it]])

    def save_model(self, filename):
        pickle.dump(self.classifier, open(filename, 'wb'))

    def load_model(self, filename):
        self.classifier = pickle.load(open(filename, 'rb'))

    def get_roc_curve(self, model_outputs, target):
        predicted, probabilities = model_outputs
        roc_curve = self.__custom_roc_curve__(target, probabilities[:, 1])
        fpr, tpr, thresholds = roc_curve

        return fpr, tpr, thresholds

    # This function must be reimplemented by those classifiers with
    # explainability. By default, it displays a message that shows that the
    # used model has not explainability.
    def explainability(self):
        print('++++ WARNING: this model does not have explainability options')

class SVM_Classifier(Super_Classifier_Class):

    def __init__(self, **kwargs):
        print('++++ Creating an SVM classifier')
        self.classifier = svm.SVC(kernel='rbf', probability=True, class_weight='balanced')

    # The rest of the functions are inherited from the super class.

class kNN_Classifier(Super_Classifier_Class):

    def __init__(self, **kwargs):
        print('++++ Creating a kNN classifier')
        self.classifier = KNeighborsClassifier(n_neighbors=kwargs['n_neighbors'])

    # The rest of the functions are inherited from the super class.

class DT_Classifier(Super_Classifier_Class):

    def __init__(self, **kwargs):
        print('++++ Creating a DT classifier')
        self.classifier = tree.DecisionTreeClassifier(max_depth=3)

    def explainability(self):
        print('++++ The explainability of this model is available\n')
        rules = tree.export_text(self.classifier)
        print('##### Rules of the decision tree #####\n')
        print(rules)

    # The rest of the functions are inherited from the super class.

class MLP_Classifier(Super_Classifier_Class):

    def __init__(self, **kwargs):
        print('++++ Creating an MLP classifier')
        self.classifier = MLPClassifier(random_state=1, max_iter=300)

class XGBoost_Classifier(Super_Classifier_Class):

    def __init__(self, **kwargs):
        print('++++ Creating an XGBoost instance')
        self.classifier = XGBClassifier(use_label_encoder=False, booster='dart', eta=0.1)
        
    # The rest of the functions are inherited from the super class.
