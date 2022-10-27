import csv
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn import svm, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from xgboost import XGBClassifier

# This is the parent of every classifier models that are going to be
# implemented for the purposes of this work.
class Super_Classifier_Class():

    def __init__(self, **kwargs):
        pass

    def train(self, input_data, output_data):
        self.classifier.fit(input_data, output_data)

    def test(self, input_data):
        classifier_output = self.classifier.predict(input_data), \
                                 self.classifier.predict_proba(input_data)

        return classifier_output

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
        auc_roc = roc_auc_score(y_true=target, y_score=probabilities[:, 1])

        metrics_values['accuracy'] = accuracy
        metrics_values['f1_score'] = f1_score
        metrics_values['precision'] = precision
        metrics_values['specificity'] = specificity
        metrics_values['recall'] = recall
        metrics_values['auc_roc'] = auc_roc
        metrics_values['confusion_matrix'] = cm

        return metrics_values

    def save_roc_curve(self, target, predicted, output_filename):
        fpr, tpr, _ = roc_curve(target, predicted[1][:, 1])
        roc_auc_value = auc(fpr, tpr)

        np.save(output_filename, (fpr, tpr))


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

    # This function reads the CSV file with the logs of the performance metrics
    # obtained in a certain experiment to compute the mean and the standard
    # deviation of the whole amount of repetitions.
    def compute_mean_and_std_performance(self, csv_log_file_path):
        accuracy_values_list = []
        f1_score_values_list = []
        precision_values_list = []
        specificity_values_list = []
        recall_values_list = []
        auc_roc_values_list = []

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
                    f1_score_values_list.append(row_aux[2])
                    precision_values_list.append(row_aux[3])
                    specificity_values_list.append(row_aux[4])
                    recall_values_list.append(row_aux[5])
                    auc_roc_values_list.append(row_aux[6])

                if ('performance_metrics_values' in row_aux):
                    reference_header_found+=1

        accuracy_values_list = np.array(accuracy_values_list).astype(np.float64)
        f1_score_values_list = np.array(f1_score_values_list).astype(np.float64)
        precision_values_list = np.array(precision_values_list).astype(np.float64)
        specificity_values_list = np.array(specificity_values_list).astype(np.float64)
        recall_values_list = np.array(recall_values_list).astype(np.float64)
        auc_roc_values_list = np.array(auc_roc_values_list).astype(np.float64)

        accuracy_summary = np.mean(accuracy_values_list), np.std(accuracy_values_list)
        f1_score_summary = np.mean(f1_score_values_list), np.std(f1_score_values_list)
        precision_summary = np.mean(precision_values_list), np.std(precision_values_list)
        specificity_summary = np.mean(specificity_values_list), np.std(specificity_values_list)
        recall_summary = np.mean(recall_values_list), np.std(recall_values_list)
        auc_roc_summary = np.mean(auc_roc_values_list), np.std(auc_roc_values_list)

        # Writing the performance metrics summary (mean and standard deviation)
        # in csv file.
        with open(csv_log_file_path, 'a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')

            summary_headers = ['mean', 'std']
            csv_writer.writerow([])
            csv_writer.writerow(['metrics_summary'])
            for it in range(0, 2):
                csv_writer.writerow([summary_headers[it], accuracy_summary[it], f1_score_summary[it], \
                     precision_summary[it], specificity_summary[it], recall_summary[it], \
                         auc_roc_summary[it]])

    def save_model(self, filename):
        pickle.dump(self.classifier, open(filename, 'wb'))

    def load_model(self, filename):
        self.classifier = pickle.load(open(filename, 'rb'))

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
