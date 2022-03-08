import csv
import pickle
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# This is the parent of every classifier models that are going to be
# implemented for the purposes of this work.
class Super_Classifier_Class():

    def __init__(self, **kwargs):
        pass

    def train(self, input_data, output_data):
        self.classifier.fit(input_data, output_data)

    def test(self, input_data):
        self.classifier.predict(input_data)
        classifier_output = self.classifier.predict(input_data), \
                                 self.classifier.predict_proba(input_data)

        return classifier_output

    # This function receives 3 arguments: "predicted" which is the output
    # returned by the model in form of label (i.e., already thresholded)
    # "probabilities" that refers to the output of the model but, in this case,
    # to the probabilities that it returns and "target" which is the actual
    # output that the classifier needs to learn.
    def classification_metrics(self, predicted, probabilities, target):
        metrics_values = {}

        accuracy = accuracy_score(y_true=target, y_pred=predicted)
        metrics_values['accuracy'] = accuracy
        f1_score_value = f1_score(y_true=target, y_pred=predicted, average='micro')
        metrics_values['f1_score'] = f1_score_value
        precision = precision_score(y_true=target, y_pred=predicted, average='micro')
        metrics_values['precision'] = precision
        recall = recall_score(y_true=target, y_pred=predicted, average='micro')
        metrics_values['recall'] = recall
        auc_roc = roc_auc_score(y_true=target, y_score=probabilities, multi_class='ovr')
        metrics_values['auc_roc'] = auc_roc

        return metrics_values

    def print_classification_performance(self, metrics_values):
        print('##### Classification performance report #####')
        print('Accuracy = %.4f'%metrics_values['accuracy'])
        print('F1 Score = %.4f'%metrics_values['f1_score'])
        print('Precision = %.4f'%metrics_values['precision'])
        print('Recall = %.4f'%metrics_values['recall'])
        print('AUC-ROC = %.4f'%metrics_values['auc_roc'])

    # This function adds the headers to the logs csv file.
    # WARNING: this function overwrites everything from the original csv file!
    def add_headers_to_csv_file(self, output_filename):
        with open(output_filename, 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')

            csv_writer.writerow(['Repetition', 'Accuracy', 'F1-Score', 'Precision', 'Recall', 'AUC-ROC'])

    # This function stores the current repetition followed by 1 row of
    # metrics_values, a variable that must be a list. The new content will be
    # appended to the last row of the file.
    def store_classification_metrics(self, repetition, metrics_values_list, output_filename):
        with open(output_filename, 'a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')

            csv_writer.writerow([repetition] + metrics_values_list)

    def save_model(self, filename):
        pickle.dump(self.classifier, open(filename, 'wb'))

    def load_model(self, filename):
        self.classifier = pickle.load(open(filename, 'rb'))

class SVM_Classifier(Super_Classifier_Class):

    def __init__(self, **kwargs):
        print('++++ Creating an SVM classifier')
        self.classifier = svm.SVC(probability=True)

    # The rest of the functions are inherited from the super class.

class kNN_Classifier(Super_Classifier_Class):

    def __init__(self, **kwargs):
        print('++++ Creating a kNN classifier')
        self.classifier = KNeighborsClassifier(n_neighbors=kwargs['n_neighbors'])

    # The rest of the functions are inherited from the super class.
