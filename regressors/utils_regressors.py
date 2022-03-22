import csv
import numpy as np
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# This is the parent of every regression models that are going to be
# implemented for the purposes of this work.
class Super_Regressor_Class():

    def __init__(self, **kwargs):
        pass

    def train(self, input_data, output_data):
        self.regressor.fit(input_data, output_data)

    def test(self, input_data):
        self.regressor.predict(input_data)
        regressor_output = self.regressor.predict(input_data)

        return regressor_output

    def model_metrics(self, predicted, target):
        metrics_values = {}

        mae_values = np.abs(np.subtract(predicted, target))
        mse_values = np.square(np.subtract(predicted, target))

        metrics_values['mean_mae'] = np.mean(mae_values)
        metrics_values['std_mae'] = np.std(mae_values)
        metrics_values['mean_mse'] = np.mean(mse_values)
        metrics_values['std_mse'] = np.std(mse_values)

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

    def compute_mean_and_std_performance(self, csv_log_file_path):
        pass

    def save_model(self, filename):
        pickle.dump(self.regressor, open(filename, 'wb'))

    def load_model(self, filename):
        self.regressor = pickle.load(open(filename, 'rb'))

    def explainability(self):
        print('++++ WARNING: this model does not have explainability options')

class SVM_Regressor(Super_Regressor_Class):

    def __init__(self, **kwargs):
        print('++++ Creating an SVM regression')
        self.regressor = SVR(kernel='linear')

    # The rest of the methods are inherited from the parent class.

class DT_Regressor(Super_Regressor_Class):

    def __init__(self, **kwargs):
        print('++++ Creating a DT regression')
        self.regressor = DecisionTreeRegressor()

class Linear_Regressor(Super_Regressor_Class):

    def __init__(self, **kwargs):
        print('++++ Creating a Linear regression')
        self.regressor = linear_model.LinearRegression()

    # The rest of the methods are inherited from the parent class.
