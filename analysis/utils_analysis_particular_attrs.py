import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

class Analysis_Covariances():

    def __init__(self):
        pass

    def __remove_missing_values_idxs__(self, input_list):
        indexes_list = []

        it = 0
        for item_aux in input_list:
            if (item_aux!=-1):
                indexes_list.append(it)
            it+=1

        return indexes_list

    def execute_analysis(self, input_dataframe, \
                                           dir_to_store_analysis, attrs_list):
        var1 = np.array(input_dataframe[attrs_list[0]])
        var2 = np.array(input_dataframe[attrs_list[1]])

        var1_ok_idxs = self.__remove_missing_values_idxs__(var1.tolist())
        var2_ok_idxs = self.__remove_missing_values_idxs__(var2.tolist())

        # This variable stores all the indexes without missing values.
        ok_idxs = list(set(var1_ok_idxs).intersection(var2_ok_idxs))

        output_array = np.array(input_dataframe['output'].tolist()).reshape(1, -1)
        class0_idxs = list(set(np.where(output_array==0)[1].tolist()).intersection(ok_idxs))
        class1_idxs = list(set(np.where(output_array==1)[1].tolist()).intersection(ok_idxs))

        plt.xlabel(attrs_list[0])
        plt.ylabel(attrs_list[1])

        regression_obj = LinearRegression().fit(var1[ok_idxs].reshape(-1, 1), \
                                                var2[ok_idxs].reshape(-1, 1))

        min_value = 0
        distance_max_min = np.max(var1)-min_value
        regression_x_values = np.arange(min_value, np.max(var1), \
                                            step=distance_max_min/len(var1))
        coefficient = regression_obj.coef_[0]
        intercept = regression_obj.intercept_
        regression_func = coefficient*regression_x_values + intercept

        plt.title('Covariance between %s and %s (without missing values)'%\
                                  (attrs_list[0], attrs_list[1]))
        plt.plot(regression_x_values, regression_func, 'k--')
        plt.scatter(var1[class0_idxs], var2[class0_idxs])
        plt.scatter(var1[class1_idxs], var2[class1_idxs])
        plt.legend(['Regression line', 'Class 0', 'Class 1'])
        output_filename = '%s/%s'%(dir_to_store_analysis, \
                'covariance_analysis_%s_%s.pdf'%(attrs_list[0], attrs_list[1]))
        plt.savefig(output_filename)

        print('++++ The covariance study has been stored at %s'%output_filename)
