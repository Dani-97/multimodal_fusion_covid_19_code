import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

class Analysis_Dispersion_1D():

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

    def execute_analysis_scatter(self, input_dataframe, \
                                           dir_to_store_analysis, attrs_list):
        var1 = np.array(input_dataframe[attrs_list[0]])

        var1_ok_idxs = self.__remove_missing_values_idxs__(var1.tolist())

        # This variable stores all the indexes without missing values.
        ok_idxs = var1_ok_idxs

        output_array = np.array(input_dataframe['output'].tolist()).reshape(1, -1)
        class0_idxs = list(set(np.where(output_array==0)[1].tolist()).intersection(ok_idxs))
        class1_idxs = list(set(np.where(output_array==1)[1].tolist()).intersection(ok_idxs))

        fig, ax = plt.subplots()

        plt.ylabel(attrs_list[0])
        ax.set_xticklabels([])

        plt.title('Dispersion of %s (without missing values)'%\
                                  (attrs_list[0]))
        plt.plot([-1, 0, 1], [np.mean(var1[ok_idxs])]*3, 'k--')
        plt.scatter([0]*len(class0_idxs), var1[class0_idxs], s=10, linewidths=0.7, marker='x')
        plt.scatter([0]*len(class1_idxs), var1[class1_idxs], s=10, linewidths=0.7, marker='x')
        plt.legend(['Regression line (mean)', 'Class 0', 'Class 1'])
        output_filename = '%s/%s'%(dir_to_store_analysis, \
                'dispersion_analysis_%s.pdf'%(attrs_list[0]))
        plt.savefig(output_filename)

        print('++++ The dispersion study has been stored at %s'%output_filename)

    def execute_analysis(self, input_dataframe, dir_to_store_analysis, \
                                                         attrs_list, nbins=6):
        var1 = np.array(input_dataframe[attrs_list[0]])

        var1_ok_idxs = self.__remove_missing_values_idxs__(var1.tolist())

        # This variable stores all the indexes without missing values.
        ok_idxs = var1_ok_idxs

        output_array = np.array(input_dataframe['output'].tolist()).reshape(1, -1)
        class0_idxs = list(set(np.where(output_array==0)[1].tolist()).intersection(ok_idxs))
        class1_idxs = list(set(np.where(output_array==1)[1].tolist()).intersection(ok_idxs))

        fig, ax = plt.subplots()
        patches = ax.patches

        var1_class0_for_hist, _ = np.histogram(np.sort(var1[class0_idxs]), bins=nbins)
        var1_class1_for_hist, _ = np.histogram(np.sort(var1[class1_idxs]), bins=nbins)

        max_value = np.max(var1[ok_idxs])
        min_value = np.min(var1[ok_idxs])
        step = (max_value-min_value)/nbins
        xlabels = []

        for it in range(0, nbins):
            min_lim = min_value+(it*step)
            max_lim = min_value+((it+1)*step)
            xlabels.append('%d-%d'%(min_lim, max_lim))

        legend = ['Class 0', 'Class 1']
        x_class0_values = np.array(list(range(0, len(var1_class0_for_hist)*2, 2)))
        plt.bar(x_class0_values, var1_class0_for_hist, width=0.5)
        x_class1_values = np.array(list(range(1, len(var1_class1_for_hist)*2, 2)))
        plt.bar(x_class1_values, var1_class1_for_hist, width=0.5)
        plt.xlabel(attrs_list[0])
        plt.legend(legend)

        bars_width = patches[0].get_width()
        plt.xticks(x_class0_values+bars_width, xlabels)

        output_filename = '%s/%s'%(dir_to_store_analysis, \
                'dispersion_analysis_%s.pdf'%(attrs_list[0]))
        plt.savefig(output_filename)

        print('++++ The dispersion study has been stored at %s'%output_filename)

class Analysis_Dispersion_2D():

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

    def execute_analysis(self, input_dataframe, dir_to_store_analysis, \
                                                        attrs_list, nbins=6):
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

        plt.title('Dispersion between %s and %s (without missing values)'%\
                                  (attrs_list[0], attrs_list[1]))
        plt.plot(regression_x_values, regression_func, 'k--')
        plt.scatter(var1[class0_idxs], var2[class0_idxs], s=10, linewidths=0.7, marker='x')
        plt.scatter(var1[class1_idxs], var2[class1_idxs], s=10, linewidths=0.7, marker='x')
        plt.legend(['Regression line', 'Class 0', 'Class 1'])
        output_filename = '%s/%s'%(dir_to_store_analysis, \
                'dispersion_analysis_%s_%s.pdf'%(attrs_list[0], attrs_list[1]))
        plt.savefig(output_filename)

        print('++++ The dispersion study has been stored at %s'%output_filename)
