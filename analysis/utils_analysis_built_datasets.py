import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import sys

class Super_Class_Analysis():

    def __init__(self):
        pass

    # This function assumes that, if you have an attribute whose values can be
    # -1, 0 and 1 and you specifiy that you want to compute the number of
    # values ==1 and !=1, the latter case must be considered as 0.
    def __obtain_pie_charts__(self, attr, input_dataframe, \
                possible_output_values_hr, possible_output_values_conditions, \
                    dir_to_store_analysis):

        def remove_comparators(input):
            if (input.find('!=')!=-1):
                return 0
            else:
                return int(input.replace('==', ''))

        # 'possible_outputs_values_conditions' contains elements like
        # '==0', '!=0', for example, so we need to remove both '=='.
        possible_output_values_numerical = \
            list(map(remove_comparators, possible_output_values_conditions))

        nof_possible_values = len(possible_output_values_hr)
        fig, ax = plt.subplots()

        values_distribution_list = []
        labels = []

        it = 0
        for output_values_aux in possible_output_values_conditions:
            tmp_value = len(input_dataframe.query('(%s%s)'%(attr, output_values_aux))[attr])
            values_distribution_list.append(tmp_value)
            it+=1

        it = 0
        total_count = np.sum(np.array(values_distribution_list))
        for current_label_aux in possible_output_values_hr:
            current_percentage_tmp = \
                (values_distribution_list[it]/total_count)*100
            labels.append(current_label_aux + ': %d (%.2f %s)'%\
                (values_distribution_list[it], current_percentage_tmp, '%'))
            it+=1

        patches, texts = plt.pie(values_distribution_list)
        plt.legend(patches, labels, loc="best")

        output_figure_path = dir_to_store_analysis + '/pie_' + attr + '.pdf'
        plt.savefig(output_figure_path)

    def __obtain_histograms_numerical_attrs__(self, attrs_list, input_dataframe, dir_to_store_analysis):
        print('++++ The histograms of the attributes will be stored at %s'%dir_to_store_analysis)
        for attr_aux in attrs_list:
            current_values = np.array(input_dataframe[attr_aux])
            current_values = np.sort(current_values[np.where(current_values!=-1)])
            plt.figure()
            plt.title('Values histogram of attribute: %s'%attr_aux)
            plt.hist(current_values, bins=80)
            plt.savefig('%s/hist_%s.pdf'%(dir_to_store_analysis, attr_aux))

    def __perform_numerical_attrs_analysis__(self, attrs_list, input_dataframe, dir_to_store_analysis):
        self.__obtain_histograms_numerical_attrs__(attrs_list, input_dataframe, \
                                                       dir_to_store_analysis)

        output_filename = dir_to_store_analysis + '/numerical_attrs_analysis.md'
        # Move the standard output to a file.
        sys.stdout = open(output_filename, 'w')

        print('|      | Mean | Std | Median | Min | Max |')
        print('|:----:|:----:|:------:|:---:|:------:|:---:|')
        for attr_aux in attrs_list:
            # It is important to remove all the missing values from the
            # statistics.
            current_values = np.array(input_dataframe[attr_aux])
            current_values = current_values[np.where(current_values!=-1)]
            mean_value = np.mean(current_values)
            std_value = np.std(current_values)
            median_value = np.median(current_values)
            min_value = np.min(current_values)
            max_value = np.max(current_values)
            print('| %s | %.2f | %.2f | %d | %.2f | %.2f |'%\
                (attr_aux, mean_value, std_value, median_value, \
                    min_value, max_value))

        sys.stdout.close()

    def __perform_individual_analysis_markdown__(self, attr, input_dataframe, \
                    name_classes, conditions_list, possible_output_values, \
                        attr_possible_values, dir_to_store_analysis):
        # Total number of rows of the dataset.
        nof_rows = len(input_dataframe)
        # Total number of classes of the dataset (based on the number of
        # names contained in the 'name_classes' list).
        nof_classes = len(name_classes)
        # Number of possible values that the current attribute can have.
        nof_possible_attr_values = len(attr_possible_values)

        # Number of items that fulfill each condition of the conditions list.
        nof_fulfill = []
        # Number of items that fulfill each condition of the conditions list
        # given each dataset class.
        nof_fulfill_per_class = []
        it = 0
        for condition_aux in conditions_list:
            nof_fulfill.append(len(input_dataframe.query('%s%s'%(attr, condition_aux))))
            # This variable only stores the fulfill counts for the current variable.
            fulfill_tmp = []
            for current_output_value_aux in possible_output_values:
                query_str_aux = '(%s%s) and (%s)'%(attr, condition_aux, current_output_value_aux)
                fulfill_tmp.append(len(input_dataframe.query(query_str_aux)))
            nof_fulfill_per_class.append(fulfill_tmp)
            it+=1

        print('| **%s** |'%attr, end='')
        print(' |'*nof_possible_attr_values)
        print('|:----------------:|', end='')
        print(':----------------:|'*nof_possible_attr_values)

        print('| |', end='')
        for column_aux in range(0, nof_possible_attr_values):
            if (nof_rows!=0):
                current_percent = (nof_fulfill[column_aux]/nof_rows)*100
            else:
                current_percent = 0.0
            print(' **%s** (%d [%.2f %s]) |'%(attr_possible_values[column_aux], \
                nof_fulfill[column_aux], current_percent, '%'), end='')
        print('')

        for row_aux in range(0, nof_classes):
            print('| **%s** | '%name_classes[row_aux], end='')
            for column_aux in range(0, nof_possible_attr_values):
                # This variables (numerator and denominator) are created only
                # for aesthetic reasons, to improve the code reading.
                numerator = np.array(nof_fulfill_per_class)[column_aux, row_aux]
                denominator = np.array(nof_fulfill)[column_aux]
                if (denominator!=0):
                    current_percent = (numerator/denominator)*100
                else:
                    current_percent = 0.0
                print(' %d [%.2f %s] |'%(numerator, current_percent, '%'), end='')
            print('')

        print('')

    def __execute_analysis_aux__(self, attrs_list, input_dataframe, \
                    name_classes, conditions_list, possible_output_values, \
                                  attrs_possible_values_hr, dir_to_store_analysis):
        output_filename = dir_to_store_analysis + '/correlations_study.md'
        # Move the standard output to a file.
        sys.stdout = open(output_filename, 'w')

        it = 0
        for attr_aux in attrs_list:
            self.__obtain_pie_charts__(attr_aux, input_dataframe, \
                              attrs_possible_values_hr[it], conditions_list[it], \
                                  dir_to_store_analysis)
            self.__perform_individual_analysis_markdown__(attr_aux, \
                          input_dataframe, name_classes, conditions_list[it], \
                              possible_output_values[it], attrs_possible_values_hr[it], \
                                  dir_to_store_analysis)
            it+=1

        sys.stdout.close()

class Analysis_Only_Hospitalized(Super_Class_Analysis):

    def __init__(self):
        pass

    def execute_analysis(self, input_dataframe, dir_to_store_analysis):
        attrs_list = ['hta', 'diabetes_mellitus', 'epoc', 'asma', 'hepatopatia', 'leucemia', \
            'linfoma', 'neoplasia', 'hiv', 'transplante_organo_solido', \
                'quimioterapia_ultimos_3_meses', 'biologicos_ultimos_3_meses', \
                    'corticoides_cronicos_mas_3_meses']
        # This will compare the attribute in the way of the following example:
        # 'attr!=1' and 'attr==1'.
        conditions_list = [['!=1', '==1']]*13
        possible_output_values = [['output==0', 'output==1']]*13
        # 'hr' refers to 'human readable'.
        attrs_possible_values_hr = [['No', 'Yes']]*13
        name_classes = ['Non Exitus', 'Exitus']

        super().__execute_analysis_aux__(attrs_list, input_dataframe, \
                        name_classes, conditions_list, possible_output_values, \
                                      attrs_possible_values_hr, dir_to_store_analysis)

class Analysis_Only_Hospitalized_Numerical_Variables(Super_Class_Analysis):

    def __init__(self):
        pass

    def execute_analysis(self, input_dataframe, dir_to_store_analysis):
        attrs_list = ['edad', 'talla_cm', 'peso_kg', 'imc', \
                         'linfocitos', 'linfocitos_porcentaje', 'dimeros_d', \
                             'ldh', 'creatinina', 'filtrado_glomerular_estimado', \
                                 'prc', 'ferritina', 'il6']

        super().__perform_numerical_attrs_analysis__(attrs_list, \
                                    input_dataframe, dir_to_store_analysis)

class Analysis_Only_Hospitalized_Discretized_Clinical_Data(Super_Class_Analysis):

    def __init__(self):
        pass

    def execute_analysis(self, input_dataframe, dir_to_store_analysis):
        attrs_list = ['linfocitos', 'dimeros_d', 'ldh', 'creatinina', \
            'filtrado_glomerular_estimado', 'prc', 'ferritina', 'il6']
        conditions_list = [['==-1', '==0', '==1', '==2'], ['==-1', '==0', '==1'], \
            ['==-1', '==0', '==1', '==2'], ['==-1', '==0', '==1', '==2'], ['==-1', '==0', '==1', '==2'], \
                ['==-1', '==0', '==1', '==2'], ['==-1', '==0', '==1', '==2'], ['==-1', '==0', '==1']]
        possible_output_values = [['output==0', 'output==1']]*8
        # 'hr' refers to 'human readable'.
        attrs_possible_values_hr = [['Missing', 'Low', 'Normal', 'High'], ['Missing', 'High', 'Normal'], \
            ['Missing', 'Low', 'Normal', 'High'], ['Missing', 'Low', 'Normal', 'High'], \
                ['Missing', 'Severe Renal Insufficiency', 'Renal Insufficiency', 'Normal'], \
                    ['Missing', 'Low', 'Normal', 'High'], ['Missing', 'Low', 'Normal', 'High'], \
                        ['Missing', 'Normal', 'High']]
        name_classes = ['Non Exitus', 'Exitus']

        super().__execute_analysis_aux__(attrs_list, input_dataframe, \
                        name_classes, conditions_list, possible_output_values, \
                                      attrs_possible_values_hr, dir_to_store_analysis)

class Analysis_Only_Hospitalized_Joint_Inmunosupression(Super_Class_Analysis):

    def __init__(self):
        pass

    def execute_analysis(self, input_dataframe, dir_to_store_analysis):
        attrs_list = ['inmunosupression']
        # This will compare the attribute in the way of the following example:
        # 'attr!=1' and 'attr==1'.
        conditions_list = [['!=1', '==1']]
        # 'hr' refers to 'human readable'.
        attrs_possible_values_hr = [['No', 'Yes']]
        possible_output_values = [['output==0', 'output==1']]
        name_classes = ['Non Exitus', 'Exitus']

        super().__execute_analysis_aux__(attrs_list, input_dataframe, \
                        name_classes, conditions_list, possible_output_values, \
                                      attrs_possible_values_hr, dir_to_store_analysis)

class Analysis_Hospitalized_And_Urgencies(Super_Class_Analysis):

    def __init__(self):
        pass

    def execute_analysis(self, input_dataframe, dir_to_store_analysis):
        attrs_list = ['hta', 'diabetes_mellitus', 'epoc', 'asma', 'hepatopatia', 'leucemia', \
            'linfoma', 'neoplasia', 'hiv', 'transplante_organo_solido', \
                'quimioterapia_ultimos_3_meses', 'biologicos_ultimos_3_meses', \
                    'corticoides_cronicos_mas_3_meses']
        # This will compare the attribute in the way of the following example:
        # 'attr!=1' and 'attr==1'.
        conditions_list = [['!=1', '==1']]*13
        # 'hr' refers to 'human readable'.
        attrs_possible_values_hr = [['No', 'Yes']]*13
        possible_output_values = [['output==0', 'output==1']]*13
        name_classes = ['Hospitalized', 'Urgencies']

        super().__execute_analysis_aux__(attrs_list, input_dataframe, \
                        name_classes, conditions_list, possible_output_values, \
                                      attrs_possible_values_hr, dir_to_store_analysis)

class Analysis_Hospitalized_And_Urgencies_Numerical_Variables(Super_Class_Analysis):

    def __init__(self):
        pass

    def execute_analysis(self, input_dataframe, dir_to_store_analysis):
        attrs_list = ['edad', 'talla_cm', 'peso_kg', 'imc', \
                         'linfocitos', 'linfocitos_porcentaje', 'dimeros_d', \
                             'ldh', 'creatinina', 'filtrado_glomerular_estimado', \
                                 'prc', 'ferritina', 'il6']

        super().__perform_numerical_attrs_analysis__(attrs_list, \
                                    input_dataframe, dir_to_store_analysis)
