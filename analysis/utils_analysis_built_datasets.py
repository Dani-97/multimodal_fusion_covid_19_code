import argparse
import numpy as np
import pandas as pd

class Super_Class_Analysis():

    def __init__(self):
        pass

    def __perform_individual_analysis_markdown__(self, attr, input_dataframe, \
                        name_classes, conditions_list, possible_output_values, attr_possible_values):
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

class Analysis_Only_Hospitalized(Super_Class_Analysis):

    def __init__(self):
        pass

    def execute_analysis(self, input_dataframe):
        attrs_list = ['hta', 'diabetes_mellitus', 'epoc', 'asma', 'hepatopatia', 'leucemia', \
            'linfoma', 'neoplasia', 'hiv', 'transplante_organo_solido', \
                'quimioterapia_ultimos_3_meses', 'biologicos_ultimos_3_meses', \
                    'corticoides_cronicos_mas_3_meses']
        # This will compare the attribute in the way of the following example:
        # 'attr!=1' and 'attr==1'.
        conditions_list = [['!=1', '==1']]*13
        possible_output_values = [['output==0', 'output==1']]*13
        # 'hr' refers to 'human readable'.
        attrs_possible_values_hr = [['No', 'Sí']]*13
        name_classes = ['Non Exitus', 'Exitus']

        it = 0
        for attr_aux in attrs_list:
            super().__perform_individual_analysis_markdown__(attr_aux, \
                          input_dataframe, name_classes, conditions_list[it], \
                              possible_output_values[it], attrs_possible_values_hr[it])
            it+=1

class Analysis_Only_Hospitalized_Discretized_Clinical_Data(Super_Class_Analysis):

    def __init__(self):
        pass

    def execute_analysis(self, input_dataframe):
        attrs_list = ['linfocitos', 'dimeros_d', 'ldh', 'creatinina', \
            'filtrado_glomerular_estimado', 'prc', 'ferritina', 'il6']
        conditions_list = [['==-1', '==0', '==1', '==2'], ['==-1', '==0', '==1'], \
            ['==-1', '==0', '==1', '==2'], ['==-1', '==0', '==1', '==2'], ['==-1', '==0', '==1', '==2'], \
                ['==-1', '==0', '==1', '==2'], ['==-1', '==0', '==1', '==2'], ['==-1', '==0', '==1']]
        possible_output_values = [['output==0', 'output==1']]*8
        # 'hr' refers to 'human readable'.
        attrs_possible_values_hr = [['Missing', 'L', 'Normal', 'H'], ['Missing', 'H', 'Normal'], \
            ['Missing', 'L', 'Normal', 'H'], ['Missing', 'L', 'Normal', 'H'], \
                ['Missing', 'Insuficiencia renal grave', 'Insuficiencia renal', 'Normal'], \
                    ['Missing', 'L', 'Normal', 'H'], ['Missing', 'L', 'Normal', 'H'], \
                        ['Missing', 'Normal', 'H']]
        name_classes = ['Non Exitus', 'Exitus']

        it = 0
        for attr_aux in attrs_list:
            super().__perform_individual_analysis_markdown__(attr_aux, \
                          input_dataframe, name_classes, conditions_list[it], \
                              possible_output_values[it], attrs_possible_values_hr[it])
            it+=1

class Analysis_Only_Hospitalized_Joint_Inmunosupression(Super_Class_Analysis):

    def __init__(self):
        pass

    def execute_analysis(self, input_dataframe):
        attrs_list = ['inmunosupression']
        # This will compare the attribute in the way of the following example:
        # 'attr!=1' and 'attr==1'.
        conditions_list = [['!=1', '==1']]
        # 'hr' refers to 'human readable'.
        attrs_possible_values_hr = [['No', 'Sí']]
        possible_output_values = [['output==0', 'output==1']]
        name_classes = ['Non Exitus', 'Exitus']

        it = 0
        for attr_aux in attrs_list:
            super().__perform_individual_analysis_markdown__(attr_aux, \
                          input_dataframe, name_classes, conditions_list[it], \
                              possible_output_values[it], attrs_possible_values_hr[it])
            it+=1

class Analysis_Hospitalized_And_Urgencies(Super_Class_Analysis):

    def __init__(self):
        pass

    def execute_analysis(self, input_dataframe):
        attrs_list = ['hta', 'diabetes_mellitus', 'epoc', 'asma', 'hepatopatia', 'leucemia', \
            'linfoma', 'neoplasia', 'hiv', 'transplante_organo_solido', \
                'quimioterapia_ultimos_3_meses', 'biologicos_ultimos_3_meses', \
                    'corticoides_cronicos_mas_3_meses']
        # This will compare the attribute in the way of the following example:
        # 'attr!=1' and 'attr==1'.
        conditions_list = [['!=1', '==1']]*13
        # 'hr' refers to 'human readable'.
        attrs_possible_values_hr = [['No', 'Sí']]*13
        possible_output_values = [['output==0', 'output==1']]*13
        name_classes = ['Hospitalized', 'Urgencies']

        it = 0
        for attr_aux in attrs_list:
            super().__perform_individual_analysis_markdown__(attr_aux, \
                          input_dataframe, name_classes, conditions_list[it], \
                              possible_output_values[it], attrs_possible_values_hr[it])
            it+=1
