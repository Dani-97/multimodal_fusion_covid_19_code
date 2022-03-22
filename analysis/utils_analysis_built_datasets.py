import argparse
import pandas as pd

class Super_Class_Analysis():

    def __init__(self):
        pass

    # This function uses the name of the attribute as the input as well as the
    # input dataframe.
    def __perform_individual_analysis__(self, attr, input_dataframe, name_class0, name_class1):
        # Total number of rows of the dataset.
        nof_rows = len(input_dataframe)

        nof_attr = len(input_dataframe.query('%s==1'%attr))
        nof_non_attr = len(input_dataframe.query('%s!=1'%attr))

        nof_attr_and_class0 = len(input_dataframe.query('(%s==1) and (output==0)'%attr))
        nof_attr_and_class1 = len(input_dataframe.query('(%s==1) and (output==1)'%attr))

        nof_non_attr_and_class0 = len(input_dataframe.query('(%s!=1) and (output==0)'%attr))
        nof_non_attr_and_class1 = len(input_dataframe.query('(%s!=1) and (output==1)'%attr))

        print('##################### STATISTICS OF %s ##########################'%attr)

        percentage_aux = (nof_attr/nof_rows)*100
        print('++++ Number of rows with %s set to Positive = [%d (%.2f %s)]'%(attr, nof_attr, percentage_aux, '%'))
        percentage_aux = (nof_non_attr/nof_rows)*100
        print('++++ Number of rows with %s set to Negative = [%d (%.2f %s)]'%(attr, nof_non_attr, percentage_aux, '%'))

        percentage_aux = (nof_attr_and_class0/nof_attr)*100
        print('++++ Number of samples with %s Positive and %s = [%d (%.2f %s)]'%(attr, \
                                    name_class0, nof_attr_and_class0, percentage_aux, '%'))

        percentage_aux = (nof_attr_and_class1/nof_attr)*100
        print('++++ Number of samples with %s Positive and %s = [%d (%.2f %s)]'%(attr, \
                                    name_class1, nof_attr_and_class1, percentage_aux, '%'))

        percentage_aux = (nof_non_attr_and_class0/nof_non_attr)*100
        print('++++ Number of samples with %s Negative and %s = [%d (%.2f %s)]'%(attr, \
                                    name_class0, nof_non_attr_and_class0, percentage_aux, '%'))

        percentage_aux = (nof_non_attr_and_class1/nof_non_attr)*100
        print('++++ Number of samples with %s Negative and %s = [%d (%.2f %s)]'%(attr, \
                                    name_class1, nof_non_attr_and_class1, percentage_aux, '%'))

        print('')

class Analysis_Only_Hospitalized(Super_Class_Analysis):

    def __init__(self):
        pass

    def execute_analysis(self, input_dataframe):
        attrs_list = ['hta', 'diabetes_mellitus', 'epoc', 'asma', 'hepatopatia', 'leucemia', \
            'linfoma', 'neoplasia', 'hiv', 'transplante_organo_solido', \
                'quimioterapia_ultimos_3_meses', 'biologicos_ultimos_3_meses', \
                    'corticoides_cronicos_mas_3_meses']
        name_class0, name_class1 = 'Non Exitus', 'Exitus'

        for attr_aux in attrs_list:
            super().__perform_individual_analysis__(attr_aux, input_dataframe, name_class0, name_class1)

class Analysis_Only_Hospitalized_Joint_Inmunosupression(Super_Class_Analysis):

    def __init__(self):
        pass

    def execute_analysis(self, input_dataframe):
        attrs_list = ['inmunosupression']
        name_class0, name_class1 = 'Non Exitus', 'Exitus'

        for attr_aux in attrs_list:
            super().__perform_individual_analysis__(attr_aux, input_dataframe, name_class0, name_class1)

class Analysis_Hospitalized_And_Urgencies(Super_Class_Analysis):

    def __init__(self):
        pass

    def execute_analysis(self, input_dataframe):
        attrs_list = ['hta', 'diabetes_mellitus', 'epoc', 'asma', 'hepatopatia', 'leucemia', \
            'linfoma', 'neoplasia', 'hiv', 'transplante_organo_solido', \
                'quimioterapia_ultimos_3_meses', 'biologicos_ultimos_3_meses', \
                    'corticoides_cronicos_mas_3_meses']
        name_class0, name_class1 = 'Hospitalized', 'Urgencies'

        for attr_aux in attrs_list:
            super().__perform_individual_analysis__(attr_aux, input_dataframe, name_class0, name_class1)
